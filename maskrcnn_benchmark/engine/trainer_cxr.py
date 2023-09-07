# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import sys
import os
import math
import time
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, broadcast_data, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.ema import ModelEma
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from .inference import inference
import pdb

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def evaluate(all_predictions, hierarchy = False, use_PNUprompt=False, icd=False):
    logits, labels = [], []
    for p in all_predictions:
        for k, v in p.items():
            logits.append(v[0]['logits'])
            labels.append(v[1])
    lg = [torch.cat(x) for x in zip(*logits)]
    labels = torch.cat(labels)
    lb = [labels] * len(lg)
    if hierarchy and not icd:
        lg[1] = lg[1][:,:-2]
        lg[2] = lg[2][:,-2:]
        lb[1] = lb[1][:,:-1]
        lb[2] = lb[2][:,-1:]
    auc = [roc_auc_score(l[l!=2],x[:,1::2][l!=2]) for x, l in zip(lg, lb)]
    pred = [torch.stack([x.argmax(dim=1) for x in x.split(2,dim=1)], dim=1 ) for x in lg]
    acc = [accuracy_score(l[l!=2],x[l!=2]) for x, l in zip(pred, lb)]
    f1 = [f1_score(l[l!=2],x[l!=2]) for x,l  in zip(pred, lb)]

    return auc, acc, f1 


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_data_loader=None,
        meters=None,
        zero_shot=False
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()

    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        patience_counter = 0
        previous_best = 0.0

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1
    for iteration, data in enumerate(data_loader, start_iter):

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        data['images'] = data['images'].to(device)
        # Freeze language backbone
        if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if hasattr(model, "module"):
                model.module.language_backbone.eval()
            else:
                model.language_backbone.eval()

        if cfg.SOLVER.USE_AMP:
            with autocast():
                loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

        # Adapt the weight decay: only support multiStepLR
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
            if milestone_target < len(scheduler.milestones):
                next_milestone = list(scheduler.milestones)[milestone_target]
            else:
                next_milestone = float('inf')
            if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                gamma = scheduler.gamma
                logger.info("Drop the weight decay by {}!".format(gamma))
                for param in optimizer.param_groups:
                    if 'weight_decay' in param:
                        param['weight_decay'] *= gamma
                # move the target forward
                milestone_target += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
        # if iteration % 1 == 0 or iteration == max_iter:
            #logger.info(
            if global_rank <= 0:
                print(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "wd: {wd:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
            if is_main_process():
                print("Evaluating")
            eval_result = 0.0
            model.eval()

            all_labels = sum(data_loader.dataset.label_prompt.values[:,:2].tolist(),[])
            level_labels = [all_labels] * len(cfg.MODEL.LABEL_EMBEDDING_DIM) 
            results_dict = {}
            cpu_device = torch.device("cpu")
            for i, data in enumerate(val_data_loader):
                with torch.no_grad():
                    data['images'] = data['images'].to(device)
                    output = model(data, text = all_labels, labels_prompts = level_labels )
                    output = {k: [y.to(cpu_device) for y in v] for k, v in output.items()}
                results_dict.update(
                    {i: (output, data['label'])}
                )
            all_predictions = all_gather(results_dict)
            if is_main_process():
                icd = True if "mimic-cxrv2-icd" in cfg.DATASETS.TRAIN else False
                res = evaluate(all_predictions, hierarchy = cfg.MODEL.HIERARCHY, use_PNUprompt=cfg.MODEL.USE_PNUPROMPT, icd=icd)
                print(f'Evaluation on val, AUC={res[0]}, accuracy={res[1]}, f1={res[2]}')
                torch.save(res, os.path.join(cfg.OUTPUT_DIR,f'predictions_{iteration}.pkl'))
                eval_result = res[0][1]
            model.train()

            if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
                model_ema.ema.eval()
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, data in enumerate(val_data_loader):
                    with torch.no_grad():
                        data['images'] = data['images'].to(device)
                        output = model(data, text = all_labels, labels_prompts = level_labels)
                    results_dict.update(
                        {i: (output, data['label'])}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    icd = True if "mimic-cxrv2-icd" in cfg.DATASETS.TRAIN else False
                    res = evaluate(all_predictions, all_predictions, hierarchy = cfg.MODEL.HIERARCHY, use_PNUprompt=cfg.MODEL.USE_PNUPROMPT, icd=icd)
                    print(f'Evaluation on val, AUC={res[0]}, accuracy={res[1]}, f1={res[2]}')
                    torch.save(res, os.path.join(cfg.OUTPUT_DIR,f'predictions_{iteration}.pkl'))
                eval_result = res[0][1]
            arguments.update(eval_result=eval_result)

            if cfg.SOLVER.USE_AUTOSTEP:
                eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)
            
            if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
                if eval_result < previous_best:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    previous_best = eval_result
                    checkpointer.save("model_best", **arguments)
                print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)
                if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
                    if is_main_process():
                        print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
                    break

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )




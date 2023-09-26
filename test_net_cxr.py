# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from build_data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather, is_main_process
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.stats import get_model_complexity_info

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve

def evaluate(logit, labels, hierarchy = False,  softmax=False, merge_level=False,
            per_label = False, ppl =3, pos_pos = 1, use_thr=False, multi_class=False, average=True):
    lg = copy.deepcopy(logit)
    if softmax:
        lg = [x.reshape((-1,ppl)).softmax(dim=1).reshape(x.shape) for x in lg]
    lb = [labels] * len(lg)
    if multi_class:
        pred = [x[:,1::ppl].argmax(dim=1) for x in lg]
        lb = labels.argmax(dim=1)
        acc = [accuracy_score(lb, x) for x in pred]
        return acc

    if not hierarchy:
        lg.append(lg[1])
        lb.append(lb[1])

    assert len(lg)==3 and len(lb)==3
    if not merge_level:
        lg[1] = lg[1][:,:-ppl]
        lg[2] = lg[2][:,-ppl:]
        lb[1] = lb[1][:,:-1]
        lb[2] = lb[2][:,-1:]
    else:
        lg[1][:,-ppl:] = lg[2][:,-ppl:]
        lg = lg[:2]
        lb = lb[:2]

    if not use_thr:
        if not per_label:
            auc = [roc_auc_score(l[l!=2],x[:,pos_pos::ppl][l!=2]) for x, l in zip(lg, lb)]
            pred = [torch.stack([x.argmax(dim=1) for x in x.split(ppl,dim=1)], dim=1 ) for x in lg]
            for x in pred:
                x[x==2]=0
            acc = [accuracy_score(l[l!=2],x[l!=2]) for x, l in zip(pred, lb)]
            f1 = [f1_score(l[l!=2],x[l!=2]) for x,l  in zip(pred, lb)]
        else:
            pred = [torch.stack([x.argmax(dim=1) for x in x.split(ppl,dim=1)], dim=1 ) for x in lg]
            for x in pred:
                x[x==2]=0
            if average:
                auc = [np.mean([roc_auc_score(l[l!=2],x[l!=2]) for x,l in zip(xx[:,pos_pos::ppl].T, ll.T) if len(l.unique())>1 ]) for xx, ll in zip(lg, lb)]
                acc = [np.mean([accuracy_score(l[l!=2],x[l!=2]) for x,l in zip(xx.T,ll.T) if len(l.unique())>1  ]) for xx, ll in zip(pred, lb)]
                f1 = [np.mean([f1_score(l[l!=2],x[l!=2]) for x,l in zip(xx.T,ll.T) if len(l.unique())>1  ]) for xx,ll  in zip(pred, lb)]
            else:
                auc = [[roc_auc_score(l[l!=2],x[l!=2]) if len(l.unique())>1 else -1 for x,l in zip(xx[:,pos_pos::ppl].T, ll.T)  ] for xx, ll in zip(lg, lb)]
                acc = [[accuracy_score(l[l!=2],x[l!=2])  if len(l.unique())>1 else -1 for x,l in zip(xx.T,ll.T) ] for xx, ll in zip(pred, lb)]
                f1 = [[f1_score(l[l!=2],x[l!=2]) if len(l.unique())>1 else -1 for x,l in zip(xx.T,ll.T) ] for xx,ll  in zip(pred, lb)]
        return auc, acc, f1 
    else:
        lg = [x[:,pos_pos::ppl] for x in lg]
        if not per_label:
            auc = [roc_auc_score(l[l!=2],x[l!=2]) for x, l in zip(lg, lb)]
            prt = [precision_recall_curve(l[l!=2],x[l!=2]) for x, l in zip(lg, lb)]
            f1 = [2*p*r/(p+r) for p, r, t in prt]
            max_f1 = [np.nanmax(x) for x in f1]
            max_f1_thresh = [t[np.nanargmax(f)] for (_,_,t), f in zip(prt , f1)]
            acc = [accuracy_score(l[l!=2],x[l!=2]>t) for x, l, t in zip(lg, lb, max_f1_thresh)]
        else:
            prt = [[precision_recall_curve(l[l!=2],x[l!=2]) for x,l in zip(xx.T, ll.T) if len(l.unique())>1] for xx, ll in zip(lg, lb)]
            f1 = [[2*p*r/(p+r) for p, r, t in x] for x in prt]
            if average:
                auc = [np.mean([roc_auc_score(l[l!=2],x[l!=2]) for x,l in zip(xx.T, ll.T) if len(l.unique())>1 ] ) for xx, ll in zip(lg, lb)]
                max_f1 = [np.mean([np.nanmax(x) for x in xx]) for xx in f1]
                max_f1_thresh = [[t[np.nanargmax(f)] for (_,_,t), f in zip(x,y)] for x,y in zip(prt , f1)]
                acc = [np.mean([accuracy_score(l[l!=2],x[l!=2]>t) for x,l,t in zip(xx.T,ll.T, tt) if len(l.unique())>1 ]) for xx, ll, tt in zip(lg, lb, max_f1_thresh)]
            else:
                auc = [[roc_auc_score(l[l!=2],x[l!=2]) if len(l.unique())>1 else -1 for x,l in zip(xx.T, ll.T)  ]  for xx, ll in zip(lg, lb)]
                max_f1 = [[np.nanmax(x) for x in xx] for xx in f1]
                max_f1_thresh = [[t[np.nanargmax(f)] for (_,_,t), f in zip(x,y)] for x,y in zip(prt , f1)]
                acc = [[accuracy_score(l[l!=2],x[l!=2]>t)  if len(l.unique())>1 else -1 for x,l,t in zip(xx.T,ll.T, tt)  ] for xx, ll, tt in zip(lg, lb, max_f1_thresh)]
        return auc, acc, max_f1 



def run_test(cfg, model, data_loader_te, distributed, log_dir):
    if distributed and hasattr(model, 'module'):
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps

    all_labels = sum(data_loader_te.dataset.label_prompt.values.tolist(),[])
    level_labels = [all_labels] * len(cfg.MODEL.LABEL_EMBEDDING_DIM) 

    model.eval()
    results_dict = {}
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device("cpu")
    for i, data in tqdm(enumerate(data_loader_te)):
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
        logits, labels = [], []
        for p in all_predictions:
            for k, v in p.items():
                logits.append(v[0]['logits'])
                labels.append(v[1])
        lg = [torch.cat(x) for x in zip(*logits)]
        labels = torch.cat(labels)

        torch.save((lg, labels), os.path.join(log_dir, f'predictions_res_{cfg.DATASETS.TRAIN[0]}.pkl'))

        res = evaluate(lg,labels, hierarchy = cfg.MODEL.HIERARCHY,  softmax=True, 
                       merge_level=False, per_label = False, ppl=3)
        print(f'Evaluation on test, Micro, AUC, accuracy, f1')
        print(f'level 1 and level 2, [{res[0][1]}, {res[1][1]}, {res[2][1]}, {res[0][2]}, {res[1][2]}, {res[2][2]}]')

        res = evaluate(lg,labels, hierarchy = cfg.MODEL.HIERARCHY,  softmax=True,
                                merge_level=False,  per_label = True, ppl=3)
        print(f'Evaluation on test, Macro, AUC, accuracy, f1')
        print(f'level 1 and level 2, [{res[0][1]}, {res[1][1]}, {res[2][1]}, {res[0][2]}, {res[1][2]}, {res[2][2]}]')

        print('using threshod of max f1')
        res = evaluate(lg,labels, hierarchy = cfg.MODEL.HIERARCHY, softmax=False, 
                       merge_level=False, per_label = False, ppl=3, use_thr=True)
        print(f'Evaluation on test, Micro, AUC, accuracy, f1')
        print(f'level 1 and level 2, [{res[0][1]}, {res[1][1]}, {res[2][1]}, {res[0][2]}, {res[1][2]}, {res[2][2]}]')

        res = evaluate(lg,labels, hierarchy = cfg.MODEL.HIERARCHY,  softmax=False,
                                merge_level=False,  per_label = True, ppl=3, use_thr=True)
        print(f'Evaluation on test, Macro, AUC, accuracy, f1')
        print(f'level 1 and level 2, [{res[0][1]}, {res[1][1]}, {res[2][1]}, {res[0][2]}, {res[1][2]}, {res[2][2]}]')

        if "chexpert5x200" in cfg.DATASETS.TRAIN :
            res_MC = evaluate(lg,labels, hierarchy = cfg.MODEL.HIERARCHY,  softmax=True, 
                       merge_level=False, per_label = False, ppl=3, multi_class=True)
            print(f'multi_class accuracy with softmax: {res_MC}')
            res_MC = evaluate(lg,labels, hierarchy = cfg.MODEL.HIERARCHY,  softmax=False, 
                       merge_level=False, per_label = False, ppl=3, multi_class=True)
            print(f'multi_class accuracy without softmax: {res_MC}')
                



def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="config.yml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--data-path",
        default="./",
        help="path to data folder",
        type=str,
        )
    parser.add_argument(
        "--dataset",
        default="mimic-cxrv2",
        help="path to data folder",
        type=str,
        )
    parser.add_argument(
        "--weight",
        default='model_final',
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.defrost()
    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    model_path = os.path.dirname(args.weight)
    config_file = os.path.join(model_path,args.config_file)

    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.data_path = args.data_path
    cfg.TEST.IMS_PER_BATCH = 16
    cfg.MODEL.DEVICE = 'cuda:9'
    cfg.DATASETS.TRAIN = (args.dataset,)
    cfg.freeze() 

    log_dir = cfg.OUTPUT_DIR
    if args.weight:
        log_dir = os.path.join(log_dir, "eval", os.path.splitext(os.path.basename(args.weight))[0])
    if log_dir:
        mkdir(log_dir)
    logger = setup_logger("maskrcnn_benchmark", log_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)


    checkpointer = DetectronCheckpointer(cfg, model)
    if args.weight:
        _ = checkpointer.load(args.weight, force=True)
    else:
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

    data_loader, data_loader_val, data_loader_te  = make_data_loader(
        cfg,
        is_distributed = distributed,
        start_iter=0  # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )

    # print("Number of training iterations is {}".format(len(data_loader)))
    print("Number of test iterations is {}".format(len(data_loader_te)))

    run_test(cfg, model, data_loader_te, distributed, log_dir)


if __name__ == "__main__":
    main()

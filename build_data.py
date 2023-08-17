# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
import os

import torch.utils.data
import torch.distributed as dist
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

from maskrcnn_benchmark.data import samplers

from maskrcnn_benchmark.data.collate_batch import BatchCollator_cxr
from maskrcnn_benchmark.data.transforms import build_transforms

from transformers import AutoTokenizer
from maskrcnn_benchmark.data.datasets.cxr import MimicCXR_V2

def build_dataset(data_path, dataset='MimicCXR_V2', transforms=None):
    if dataset.lower() =='mimiccxr_v2':
        dataset = MimicCXR_V2(data_path,  transforms = transforms)
        tr_set, val_set, te_set = dataset.split()
        return tr_set, val_set, te_set


def make_data_sampler(dataset, shuffle, distributed, num_replicas=None, rank=None, use_random_seed=True):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas, rank=rank,
                                           use_random=use_random_seed)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0, drop_last=False
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=drop_last
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=drop_last
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def make_data_loader(cfg,  is_distributed=False, num_replicas=None, rank=None, start_iter=0):
    num_gpus = num_replicas or get_world_size()

    

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy 
    # aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    aspect_grouping =  []

    print("The combined datasets are: {}.".format(cfg.DATASETS.TRAIN ))

    transforms = build_transforms(cfg, is_train=False) 
    datasets = build_dataset(cfg.data_path, dataset='MimicCXR_V2', transforms=transforms)

    data_loaders = []
    for di, dataset in enumerate(datasets):
        is_train = di==0

        if is_train:
            images_per_batch = cfg.SOLVER.IMS_PER_BATCH
            assert (
                    images_per_batch % num_gpus == 0
            ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
            "of GPUs ({}) used.".format(images_per_batch, num_gpus)
            images_per_gpu = images_per_batch // num_gpus
            shuffle = True
            num_iters = cfg.SOLVER.MAX_ITER
        else:
            images_per_batch = cfg.TEST.IMS_PER_BATCH
            assert (
                    images_per_batch % num_gpus == 0
            ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
            "of GPUs ({}) used.".format(images_per_batch, num_gpus)
            images_per_gpu = images_per_batch // num_gpus
            shuffle = False if not is_distributed else True
            num_iters = None
            start_iter = 0

        if images_per_gpu > 1:
            logger = logging.getLogger(__name__)
            logger.warning(
                "When using more than one image per GPU you may encounter "
                "an out-of-memory (OOM) error if your GPU does not have "
                "sufficient memory. If this happens, you can reduce "
                "SOLVER.IMS_PER_BATCH (for training) or "
                "TEST.IMS_PER_BATCH (for inference). For training, you must "
                "also adjust the learning rate and schedule length according "
                "to the linear scaling rule. See for example: "
                "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
            )

        if is_train and cfg.SOLVER.MAX_EPOCH > 0:
            num_iters = cfg.SOLVER.MAX_EPOCH * len(dataset) // cfg.SOLVER.IMS_PER_BATCH
            print("Number of iterations are {}".format(num_iters))
            cfg.defrost()
            cfg.SOLVER.MAX_ITER = num_iters
            cfg.SOLVER.DATASET_LENGTH = len(dataset)
            cfg.freeze()
        if is_train and cfg.SOLVER.MULTI_MAX_EPOCH:
            num_iters = None
            cfg.defrost()
            cfg.SOLVER.MULTI_MAX_ITER += (cfg.SOLVER.MULTI_MAX_EPOCH[di] * len(dataset) // cfg.SOLVER.IMS_PER_BATCH,)
            cfg.freeze()

        if is_train and cfg.DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE:
            from .datasets.custom_distributed_sampler import DistributedSamplerChunkByNode
            chunk_or_not = []
            for i in dataset_list:
                if "bing_caption" in i:
                    chunk_or_not.append(True)
                else:
                    chunk_or_not.append(False)
            assert(len(chunk_or_not) == len(dataset.datasets))
            '''
            If we are training on 4 nodes, each with 8 GPUs
            '''
            num_nodes = int(os.getenv('NODE_COUNT', os.getenv('OMPI_COMM_WORLD_SIZE', 1)))
            local_size = cfg.num_gpus//num_nodes
            node_rank = int(os.getenv('NODE_RANK', os.getenv('OMPI_COMM_WORLD_RANK', 0)))
            local_rank = cfg.local_rank
            sampler = DistributedSamplerChunkByNode(
                dataset = dataset,
                all_datasets = dataset.datasets, # Assumming dataset is a ConcateDataset instance,
                chunk_or_not = chunk_or_not,
                num_replicas = cfg.num_gpus, # total GPU number, e.g., 32
                rank = dist.get_rank(), # Global Rank, e.g., 0~31
                node_rank = node_rank, # Node Rank, e.g., 0~3
                node_number = num_nodes, # how many node e.g., 4
                process_num_per_node = local_size, # e.g., 8
                rank_within_local_node = local_rank, # e.g., 0~7
            )
        else:
            sampler = make_data_sampler(dataset, shuffle, is_distributed, num_replicas=num_replicas, rank=rank,
                                        use_random_seed=cfg.DATALOADER.USE_RANDOM_SEED)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter, drop_last=is_train
        )
        collator =BatchCollator_cxr( cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)

    return data_loaders

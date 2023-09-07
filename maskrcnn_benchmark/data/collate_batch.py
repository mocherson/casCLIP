# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
import pandas as pd
from maskrcnn_benchmark.structures.image_list import to_image_list

import pdb
class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        positive_map = None
        positive_map_eval = None
        greenlight_map = None

        if isinstance(targets[0], dict):
            return images, targets, img_ids, positive_map, positive_map_eval

        if "greenlight_map" in transposed_batch[1][0].fields():
            greenlight_map = torch.stack([i.get_field("greenlight_map") for i in transposed_batch[1]], dim = 0)

        if "positive_map" in transposed_batch[1][0].fields():
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([v.get_field("positive_map").shape[1] for v in transposed_batch[1]])
            nb_boxes = sum([v.get_field("positive_map").shape[0] for v in transposed_batch[1]])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for v in transposed_batch[1]:
                cur_pos = v.get_field("positive_map")
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            positive_map = batched_pos_map.float()
        

        if "positive_map_eval" in transposed_batch[1][0].fields():
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([v.get_field("positive_map_eval").shape[1] for v in transposed_batch[1]])
            nb_boxes = sum([v.get_field("positive_map_eval").shape[0] for v in transposed_batch[1]])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for v in transposed_batch[1]:
                cur_pos = v.get_field("positive_map_eval")
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
            positive_map_eval = batched_pos_map.float()


        return images, targets, img_ids, positive_map, positive_map_eval, greenlight_map


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        # return list(zip(*batch))
        transposed_batch = list(zip(*batch))

        images = transposed_batch[0]
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        positive_map = None
        positive_map_eval = None

        if isinstance(targets[0], dict):
            return images, targets, img_ids, positive_map, positive_map_eval

        return images, targets, img_ids, positive_map, positive_map_eval


class BatchCollator_cxr(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, hierarchy=False, use_PNUprompt=False, size_divisible=0):
        self.hierarchy = hierarchy
        self.use_PNUprompt = use_PNUprompt
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = {key: [d[key] for d in batch] for key in batch[0] }

        transposed_batch['images'] = to_image_list(sum(transposed_batch['images'],[]), self.size_divisible)
        transposed_batch['label'] = torch.Tensor(np.nan_to_num(np.stack(transposed_batch['label']).astype(float)))

        if self.hierarchy and self.use_PNUprompt:
            transposed_batch['labels_prompts'] = [sum(transposed_batch['label_prompt'][0][0].values.tolist(), []), transposed_batch['label_prompt'][0][1].tolist()]
            transposed_batch['prompt_target'] = [ torch.LongTensor(x) for x in zip(*transposed_batch['prompt_target']) ]
        elif self.hierarchy and not self.use_PNUprompt:
            transposed_batch['labels_prompts'] = []
            transposed_batch['prompt_target'] = []
            for lp in zip(*transposed_batch['label_prompt']):
                prompt_df = pd.DataFrame([{s:1 for s in x} for x in lp]).fillna(0)
                transposed_batch['labels_prompts'].append( prompt_df.columns.tolist() )
                transposed_batch['prompt_target'].append(torch.Tensor(prompt_df.values) )
        elif not self.hierarchy and self.use_PNUprompt:
            prompt_df = pd.concat(transposed_batch['prompt_target'],axis=1).fillna(0).T
            transposed_batch['labels_prompts'] =[ sum([transposed_batch['label_prompt'][0].loc[x].tolist() for x in prompt_df.columns], [])]
            transposed_batch['prompt_target'] = [ torch.LongTensor(prompt_df.values) ]
        else:
            prompt_df = pd.DataFrame([{s:1 for s in x} for x in transposed_batch['label_prompt']]).fillna(0)
            transposed_batch['labels_prompts'] = [ prompt_df.columns.tolist() ]
            transposed_batch['prompt_target'] = [ torch.Tensor(prompt_df.values) ]

        return transposed_batch



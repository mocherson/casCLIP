import os
import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import glob
import pydicom
import pandas as pd
import numpy as np
import copy

import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList

from PIL import Image, ImageDraw
from torchvision.datasets.vision import VisionDataset
from skimage.color import gray2rgb



class MimicCXR_V2(VisionDataset):
    """MimicCXR_V2 dataset imported from TorchVision.
        It is modified to handle several image sources

    Args:
        root (string): Path to the dataset
        metafile (string): Path to meta data.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            metafile: str = 'cxr-study-list.csv',
            labelfile: str = 'mimic-cxr-2.0.0-chexpert.csv',
            splitfile: str = 'mimic-cxr-2.0.0-split.csv',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(MimicCXR_V2, self).__init__(root, transforms, transform, target_transform)

        meta_data = pd.read_csv(os.path.join(root, metafile))
        label_data = pd.read_csv(os.path.join(root, labelfile))
        self.split_data = pd.read_csv(os.path.join(root, splitfile))
        self.meta_data = meta_data.merge(label_data,on=['subject_id','study_id'], how='left')
        self.all_meta_data = self.meta_data
        self.label_prompt = ['no disease found' if x=='No Finding' else
                             f'{x} found' if x=='Support Devices' else
                             f'disease {x} found' for x in self.meta_data.columns[4:] ]


    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (images, target, index). images is a list of images. target includes caption
        """
        data = self.meta_data.iloc[index]
        subject_id = data['subject_id']
        study_id = data['study_id']
        text = data['text']
        image_folder = data['path'][:-4]
        image_files = glob.glob(os.path.join(self.root, image_folder)+'/*.jpg')
        label = data.iloc[4:]
        present = label[label==1].index
        # absent = label[label==1].index
        uncertain = label[label==-1].index
        # unmentioned = label[label.isna()].index
        label_prompt = []
        for x in present:
            if 'No Finding'==x:
                label_prompt.append('no disease found')
            elif 'Support Devices'==x:
                label_prompt.append('Support Devices found')
            else:
                label_prompt.append(f'disease {x} found')

        for x in uncertain:
            if 'Support Devices'==x:
                label_prompt.append('not sure if Support Devices found')
            else:
                label_prompt.append(f'not sure if disease {x} found')

        if len(label_prompt)==0:
            label_prompt = ['no disease found']

        n_prompt = len(label_prompt) 

        images = [Image.open(fn).convert("RGB") for fn in image_files]
        if self.transforms is not None:
            images = [self.transforms(img) for img in images]
        n_img = len(images)

        return {'images': images, 'text': text, 'label': label.values, 'label_prompt': label_prompt, 'n_img': n_img, 'n_prompt': n_prompt, 'index':index, 'study_id': study_id}

    def split(self):
        study_id_train = self.split_data.query('split=="train"')['study_id'].unique()
        study_id_val = self.split_data.query('split=="validate"')['study_id'].unique()
        study_id_test = self.split_data.query('split=="test"')['study_id'].unique()

        tr_df = self.all_meta_data.loc[self.all_meta_data['study_id'].isin(study_id_train)]
        val_df = self.all_meta_data.loc[self.all_meta_data['study_id'].isin(study_id_val)]
        te_df = self.all_meta_data.loc[self.all_meta_data['study_id'].isin(study_id_test)]

        tr_set, val_set, te_set = copy.copy(self), copy.copy(self), copy.copy(self)
        tr_set.meta_data, val_set.meta_data, te_set.meta_data  = tr_df, val_df, te_df

        return tr_set, val_set, te_set





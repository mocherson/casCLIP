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
            labelfile: str = 'mimic-cxr-2.0.0-chexpert_fill.csv',
            splitfile: str = 'mimic-cxr-2.0.0-split.csv',
            hierarchy: bool = False,
            use_PNUprompt: bool =False,
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
        self.label_prompt = pd.DataFrame ([  ['Disease Atelectasis is not found.', 'Disease Atelectasis is found.', 'Not sure if Disease Atelectasis is found.'],
                ['Disease Cardiomegaly is not found.', 'Disease Cardiomegaly is found.', 'Not sure if Disease Cardiomegaly is found.'], 
                ['Disease Consolidation is not found.', 'Disease Consolidation is found.', 'Not sure if Disease Consolidation is found.'], 
                ['Disease Edema is not found.', 'Disease Edema is found.', 'Not sure if Disease Edema is found.'], 
                ['Disease Enlarged Cardiomediastinum is not found.', 'Disease Enlarged Cardiomediastinum is found.', 'Not sure if Disease Enlarged Cardiomediastinum is found.'], 
                ['Disease Fracture is not found.', 'Disease Fracture is found.', 'Not sure if Disease Fracture is found.'], 
                ['Disease Lung Lesion is not found.', 'Disease Lung Lesion is found.', 'Not sure if Disease Lung Lesion is found.'], 
                ['Disease Lung Opacity is not found.', 'Disease Lung Opacity is found.', 'Not sure if Disease Lung Opacity is found.'], 
                ['Disease Pleural Effusion is not found.', 'Disease Pleural Effusion is found.', 'Not sure if Disease Pleural Effusion is found.'], 
                ['Pleural disease other than Effusion is not found.', 'Pleural disease other than Effusion is found.', 'Not sure if Pleural disease other than Effusion is found.'], 
                ['Disease Pneumonia is not found.', 'Disease Pneumonia is found.', 'Not sure if Disease Pneumonia is found.'], 
                ['Disease Pneumothorax is not found.', 'Disease Pneumothorax is found.', 'Not sure if Disease Pneumothorax is found.'], 
                ['Support Device is not found.', 'Support Device is found.', 'Not sure if Support Device is found.'],
               ['Chest Disease is found.', 'Chest Disease is not found.', 'Not sure if Chest Disease is found.']],
            index = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices','No Finding'],
            columns = ['negative', 'positive', 'uncertain'])

        self.hierarchy = hierarchy
        self.use_PNUprompt = use_PNUprompt


    def __len__(self):
        return len(self.meta_data)  

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        """
        data = self.meta_data.iloc[index]
        subject_id = data['subject_id']
        study_id = data['study_id']
        text = data['text']
        image_folder = data['path'][:-4]
        image_files = glob.glob(os.path.join(self.root, image_folder)+'/*.jpg')
        label = data.loc[self.label_prompt.index]
        # present = label[label==1].index
        # absent = label[label==0].index
        # uncertain = label[label==2].index
        # unmentioned = label[label.isna()].index
        if not self.hierarchy:
            label_prompt = []  
            prompt_target = []
            if not self.use_PNUprompt:        
                for x, v in label.items():
                    if v==1:
                        label_prompt.append(self.label_prompt.loc[x, 'positive'])
                    elif v==0:
                        label_prompt.append(self.label_prompt.loc[x, 'negative'])
                    elif v==2:
                        label_prompt.append(self.label_prompt.loc[x, 'uncertain'])
                    else:
                        continue

                if len(label_prompt)==0:
                    label_prompt = ['Not sure if Chest Disease is found.']
            else:
                prompt_target = label.dropna()
                label_prompt = self.label_prompt

                if len(prompt_target)==0: 
                    prompt_target = pd.Series({'No Finding':2.0})
        else:
            label_prompt = [[], []]  
            prompt_target = [[], []]
            if not self.use_PNUprompt:        
                for x, v in label.items():
                    if v==1:
                        if x == 'No Finding':
                            label_prompt[1].append(self.label_prompt.loc[x, 'positive'])
                        else:
                            label_prompt[0].append(self.label_prompt.loc[x, 'positive'])
                    elif v==0 or pd.isna(v):
                        if x == 'No Finding':
                            label_prompt[1].append(self.label_prompt.loc[x, 'negative'])
                        else:
                            label_prompt[0].append(self.label_prompt.loc[x, 'negative'])
                    elif v==2:
                        if x == 'No Finding':
                            label_prompt[1].append(self.label_prompt.loc[x, 'uncertain'])
                        else:
                            label_prompt[0].append(self.label_prompt.loc[x, 'uncertain'])
                    else:
                        pass
            else:
                label_prompt[0] = self.label_prompt.iloc[:-1]
                label_prompt[1] = self.label_prompt.iloc[-1]
                prompt_target[0] = label.drop('No Finding').fillna(0).values
                prompt_target[1] = label['No Finding'] 

        n_prompt = len(label_prompt) 

        images = [Image.open(fn).convert("RGB") for fn in image_files]
        if self.transforms is not None:
            images = [self.transforms(img) for img in images]
        n_img = len(images)

        return {'images': images, 'text': text, 'label': label.values, 'label_prompt': label_prompt, 'prompt_target':prompt_target, 
                'n_img': n_img, 'n_prompt': n_prompt, 'index':index, 'study_id': study_id}

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


class MimicCXR_ICD(MimicCXR_V2):
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
            metafile: str = 'cxr-study-list-icd.csv',
            labelfile: str = 'mimic-cxr-2.0.0-chexpert_fill.csv',
            splitfile: str = 'mimic-cxr-2.0.0-split.csv',
            icd_mapping: str = 'icd_mapping.csv',
            hierarchy: bool = False,
            use_PNUprompt: bool =False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(MimicCXR_ICD, self).__init__(root, metafile, labelfile, splitfile,hierarchy,use_PNUprompt, transform, target_transform, transforms)

        self.icd_map = pd.read_csv(os.path.join(root, icd_mapping), index_col=['icd_code', 'icd_version'])



    def __len__(self):
        return len(self.meta_data)  

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        """
        data = self.meta_data.iloc[index]
        subject_id = data['subject_id']
        study_id = data['study_id']
        text = data['text']
        image_folder = data['path'][:-4]
        image_files = glob.glob(os.path.join(self.root, image_folder)+'/*.jpg')
        label = data.loc[self.label_prompt.index]
        if pd.isna(data['icd_code']):
            icd_code = []
            icd_version = []
            label_prompt = [[], [], []]
        else:
            icd_code = eval(data['icd_code'])
            icd_version = eval(data['icd_version'])
            diag = pd.concat([self.icd_map.loc[x] for x in zip(icd_code,icd_version)], axis=1)
            label_prompt = diag.loc[['long_title','category_title','chapter_title']].values.tolist() 

        images = [Image.open(fn).convert("RGB") for fn in image_files]
        if self.transforms is not None:
            images = [self.transforms(img) for img in images]
        n_img = len(images)

        # present = label[label==1].index
        # absent = label[label==0].index
        # uncertain = label[label==2].index
        # unmentioned = label[label.isna()].index
        
        if not self.hierarchy:
            label_prompt = [sum(label_prompt,[]) ]  
            
        n_prompt = len(label_prompt) 

        

        return {'images': images, 'text': text, 'label': label.values, 'label_prompt': label_prompt, 
                'n_img': n_img, 'n_prompt': n_prompt, 'index':index, 'study_id': study_id}

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

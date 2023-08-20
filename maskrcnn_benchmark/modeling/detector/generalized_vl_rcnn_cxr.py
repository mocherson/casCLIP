# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized VL R-CNN framework
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, broadcast_data, get_rank

from ..backbone import build_backbone
from ..rpn import build_rpn
from ..roi_heads import build_roi_heads

from ..language_backbone import build_language_backbone
from transformers import AutoTokenizer

import random
import timeit
import pdb
from copy import deepcopy
from loss import ClipLoss, ClipLabelLoss

def random_word(input_ids, mask_token_id, vocabs, padding_token_id, greenlight_map):
    """
    greenlight_map, batch_size x 256 (seq_len):
        0 means this location cannot be calculated in the MLM loss
        -1 means this location cannot be masked!!
        1 means this location can be masked and can be calculated in the MLM loss
    """
    output_label = deepcopy(input_ids)
    for j in range(input_ids.size(0)):
        for i in range(input_ids.size(1)):
            prob = random.random()
            # mask token with probability
            ratio = 0.15
            if greenlight_map is not None and greenlight_map[j,i] == -1:
                output_label[j,i] = -100
                continue

            if (not input_ids[j,i] == padding_token_id) and prob < ratio:
                prob /= ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    input_ids[j,i] = mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    input_ids[j,i] = random.choice(vocabs)

            else:
                # no masking token (will be ignored by loss function later)
                output_label[j,i] = -100
            
            if greenlight_map is not None and greenlight_map[j,i] != 1:
                output_label[j,i] = -100 # If this location should not be masked
    return input_ids, output_label


class GeneralizedVLRCNN_CXR(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedVLRCNN_CXR, self).__init__()
        self.cfg = cfg

        # visual encoder
        self.backbone = build_backbone(cfg)

        # language encoder
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            # self.tokenizer = build_tokenizer("clip")
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [item for key, item in self.tokenizer_vocab.items()]

        self.language_backbone = build_language_backbone(cfg)
        self.image_agg = lambda x: x.max(dim=0)[0]
        self.visual_proj = nn.Linear(cfg.MODEL.BACKBONE.OUT_CHANNELS*5, self.language_backbone.body.language_dim)
        self.logit_scale = nn.Parameter(torch.ones([2]) * np.log(1 / 0.07))

        self.clip_loss = ClipLoss()
        self.label_loss = ClipLabelLoss()

        # self.label_prompt_backbone = build_language_backbone(cfg)
        self.label_level_proj = nn.Linear(self.language_backbone.body.language_dim, self.language_backbone.body.language_dim)



        self.DEBUG = cfg.MODEL.DEBUG

        self.freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        self.freeze_fpn = cfg.MODEL.FPN.FREEZE    
        self.freeze_language_backbone = self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE
        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.language_backbone.parameters():
                p.requires_grad = False
            # for p in self.label_prompt_backbone.parameters():
            #     p.requires_grad = False
        
        self.use_mlm_loss = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS 
        self.mlm_loss_for_only_positives = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GeneralizedVLRCNN_CXR, self).train(mode)
        if self.freeze_backbone:
            self.backbone.body.eval()
            for p in self.backbone.body.parameters():
                p.requires_grad = False
        if self.freeze_fpn:
            self.backbone.fpn.eval()
            for p in self.backbone.fpn.parameters():
                p.requires_grad = False

        if self.freeze_language_backbone:
            self.language_backbone.eval()
            for p in self.language_backbone.parameters():
                p.requires_grad = False

            # self.label_prompt_backbone.eval()
            # for p in self.label_prompt_backbone.parameters():
            #     p.requires_grad = False

    def forward(self, data, **kwargs):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

            mask_black_list: batch x 256, indicates whether or not a certain token is maskable or not

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        images = data['images'] if 'images' not in kwargs else kwargs['images']
        n_img = data['n_img'] if 'n_img' not in kwargs else kwargs['n_img']
        label_prompt = data['label_prompt'] if self.training else kwargs['label_prompt']
        captions = data['text'] if self.training else kwargs['text'] if 'text' in kwargs else None
        prompt_target = data['prompt_target']
        
        images = to_image_list(images)
        # batch_size = images.tensors.shape[0]
        device = images.tensors.device

        # language embedding
        language_dict_features = {}
        text_emb = None
        if captions is not None:
            #print(captions[0])
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                         max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                         padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                         return_special_tokens_mask=True,
                                                         return_tensors='pt',
                                                         truncation=True).to(device)
            if self.use_mlm_loss:
                if not self.mlm_loss_for_only_positives:
                    greenlight_map = None
                input_ids, mlm_labels = random_word(
                    input_ids=tokenized.input_ids, 
                    mask_token_id=self.tokenizer.mask_token_id,
                    vocabs=self.tokenizer_vocab_ids,
                    padding_token_id=self.tokenizer.pad_token_id,
                    greenlight_map=greenlight_map)
            else:
                input_ids = tokenized.input_ids
                mlm_labels = None
            
            
            tokenizer_input = {"input_ids": input_ids,
                               "attention_mask": tokenized.attention_mask}

            if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
                with torch.no_grad():
                    language_dict_features = self.language_backbone(tokenizer_input)
            else:
                language_dict_features = self.language_backbone(tokenizer_input)
            
            text_emb = F.normalize(language_dict_features['hidden'][:,0,:], dim=-1)

        # visual embedding
        swint_feature_c4 = None
        if 'vl' in self.cfg.MODEL.SWINT.VERSION:
            # the backbone only updates the "hidden" field in language_dict_features
            inputs = {"img": images.tensors, "lang": language_dict_features}
            visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
        else:
            visual_features = self.backbone(images.tensors)

        visual_features_agg = [torch.stack([self.image_agg(x) for x in torch.split(vf,n_img)]) for vf in visual_features]
        visual_features_agg = torch.cat([torch.nn.MaxPool2d(x.shape[-2:])(x).squeeze() for x in visual_features_agg], dim=1)
        visual_emb = F.normalize(self.visual_proj(visual_features_agg), dim=-1)

        #  encode label prompt
        label_tokenized = self.tokenizer.batch_encode_plus(label_prompt,
                                                         max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                         padding="longest",
                                                         return_special_tokens_mask=True,
                                                         return_tensors='pt',
                                                         truncation=True).to(device)            
            
        label_tokenizer_input = {"input_ids": label_tokenized.input_ids,
                           "attention_mask": label_tokenized.attention_mask}

        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            with torch.no_grad():
                label_dict_features = self.language_backbone(label_tokenizer_input)
        else:
            label_dict_features = self.language_backbone(label_tokenizer_input)
        
        label_emb = F.normalize(label_dict_features['hidden'][:,0,:], dim=-1)
        visual_emb1 = F.normalize(self.label_level_proj(visual_emb), dim=-1)
        text_emb1 = F.normalize(self.label_level_proj(text_emb), dim=-1)




        if self.training:
            cliploss = self.clip_loss(visual_emb, text_emb, self.logit_scale[0].exp(), output_dict=False)
            cliploss_visual_label = self.label_loss(visual_emb1, label_emb, self.logit_scale[1].exp(), prompt_target, output_dict=False)
            cliploss_text_label = self.label_loss(text_emb1, label_emb, self.logit_scale[1].exp(), prompt_target, output_dict=False)
            losses = {"cliploss": cliploss * self.cfg.SOLVER.LOSS_WEIGHT.CLIPLOSS, 
                      'cliploss_visual_label':cliploss_visual_label * self.cfg.SOLVER.LOSS_WEIGHT.CLIPLOSS_VISUAL_LABEL, 
                      'cliploss_text_label':cliploss_text_label * self.cfg.SOLVER.LOSS_WEIGHT.CLIPLOSS_TEXT_LABEL}
            # torch.save({'losses': losses,
            #             'logit_scale': self.logit_scale,
            #             'visual_emb': visual_emb,
            #             'text_emb': text_emb,
            #             'visual_emb1': visual_emb1,
            #             'label_emb': label_emb,
            #             'text_emb1': text_emb1,
            #             'prompt_target': prompt_target,
            #             'label_prompt':label_prompt}, f'var_{device}.pk')
            return losses
        else:
            vt_logits_l0 = visual_emb@text_emb.T
            vt_logits_l1 = visual_emb1@label_emb.T
            return {'vt_logits_l0' : vt_logits_l0, 
                    'vt_logits_l1' : vt_logits_l1,
                    'visual_emb_l0' : visual_emb,
                    'text_emb_l0' : text_emb,
                    'visual_emb_l1' : visual_emb1,
                    'text_emb_l1' : text_emb1,
                    'label_emb' : label_emb,
                    'visual_features' : visual_features,
                    'language_dict_features' : language_dict_features
                        } 



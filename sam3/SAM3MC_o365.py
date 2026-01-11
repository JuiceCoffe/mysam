# sam3/modeling_d2.py
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

import os
import numpy as np
import torchvision

from sam3.model_builder import (
    _create_vision_backbone,
    _create_text_encoder,
    _create_vl_backbone,
    _create_sam3_transformer,
    _create_dot_product_scoring,
    _create_segmentation_head,
    _create_geometry_encoder,
    _create_sam3_model,
)
from sam3.model.data_misc import FindStage, interpolate

from maft.utils.text_templetes import VILD_PROMPT
# VILD_PROMPT = ["{}"]

from .loss.matcher import HungarianMatcher
from .loss.criterion import SetCriterion
from sam3.model.content_dependent_transfer import ContentDependentTransfer

from maft.modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits

import random

import math


@META_ARCH_REGISTRY.register()
class SAM3MC_o365(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device_type = cfg.MODEL.DEVICE
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False) 
        
        # -------------------------------------------------------
        # 2. 实例化 SAM3 Model
        # -------------------------------------------------------       
        compile_mode = "default" if cfg.MODEL.SAM3.COMPILE else None
        
        vision_encoder = _create_vision_backbone(
            compile_mode=compile_mode, 
            enable_inst_interactivity=cfg.MODEL.SAM3.ENABLE_INST_INTERACTIVITY
        )
        text_encoder = _create_text_encoder(cfg.MODEL.SAM3.BPE_PATH)
        backbone = _create_vl_backbone(vision_encoder, text_encoder)
        transformer = _create_sam3_transformer()
        dot_prod_scoring = _create_dot_product_scoring()

        segmentation_head = (
            _create_segmentation_head(compile_mode=compile_mode)
            if cfg.MODEL.SAM3.ENABLE_SEGMENTATION
            else None
        )
            
        input_geometry_encoder = _create_geometry_encoder()

        enable_inst_interactivity = False # 遵照sam3设置
        if enable_inst_interactivity:
            sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
            inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
        else:
            inst_predictor = None

        self.detector = _create_sam3_model(
            backbone,
            transformer,
            input_geometry_encoder,
            segmentation_head,
            dot_prod_scoring,
            inst_predictor,
            cfg.eval_only,
        )
        if cfg.eval_only:
            self.detector.eval()
        print("SAM3创建成功!")


        # -------------------------------------------------------
        # 新增模块
        # -------------------------------------------------------
        self.mask_pooling = MaskPooling()

        # 【新增】初始化 logit_bias
        # 我们希望初始概率 p = 0.01
        # Sigmoid(x) = 0.01  =>  x = log(0.01 / (1 - 0.01)) ≈ -4.595
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.logit_bias = nn.Parameter(torch.ones([]) * bias_value)

        target_multiplier = 2.9029
        init_value = math.log(target_multiplier)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_value)

        self.num_cdt = cfg.MODEL.SAM3.NUM_CDT    
        self.use_cdt = False if cfg.MODEL.SAM3.NUM_CDT == 0 else True
        if self.use_cdt:
            # 使用 nn.ModuleList 包装
            self.cdt = nn.ModuleList([
                ContentDependentTransfer(d_model=256, nhead=8, panoptic_on=True) 
                for _ in range(self.num_cdt)
            ])
        else:
            self.cdt = None
        # -------------------------------------------------------
        # 训练配置
        # -------------------------------------------------------
        # 你需要检查 sam3_loss 的初始化参数

        self.train_dataname = None
        self.test_dataname = None
        self.test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        self.train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(self.train_metadata, self.train_metadata)

        self.use_aux = cfg.SOLVER.USE_AUX

        # -------------------------------------------------------
        # criterion损失函数
        # -------------------------------------------------------

        # loss weights
        class_weight = cfg.SOLVER.CLASS_WEIGHT
        dice_weight = cfg.SOLVER.DICE_WEIGHT
        mask_weight = cfg.SOLVER.MASK_WEIGHT

        weight_dict = {}
        criterion_weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        weight_dict.update(criterion_weight_dict)

        if self.use_aux:
            for i in range (5):
                for k in criterion_weight_dict.keys():
                    weight_dict[f"{k}_{i}"] = criterion_weight_dict[k]
        
        

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
        )
        
        losses = ["labels", "masks"]

        self.criterion = SetCriterion(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
        )

        # -------------------------------------------------------
        # 【新增】Inference 参数配置
        # -------------------------------------------------------
        self.semantic_on = cfg.TEST.SEMANTIC_ON
        self.instance_on = cfg.TEST.INSTANCE_ON 
        self.panoptic_on = cfg.TEST.PANOPTIC_ON 
        
        # 阈值设置 (如果没有在 cfg 定义，给默认值)
        self.object_mask_threshold = 0.01
        self.overlap_threshold = 0.8
        self.test_topk_per_image = 100

        self._freeze()

    def _freeze(self, ):
        for name, param in self.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            # elif 'dot_prod_scoring' in name:
            #     param.requires_grad = False
            elif 'geometry_encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        print('='*10,'Parameters to be trained', '='*10)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        # exit()

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            if isinstance(gt_masks, BitMasks):
                gt_masks = gt_masks.tensor
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        # --- 修改开始：解耦 train 和 test 的读取逻辑 ---
        
        # 1. 获取测试集类别 (优先读取 stuff_classes 以包含全景类别)
        try:
            class_names = split_labels(metadata.stuff_classes)
        except AttributeError:
            class_names = split_labels(metadata.thing_classes)

        # 2. 获取训练集类别 (独立处理，避免影响测试集)
        try:
            train_class_names = split_labels(train_metadata.stuff_classes)
        except AttributeError:
            train_class_names = split_labels(train_metadata.thing_classes)
            
        # --- 修改结束 ---

        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        self.vis_class_names = class_names
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names)) 
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self, dataname):
        if self.training:
            if self.train_dataname != dataname:
                text_classifier = []
                language_mask = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.train_class_names), "classes.")
                for idx in range(0, len(self.train_class_names), bs):
                    state_text = self.detector.backbone.forward_text(self.train_class_names[idx:idx+bs], device=self.device)

                    batch_text_feat = state_text["language_features"].detach()
                    mask = state_text["language_mask"] # bs, L
                    batch_text_feat = batch_text_feat.permute(1,0,2) # -> bs, L, D 
                    text_classifier.append(batch_text_feat)
                    language_mask.append(mask) # bs, L
                text_classifier = torch.cat(text_classifier, dim=0)
                language_mask = torch.cat(language_mask, dim=0) # (num_names * VILD_PROMPT,  L)
                # average across templates and normalization.
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, VILD_PROMPT, L, D
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 # [num_names, VILD_PROMPT, L, D] 掩码掉 padding 部分
                language_features = text_classifier.mean(1) # num_names, L, D
                text_classifier = text_classifier.mean(-2) 
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                
                self.language_features = language_features.detach() # num_names , L, D
                self.language_mask = torch.min(language_mask.view(language_features.shape[0],len(VILD_PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                self.train_text_classifier = text_classifier.detach()
                self.train_dataname = dataname
            return self.train_text_classifier.clone(), self.train_num_templates
        else:
            if self.test_dataname != dataname:
                self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(self.test_metadata[dataname], self.train_metadata)
                text_classifier = []
                language_mask = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.test_class_names), "classes.")
                for idx in range(0, len(self.test_class_names), bs):
                    state_text = self.detector.backbone.forward_text(self.test_class_names[idx:idx+bs], device=self.device)

                    batch_text_feat = state_text["language_features"].detach()
                    mask = state_text["language_mask"] # bs, L
                    batch_text_feat = batch_text_feat.permute(1,0,2) # -> bs, L, D 
                    text_classifier.append(batch_text_feat)
                    language_mask.append(mask) # bs, L
                text_classifier = torch.cat(text_classifier, dim=0)
                language_mask = torch.cat(language_mask, dim=0) # (num_names * VILD_PROMPT,  L)
                # average across templates and normalization.
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, VILD_PROMPT, L, D
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 # [num_names, VILD_PROMPT, L, D] 掩码掉 padding 部分
                language_features = text_classifier.mean(1) # num_names, L, D
                text_classifier = text_classifier.mean(-2) 
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                
                self.language_features = language_features.detach() # num_names , L, D
                self.language_mask = torch.min(language_mask.view(language_features.shape[0],len(VILD_PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                self.test_text_classifier = text_classifier.detach()
                self.test_dataname = dataname
            return self.test_text_classifier.clone(), self.test_num_templates

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # print('='*10,'Parameters to be trained', '='*10)
        # for name, param in self.named_parameters():
        #     if param.requires_grad == True:
        #         print(name)
        # exit()


        images = [x["image"].to(self.device) for x in batched_inputs]
        # print("shape of first image:", images[0].shape)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 14)
        # print("shape of images.tensor:", images.tensor.shape)
        img_h, img_w = images.tensor.shape[-2:]

        bs = images.tensor.shape[0]
        
        self.find_stage = FindStage(
            img_ids=torch.arange(bs, device=self.device, dtype=torch.long),
            text_ids=torch.arange(bs, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        with torch.no_grad():

            file_names = [x["file_name"] for x in batched_inputs]
            file_names = [x.split('/')[-1].split('.')[0] for x in file_names]
            
            if 'meta' in batched_inputs[0]:
                meta = batched_inputs[0]["meta"]
            else:
                meta = batched_inputs[0]
            
            # print("keys of meta:", meta.keys())
            dataname = get_dataname(batched_inputs[0])
            
            # 图形特征
            backbone_out_vision = self.detector.backbone.forward_image(images.tensor)
            img_feat = backbone_out_vision["vision_features"].detach() # bs, D, H', W'
            backbone_fpn = backbone_out_vision["backbone_fpn"]
            for k in range(len(backbone_fpn)):
                backbone_fpn[k] = backbone_fpn[k].detach()


            # 语言特征
            # text_classifier:[num_names, dim] 
            # language_features:[num_names, num_templates, L, dim] language_mask:[num_names, num_templates, L]
            
            # text_classifier, num_templates = self.get_text_classifier(meta['dataname'])
            text_classifier, num_templates = self.get_text_classifier(dataname)

            # others
            geometric_prompt = self.detector._get_dummy_prompt(bs)
        
        batch_gt_names_idx = []
        for i in range(bs):
            # gt_classes = get_gt_labels_from_sem_seg(batched_inputs[i]["sem_seg"].to(self.device))

            # === 修改开始：适配 Objects365 的 Instances 格式 ===
            gt_classes = []
            if "instances" in batched_inputs[i]:
                # 从实例中提取去重后的类别 ID
                if len(batched_inputs[i]["instances"]) > 0:
                    gt_classes = batched_inputs[i]["instances"].gt_classes.unique().cpu().tolist()
            elif "sem_seg" in batched_inputs[i]:
                # 兼容旧的 COCO Stuff 逻辑
                gt_classes = get_gt_labels_from_sem_seg(batched_inputs[i]["sem_seg"].to(self.device))
            # === 修改结束 ===

            gt_names_idx = []
            cur_idx = 0
            for i,num_t in enumerate(num_templates): 
                if i in gt_classes:
                    gt_names_idx += list(range(cur_idx, cur_idx + num_t))
                cur_idx += num_t
            batch_gt_names_idx.append(gt_names_idx)

        # =======================================================
        
        language_features_input = []

        # USE_GT_NAMES_ONLY = True
        USE_GT_NAMES_ONLY = False
        
        if USE_GT_NAMES_ONLY:
            language_features_input = [self.language_features[batch_gt_names_idx[i],:,:] for i in range(bs)]
            language_features_input = torch.cat(language_features_input, dim=0) # (bs, num_names * L, dim)
            language_mask_input = [self.language_mask[batch_gt_names_idx[i],:] for i in range(bs)]
            language_mask_input = torch.cat(language_mask_input, dim=0) # (bs, num_names * L)
            if bs == 1:
                language_features_input = language_features_input.unsqueeze(0)
                language_mask_input = language_mask_input.unsqueeze(0)

        else:
            language_features_input = self.language_features.expand(bs, -1, -1, -1) # (bs, num_names, L, dim)
            language_mask_input = self.language_mask.expand(bs, -1, -1) # (bs, num_names, L)

        # print("shape of input:",language_features_input.shape, language_mask_input.shape)
        language_features_input = language_features_input.reshape(bs, -1, language_features_input.shape[-1]) # (bs, num_names * L, dim)
        language_mask_input = language_mask_input.reshape(bs, -1) # (bs, num_names * L)
        # print("shape of input after reshape:",language_features_input.shape, language_mask_input.shape)
        

        backbone_out={
            "img_batch_all_stages": img_feat,
            "vision_pos_enc": backbone_out_vision["vision_pos_enc"],
            "backbone_fpn": backbone_fpn,
            "language_features": language_features_input.permute(1, 0, 2), # (num_names * L, bs, dim)
            "language_mask": language_mask_input, # bs, (num_names * L)
        }

        #=================================
        find_input = self.find_stage

        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self.detector._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
        # Run the encoder
        with torch.profiler.record_function("SAM3Image._run_encoder"):
            backbone_out, encoder_out, _ = self.detector._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        # print("keys of out before decoder:", out.keys()) # s(['encoder_hidden_states', 'prev_encoder_out'])
        # Run the decoder
        with torch.profiler.record_function("SAM3Image._run_decoder"):
            out, hs = self.detector._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

        # print("keys of out after decoder:", out.keys()) # (['encoder_hidden_states', 'prev_encoder_out', 'presence_feats', 'queries', 'presence_logit_dec', 'pred_logits', 'pred_boxes', 'pred_boxes_xyxy'])
        # Run segmentation heads
        with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
            self.detector._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
                aux_masks=True,
            )
        
        # if self.detector.training or self.detector.num_interactive_steps_val > 0:
        #     self.detector._compute_matching(out, self.detector.back_convert(find_target))

        #========================================
        outputs = out
        # print("outputs keys:", outputs.keys())
        # print('aux:',outputs['aux_outputs'][0].keys())

        out_masks = outputs["pred_masks"].clone()

        out_masks = out_masks.sigmoid()

        # presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1) # 在多类别情况下认为失效

        # out_semseg = outputs["semantic_seg"] # 原语义分割头输出，舍去
        # out_semseg = F.interpolate(
        #     out_semseg,
        #     size=(img_h, img_w),
        #     mode="bilinear",
        #     align_corners=False,
        # ).sigmoid()


        # print("out_masks shape:", out_masks.shape, "out_probs shape:", out_probs.shape, "out_semseg shape:", out_semseg.shape, "presence_score shape:", presence_score.shape)
        # out_masks shape: torch.Size([1, 200, 1008, 1008]) out_probs shape: torch.Size([1, 200]) out_semseg shape: torch.Size([1, 1, 1008, 1008]) presence_score shape: torch.Size([1, 1, 1])
        

        bs, N, H, W = out_masks.shape
        C_ = text_classifier.shape[0] # num_names 

        queries_masks = out_masks # out_probs是通过与池化prompt投影卷积实现的，多类别下失效，直接用原始mask_logits

        queries = outputs["obj_queries"] # 6, bs, N, D
        pixel_embed = outputs["pixel_embed"] # bs, D, H', W'
        instance_embeds = outputs["instance_embeds"] 

        use_aux = self.use_aux and self.training
        aux_outputs = []

        if self.use_cdt:
            # text_classifier = self.cdt(img_feat,text_classifier)
            for layer in self.cdt:
                text_classifier = layer(img_feat,text_classifier) # 逐层通过

        for i in range(6):
            assert queries.shape[0] == 6
            if use_aux or i == 5 :
                tp_queries = queries[i,:,:N,:].clone() # 避免DAC造成的tp_queries翻倍
                tp_queries = F.normalize(tp_queries, dim=-1, p=2)

                if self.use_cdt:
                    query_names_results = torch.einsum("bnd,bcd->bnc", tp_queries, text_classifier) # bs, N, C
                else:
                    query_names_results = torch.einsum("bnd,cd->bnc", tp_queries, text_classifier) # bs, N, C
                
                
                logit_scale = self.logit_scale.exp()
                logit_scale = torch.clamp(logit_scale, max=100.0)
                query_names_results = logit_scale * query_names_results + self.logit_bias

                query_cls_results= []
                cur_idx = 0
                for num_t in num_templates: 
                    query_cls_results.append(query_names_results[:,:, cur_idx: cur_idx + num_t].max(-1).values)
                    cur_idx += num_t
                query_cls_results = torch.stack(query_cls_results, dim=-1) # bs, N, num_classes
                # print(f"aux query_cls_results[{i}] shape:", query_cls_results.shape)
                    

                if i<5:
                    aux_outputs.append({'pred_logits': query_cls_results, 'pred_masks': outputs['aux_outputs'][i]["pred_masks"]})
                else:
                    query_cls_results_final = query_cls_results



        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            criterion_pred = {
                'pred_logits': query_cls_results_final,
                'pred_masks': outputs["pred_masks"],
                'aux_outputs': aux_outputs if use_aux is True else None,
            }

            losses = self.criterion(criterion_pred, targets)

            for k in list(losses.keys()):
                # print("loss:", k, losses[k].item())
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        
        else:

        # =======================================================
        
            mask_cls_logits = query_cls_results_final # 保持 Logits 状态
            mask_pred_logits = outputs["pred_masks"]  # 保持 Logits 状态


            results = []
            
            for i in range(bs):
                # 获取单张图数据
                mask_cls_i = mask_cls_logits[i]       # [Q, C]
                mask_pred_i = mask_pred_logits[i]     # [Q, H, W]
                
                # 获取原始图像尺寸
                img_h_orig = batched_inputs[i]["height"]
                img_w_orig = batched_inputs[i]["width"]
                
                # 上采样 Mask 到原始图像尺寸 (非常重要)
                # 使用 bilinear 插值 logits
                mask_pred_i = F.interpolate(
                    mask_pred_i.unsqueeze(0), 
                    size=(img_h_orig, img_w_orig), 
                    mode="bilinear", 
                    align_corners=False
                ).squeeze(0)

                res = {}
                # dataname = batched_inputs[i]["meta"]["dataname"]

                # --- A. 语义分割 (Semantic Segmentation) ---
                if self.semantic_on:
                    # 使用你原来的逻辑，但注意输入变成了 logits
                    mask_cls_prob = mask_cls_i.sigmoid()
                    mask_pred_prob = mask_pred_i.sigmoid()
                    semseg = torch.einsum("qc,qhw->chw", mask_cls_prob, mask_pred_prob)
                    res["sem_seg"] = semseg

                    # # =========== 修改开始：为可视化准备 Square 数据 ===========
                    
                    # # 1. 获取当前输入 Tensor 的尺寸 (即正方形尺寸，例如 1024x1024)
                    # # batched_inputs[i]["image"] 是经过 mapper 处理后的图
                    # tensor_h, tensor_w = batched_inputs[i]["image"].shape[-2:]
                    
                    # # 2. 将 Logits 插值到 Tensor 尺寸 (而不是原图尺寸)
                    # # 注意：这里要用原始的 mask_pred_logits[i]，不要用已经 resize 过的 mask_pred_i
                    # mask_pred_i_square = F.interpolate(
                    #     mask_pred_logits[i].unsqueeze(0), 
                    #     size=(tensor_h, tensor_w), 
                    #     mode="bilinear", 
                    #     align_corners=False
                    # ).squeeze(0).sigmoid()
                    
                    # # 3. 计算 Square 的语义分割结果
                    # semseg_square = torch.einsum("qc,qhw->chw", mask_cls_prob, mask_pred_i_square)
                    # pred_result_square = semseg_square.argmax(0).cpu()

                    # # 4. 准备 GT (它本身就是 Square 的，只要搬运到 CPU)
                    # # 如果 mapper 对 GT 做了 padding，这里就是带 padding 的正方形
                    # gt_result_square = batched_inputs[i]["sem_seg"].to(self.device)

                    # # 5. 准备 Image (它本身就是 Square 的)
                    # img_tensor_square = batched_inputs[i]["image"]

                    # # 6. 获取类别名称
                    # current_dataname = batched_inputs[i]["meta"]["dataname"]
                    # if current_dataname in self.test_metadata:
                    #     meta = self.test_metadata[current_dataname]
                    # else:
                    #     meta = MetadataCatalog.get(current_dataname)
                    
                    # try:
                    #     current_class_names = meta.stuff_classes
                    # except:
                    #     current_class_names = meta.thing_classes

                    # # 7. 绘图 (全部传入 Square 的数据)
                    # visualize_segmentation(
                    #     pred_result=pred_result_square,       # 修改点：传入 Square 预测
                    #     gt_result=gt_result_square,           # 本身就是 Square
                    #     class_names=current_class_names + ['background'],
                    #     original_image_tensor=img_tensor_square, # 本身就是 Square
                    #     save_path=f"./show_semantic/{batched_inputs[i]['file_name'].split('/')[-1].split('.')[0]}.png"
                    # )
                    # # =========== 修改结束 ===========

                # --- B. 全景分割 (Panoptic Segmentation) ---
                if self.panoptic_on:
                    excluded_datasets = ["lvis_v1_val", "lvis_v1_train"]
                    
                    if dataname not in excluded_datasets:
                        panoptic_seg, segments_info = self.panoptic_inference(
                            mask_cls_i, mask_pred_i, dataname
                        )
                        res["panoptic_seg"] = (panoptic_seg, segments_info)
                
                # --- C. 实例分割 (Instance Segmentation) ---
                if self.instance_on:
                    instances = self.instance_inference(
                        mask_cls_i, mask_pred_i, dataname
                    )
                    res["instances"] = instances

                results.append(res)

            return results



    def panoptic_inference(self, mask_cls, mask_pred, dataname):
        # mask_cls: [Q, K] (Logits)
        # mask_pred: [Q, H, W] (Logits or Probs, depends on input)
        
        # 1. 计算分数 (Sigmoid 而不是 Softmax)
        scores, labels = mask_cls.sigmoid().max(-1) # [Q]
        mask_pred = mask_pred.sigmoid() # [Q, H, W]

        # 2. 过滤掉低分 Query
        keep = scores > self.object_mask_threshold
        
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        
        # 加权 Mask
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info

        # 3. Argmax 生成全景图
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        
        # 获取 Metadata
        meta = self.test_metadata[dataname] if dataname in self.test_metadata else MetadataCatalog.get(dataname)
        thing_ids = set(meta.thing_dataset_id_to_contiguous_id.values())

        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in thing_ids
            
            # 检查 Mask 质量
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < self.overlap_threshold:
                    continue

                # 合并 Stuff 区域
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, dataname):
        # mask_cls: [Q, K] (Logits)
        # mask_pred: [Q, H, W] (Logits)
        
        image_size = mask_pred.shape[-2:]
        
        # 1. 计算分数 (Sigmoid)
        scores = mask_cls.sigmoid() # [Q, K]
        num_classes = scores.shape[-1]
        
        # 2. 展开所有 Query-Class 对
        num_queries = scores.shape[0]
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        # 找到对应的 mask index
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[topk_indices]

        # 4. 过滤 Thing Classes (如果是 Panoptic 模式)
        if self.panoptic_on:
            meta = self.test_metadata[dataname] if dataname in self.test_metadata else MetadataCatalog.get(dataname)
            if hasattr(meta, 'thing_dataset_id_to_contiguous_id'):
                thing_ids = set(meta.thing_dataset_id_to_contiguous_id.values())
            else:
                # 如果没有映射表，默认使用所有类别的连续索引
                thing_ids = set(range(len(meta.thing_classes)))
            
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab.item() in thing_ids
            
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        # 5. 生成 Instances 对象
        result = Instances(image_size)
        
        # 使用 Sigmoid 后的 Mask
        mask_pred_sigmoid = mask_pred.sigmoid()
        pred_masks_binary = (mask_pred_sigmoid > 0.5).float()
        result.pred_masks = pred_masks_binary
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4)) # SAM3 通常不直接出框，这里放空框或者用 mask2box 计算
        
        # 计算综合分数
        mask_scores_per_image = (mask_pred_sigmoid.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        if pred_masks_binary.numel() > 0:
            # BitMasks 最好接收 Bool 或 Uint8
            # result.pred_masks 是 float，这里转一下 ensure 安全
            result.pred_boxes = BitMasks(pred_masks_binary > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(0, 4, device=self.device))
            
        return result

def get_gt_labels_from_sem_seg(sem_seg):
    """
    基于 sem_seg_2_gt_masks 逻辑提取当前图像中存在的有效类别 ID。
    """
    # 确保是 2D 张量 (H, W)
    if sem_seg.dim() == 3: 
        sem_seg = sem_seg.squeeze(0)
    
    # 获取唯一类别
    classes = torch.unique(sem_seg, sorted=False, return_inverse=False, return_counts=False)
    
    # 过滤掉背景/忽略类 (通常是 255，或者是 void 类)
    # 注意：这里假设 255 是忽略索引，根据你的数据集调整
    gt_labels = classes[classes != 255]
    
    return gt_labels.cpu().numpy().tolist()


def visualize_segmentation(
    pred_result, 
    gt_result,
    class_names, 
    original_image_tensor, 
    save_path="./show/result.png", 
    fig_size=(20, 10),
    ignore_index=255
):
    """
    可视化分割结果：[原图, 预测图, GT图] 并排显示，并附带图例。
    """
    # --- 数据准备 ---
    if isinstance(pred_result, torch.Tensor):
        pred_result = pred_result.cpu().numpy()
    if isinstance(gt_result, torch.Tensor):
        gt_result = gt_result.cpu().numpy()
    if isinstance(original_image_tensor, torch.Tensor):
        # (C, H, W) -> (H, W, C)
        original_image = original_image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        original_image = original_image_tensor

    # 统一尺寸（以原图为准）
    h, w = original_image.shape[:2]
    num_classes = len(class_names)

    # --- 颜色调色板 ---
    np.random.seed(42) # 固定种子
    palette = np.random.randint(0, 255, size=(num_classes, 3))
    # 为 ignore_index (255) 分配灰色
    ignore_color = np.array([128, 128, 128], dtype=np.uint8)

    def mask_to_rgb(mask):
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(num_classes):
            rgb[mask == i] = palette[i]
        rgb[mask == ignore_index] = ignore_color
        return rgb

    # 生成 RGB 掩码图
    pred_rgb = mask_to_rgb(pred_result)
    gt_rgb = mask_to_rgb(gt_result)

    # --- 统计出现的类别用于图例 ---
    # 合并 Pred 和 GT 中出现的类别，以便在图例中全部展示
    present_in_pred = np.unique(pred_result)
    present_in_gt = np.unique(gt_result)
    all_present_classes = np.unique(np.concatenate([present_in_pred, present_in_gt]))
    
    # 过滤掉 ignore_index 和超出范围的索引
    all_present_classes = [c for c in all_present_classes if c < num_classes and c >= 0]
    
    # 按像素占比（在 Pred 中）排序，让图例更整洁
    class_counts = {c: np.sum(pred_result == c) for c in all_present_classes}
    sorted_classes = sorted(all_present_classes, key=lambda x: class_counts[x], reverse=True)

    # --- 绘图 ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=fig_size)

    # 1. 原图
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image", fontsize=15)
    ax[0].axis('off')

    # 2. 预测图
    ax[1].imshow(pred_rgb)
    ax[1].set_title("Prediction", fontsize=15)
    ax[1].axis('off')

    # 3. GT 图
    ax[2].imshow(gt_rgb)
    ax[2].set_title("Ground Truth", fontsize=15)
    ax[2].axis('off')

    # --- 图例 ---
    legend_elements = []
    for c_idx in sorted_classes:
        color = palette[c_idx] / 255.0
        name = class_names[c_idx]
        count = class_counts[c_idx]
        legend_elements.append(
            Rectangle((0, 0), 1, 1, color=color, label=f"{name} ({count:,} px)")
        )
    
    # 如果有忽略区域，添加说明
    if np.any(gt_result == ignore_index):
        legend_elements.append(
            Rectangle((0, 0), 1, 1, color=ignore_color/255.0, label="Ignore/Void")
        )

    # 放置图例
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(5, len(legend_elements)), 
        frameon=True,
        fontsize='small'
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # 为底部图例留出空间
    
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"可视化已保存: {save_path}")
    except Exception as e:
        print(f"保存失败: {e}")
    plt.close(fig)

def get_dataname(batched_input):
    """
    从输入数据中提取或推断 dataname (数据集名称)。
    逻辑：
    1. 尝试直接从 meta 或 batched_input 根目录找 'dataname'
    2. 如果找不到，通过 'file_name' 字符串匹配
    """
    # 1. 尝试从 meta 字典或根字典中直接获取
    if "meta" in batched_input and "dataname" in batched_input["meta"]:
        return batched_input["meta"]["dataname"]
    if "dataname" in batched_input:
        return batched_input["dataname"]

    # 2. 备选方案：通过 file_name 推断
    # Detectron2 的 dataset_dict 几乎百分之百包含 file_name
    file_name = batched_input.get("file_name", "")
    
    # 逻辑：目前只管 lvis，也可以顺便兼容 ade
    file_name_lower = file_name.lower()
    
    if "lvis" in file_name_lower:
        return "lvis"
    elif "ade" in file_name_lower:
        return "ade20k"
    
    # 如果都匹配不上，返回一个默认值或抛出警告
    # print(f"Warning: Could not infer dataname from {file_name}, using default 'lvis_v1_val'")
    return "lvis_v1_val"
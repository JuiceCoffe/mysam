# sam3/modeling_d2.py
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


@META_ARCH_REGISTRY.register()
class SAM3MC(nn.Module):
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
        # 3. 训练配置
        # -------------------------------------------------------
        # 你需要检查 sam3_loss 的初始化参数
        self.loss_func = None
        self.train_dataname = None
        self.test_dataname = None
        self.test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        self.train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        # --------------------------------------------------
        # sam3的其他配置
        # --------------------------------------------------
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device = self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device = self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
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
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.detector.backbone.forward_text(self.train_class_names[idx:idx+bs], device=self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
                self.train_dataname = dataname
            return self.train_text_classifier, self.train_num_templates
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
                    mask = state_text["language_mask"] # B, L
                    batch_text_feat = batch_text_feat.permute(1,0,2) # -> B, L, D 
                    text_classifier.append(batch_text_feat)
                    language_mask.append(mask) # B, L
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
                
                self.language_features = language_features # num_names , L, D
                self.language_mask = torch.min(language_mask.view(language_features.shape[0],len(VILD_PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                self.test_text_classifier = text_classifier 
                self.test_dataname = dataname
            return self.test_text_classifier, self.test_num_templates

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        # print("shape of first image:", images[0].shape)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 14)
        # print("shape of images.tensor:", images.tensor.shape)
        img_h, img_w = images.tensor.shape[-2:]

        batch_size = images.tensor.shape[0]
        

        file_names = [x["file_name"] for x in batched_inputs]
        file_names = [x.split('/')[-1].split('.')[0] for x in file_names]

        meta = batched_inputs[0]["meta"]
        
        # 图形特征
        backbone_out_vision = self.detector.backbone.forward_image(images.tensor)
        img_feat = backbone_out_vision["vision_features"] # B, C, H', W'
        backbone_fpn = backbone_out_vision["backbone_fpn"]
    

        # 语言特征
        # text_classifier:[num_names, dim] 
        # language_features:[num_names, num_templates, L, dim] language_mask:[num_names, num_templates, L]
        text_classifier, num_templates = self.get_text_classifier(meta['dataname'])

        # others
        geometric_prompt = self.detector._get_dummy_prompt()
        
        batch_gt_names_idx = []
        for i in range(batch_size):
            gt_classes = get_gt_labels_from_sem_seg(batched_inputs[i]["sem_seg"].to(self.device))
            gt_names_idx = []
            cur_idx = 0
            for i,num_t in enumerate(num_templates): 
                if i in gt_classes:
                    gt_names_idx += list(range(cur_idx, cur_idx + num_t))
                cur_idx += num_t
            batch_gt_names_idx.append(gt_names_idx)

        # =======================================================
        
        language_features_input = []

        USE_GT_NAMES_ONLY = True
        
        if USE_GT_NAMES_ONLY:
            language_features_input = [self.language_features[batch_gt_names_idx[i],:,:] for i in range(batch_size)]
            language_features_input = torch.cat(language_features_input, dim=0) # (B, num_names * L, dim)
            language_mask_input = [self.language_mask[batch_gt_names_idx[i],:] for i in range(batch_size)]
            language_mask_input = torch.cat(language_mask_input, dim=0) # (B, num_names * L)
            if batch_size == 1:
                language_features_input = language_features_input.unsqueeze(0)
                language_mask_input = language_mask_input.unsqueeze(0)

        else:
            language_features_input = self.language_features
            language_mask_input = self.language_mask

        # print("shape of input:",language_features_input.shape, language_mask_input.shape)
        language_features_input = language_features_input.reshape(batch_size, -1, language_features_input.shape[-1]) # (B, num_names * L, dim)
        language_mask_input = language_mask_input.reshape(batch_size, -1) # (B, num_names * L)
        # print("shape of input after reshape:",language_features_input.shape, language_mask_input.shape)
        

        backbone_out={
            "img_batch_all_stages": img_feat,
            "vision_pos_enc": backbone_out_vision["vision_pos_enc"],
            "backbone_fpn": backbone_fpn,
            "language_features": language_features_input.permute(1, 0, 2), # (num_names * L, B, dim)
            "language_mask": language_mask_input, # B, (num_names * L)
        }
        # outputs = self.detector.forward_grounding(
        #     backbone_out = backbone_out,
        #     find_input=self.find_stage,
        #     geometric_prompt= geometric_prompt,
        #     find_target=None,
        # )

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
            )
        
        if self.detector.training or self.detector.num_interactive_steps_val > 0:
            self.detector._compute_matching(out, self.detector.back_convert(find_target))

        #========================================
        outputs = out
        # print("outputs keys:", outputs.keys())

        out_bbox = outputs["pred_boxes"]


        out_masks = outputs["pred_masks"]
        out_masks = interpolate(
            out_masks,
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        presence_score = torch.ones(batch_size,1, device=self.device)  # 由于多类别情况下 presence_logit_dec 不适用，暂时全部设为 1 [B, 1]
        # presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1) # 在多类别情况下认为失效

        out_logits = outputs["pred_logits"] 
        out_probs = out_logits.sigmoid() # B, N, 1
        # out_probs = (out_probs * presence_score).squeeze(-1) # 实例的概率



        out_semseg = outputs["semantic_seg"]
        out_semseg = F.interpolate(
            out_semseg,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()


        # print("out_masks shape:", out_masks.shape, "out_probs shape:", out_probs.shape, "out_semseg shape:", out_semseg.shape, "presence_score shape:", presence_score.shape)
        # out_masks shape: torch.Size([1, 200, 1008, 1008]) out_probs shape: torch.Size([1, 200]) out_semseg shape: torch.Size([1, 1, 1008, 1008]) presence_score shape: torch.Size([1, 1, 1])
        

        B, N, H, W = out_masks.shape
        C = text_classifier.shape[0] # num_names


        queries_masks = torch.mul(out_masks,out_probs.unsqueeze(-1)) # 每个实例的mask乘以对应的概率得分 [B, N, H, W]
        # 现在还需要对query分类
        queries = outputs["queries"]
        # print("queries shape:", queries.shape)
        # print("classifier shape:", text_classifier.shape)
        queries_names_probs = torch.einsum("bnd,cd->bnc", queries, text_classifier).sigmoid() # B, N, num_names

        queries_seg_result = torch.zeros((B, C, H, W), dtype=out_masks.dtype, device=out_masks.device)

        for n in range(queries.shape[1]):
            query_instance_result = (queries_names_probs[:, n, :].unsqueeze(-1).unsqueeze(-1) * queries_masks[:, n, :, :].unsqueeze(1)) # [B, num_names, 1, 1] * [B, 1, H, W] -> [B, num_names, H, W]
            queries_seg_result = torch.max(queries_seg_result, query_instance_result)


        # semantic_masks = torch.mul(out_semseg , presence_score.unsqueeze(-1)) 也用不了了

        # 用pixel_embed点积分类器代替semantic head
        pixel_embed = outputs["pixel_embed"].permute(0,2,3,1) # B, H, W, D
        # print("pixel_embed shape:", pixel_embed.shape)
        sem_seg_logits = torch.einsum("bhwd,cd->bhwc", pixel_embed, text_classifier) # B, H, W, num_names
        sem_seg_logits = sem_seg_logits.permute(0,3,1,2) # B, num_names, H, W
        sem_seg_probs = sem_seg_logits.sigmoid()
        # print("sem_seg_probs shape:", sem_seg_probs.shape)


        #整合两部分的结果
        sem_seg_logits = F.interpolate(
            sem_seg_probs,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )

        seg_logits = torch.max(queries_seg_result, sem_seg_logits) # [B, num_names, H, W]
        if USE_GT_NAMES_ONLY:
            for b in range(batch_size):
                gtcls_mask = torch.ones_like(seg_logits[b], dtype=torch.bool)
                gtcls_mask[batch_gt_names_idx[b],:,:] = False
                seg_logits[b][gtcls_mask] = 0.0
        

        # =======================================================
        
        

        final_seg_logits = []
        cur_idx = 0
        for num_t in num_templates: 
            final_seg_logits.append(seg_logits[:, cur_idx: cur_idx + num_t,:,:].max(1).values)
            cur_idx += num_t
        final_seg_logits = torch.stack(final_seg_logits, dim=1)

        results = []
        for i in range(batch_size):
            orig_size = (batched_inputs[i]["height"], batched_inputs[i]["width"]) # 没经过 resize 到 1008 的大小
            res = sem_seg_postprocess(
                final_seg_logits[i], 
                (img_h, img_w),
                orig_size[0],
                orig_size[1],
            )
            results.append({"sem_seg": res})
        return results


    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

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


# sam3/config.py
from detectron2.config import CfgNode as CN

def add_sam3_config(cfg):
    """
    为 Detectron2 的 cfg 添加 SAM3 专用配置
    """
    cfg = cfg
    
    cfg.MODEL.SAM3 = CN()
    # 对应 build_sam3_image_model 中的参数
    cfg.MODEL.SAM3.MODEL_TYPE = "sam3_image" # 预留，也许有 sam3_video
    cfg.MODEL.SAM3.BPE_PATH = "sam3/assets/bpe_simple_vocab_16e6.txt.gz" # 默认路径
    cfg.MODEL.SAM3.CHECKPOINT_PATH = "/data/hmp/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt" # 官方预训练权重路径
    

    # 功能开关
    cfg.MODEL.SAM3.ENABLE_SEGMENTATION = True
    cfg.MODEL.SAM3.ENABLE_INST_INTERACTIVITY = False
    cfg.MODEL.SAM3.COMPILE = False
    
    # 可以在这里添加更多关于 Backbone/Transformer 的细节参数
    # cfg.MODEL.SAM3.NUM_FEATURE_LEVELS = 1

    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    cfg.INPUT.IMAGE_SIZE = 1008
    cfg.INPUT.COLOR_AUG_SSD = False    
    cfg.INPUT.SIZE_DIVISIBILITY = -1


    cfg.MODEL.PIXEL_MEAN = [127.5, 127.5, 127.5]
    cfg.MODEL.PIXEL_STD = [127.5, 127.5, 127.5]

    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

import argparse
class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# MIMIC Aanatomy dataset args
mimic_anatomy_args_dict = {
    'batch_size': 8,
    'imsize': 640, # to change
    'max_query_len': 50,
    'device': 'cuda',
    'bert_model': 'medicalai/ClinicalBERT',
    'num_workers': 0,
    'img_dir': '/scratch/project/cxrdata',
    'anno_dir': '/scratch/user/uqwzha16/CXR-GLIP/data/MIMICCXR',
    'anatomy': True,
    'anatomy_variation_count': 5,
    'anatomy_sampling': 'random',
}

mimic_anatomy_args = Args(**mimic_anatomy_args_dict)

# MIMIC Pathology dataset args
mimic_pathology_args_dict = {
    'batch_size': 8,
    'imsize': 640, # to change
    'max_query_len': 50,
    'device': 'cuda',
    'bert_model': 'medicalai/ClinicalBERT',
    'num_workers': 0,
    'img_dir': '/scratch/project/cxrdata',
    # 'anno_dir': '/scratch/user/uqwzha16/LLM_datasetconstruction/combined_processed_output.csv',
    'anno_dir': '/scratch/user/uqwzha16/LLM_datasetconstruction/generated_phrases.csv',
}
mimic_pathology_args = Args(**mimic_pathology_args_dict)

# MIMIC MS-CXR dataset args
ms_cxr_args_dict = {
    'batch_size': 8,
    'imsize': 640, # to change
    'device': 'cuda',
    'num_workers': 0,
    'img_dir': '/scratch/user/uqwzha16/MedRPG/ln_data/MS_CXR',
    'anno_dir': '/scratch/user/uqwzha16/MedRPG/data/MS_CXR'
}
ms_args = Args(**ms_cxr_args_dict)

# TransVG model args
transvg_args_dict = {
    'lr': 0.00005 ,
    'lr_bert': 1e-5,
    'lr_visu_cnn': 1e-5,
    'lr_visu_tra': 1e-5,
    'batch_size': 8,
    'weight_decay': 1e-4,
    'epochs': 90,
    'lr_power': 0.9,
    'clip_max_norm': 0.0,
    'eval': False,
    'optimizer': 'adamw',
    'lr_scheduler': 'step',
    'lr_drop': 60,
    'model_name': 'TransVG', 
    'memory': False,
    'm_threadHead': 4,
    'm_size': 2048,
    'm_dim': 256,
    'm_topK': 32,
    'm_linearMode': 'noLinear',
    'm_MMHA_outLinear': False,
    'm_resLearn': False,
    'btloss': False,
    'lossbt_type': 'l1',
    'lossbt_weight': 1.0,
    'GNpath': 'data/MS_CXR/Genome',
    'GNClsType': 'mcls',
    'gnlossWeightBase': 0.1,
    'CAsampleType': 'random',
    'CAsampleNum': 5,
    'CAlossWeightBase': 0.05,
    'CATextPoolType': 'cls',
    'CATemperature': 0.1,
    'CAMode': 'max_image_lcpTriple', # to change
    'ConsLossWeightBase': 1.0,
    'ablation': 'none',
    'backbone': 'resnet50',
    'bert_model': 'medicalai/ClinicalBERT',
    'detr_model': './checkpoints/detr-r50-unc.pth',
    
    'dilation': False, # to change, true -> 16, false-> 8
    'position_embedding': 'sine',
    'enc_layers': 6,
    'dec_layers': 6,
    'dim_feedforward': 2048,
    'hidden_dim': 256,
    'dropout': 0.1,
    'nheads': 8,
    'num_queries': 30,
    'pre_norm': False,
    'imsize': 640, # to change
    'max_query_len': 50,
    'emb_size': 512,
    'bert_enc_num': 12,
    'detr_enc_num': 6,
    'vl_dropout': 0.1,
    'vl_nheads': 8,
    'vl_hidden_dim': 256,
    'vl_dim_feedforward': 2048,
    'vl_enc_layers': 6,
    'contrastive_dim': 256, 
}

transvg_args = Args(**transvg_args_dict)


# MDETR model args
mdetr_args_dict = {
    # Dataset specific
    "run_name": "",
    "dataset_config": None,
    "do_qa": False,
    "predict_final": False,
    "no_detection": False,
    "split_qa_heads": False,
    "combine_datasets": ["flickr"],
    "combine_datasets_val": ["flickr"],
    "coco_path": "",
    "vg_img_path": "",
    "vg_ann_path": "",
    "clevr_img_path": "",
    "clevr_ann_path": "",
    "phrasecut_ann_path": "",
    "phrasecut_orig_ann_path": "",
    "modulated_lvis_ann_path": "",

    # Training hyper-parameters
    "lr": 1e-4,
    "lr_backbone": 1e-5,
    "text_encoder_lr": 5e-5,
    "batch_size": 8,
    "weight_decay": 1e-4,
    "epochs": 40,
    "lr_drop": 35,
    "epoch_chunks": -1,
    "optimizer": "adam",
    "clip_max_norm": 0.1,
    "eval_skip": 1,
    "schedule": "linear_with_warmup",
    "ema": False,
    "ema_decay": 0.9998,
    "fraction_warmup_steps": 0.01,

    # Model parameters
    "frozen_weights": None,
    "freeze_text_encoder": False,
    "text_encoder_type": "roberta-base",

    # Backbone
    "backbone": "resnet101",
    "dilation": False,
    "position_embedding": "sine",

    # Transformer
    "num_classes": 255,
    "enc_layers": 6,
    "dec_layers": 6,
    "dim_feedforward": 2048,
    "hidden_dim": 256,
    "dropout": 0.1,
    "nheads": 8,
    "num_queries": 1,
    "pre_norm": False,
    "pass_pos_and_query": True,

    # Segmentation
    "mask_model": "none",
    "remove_difficult": False,
    "masks": False,

    # Loss
    "aux_loss": True,
    "set_loss": "hungarian",
    "contrastive_loss": False,
    "contrastive_align_loss": True,
    "contrastive_loss_hdim": 64,
    "temperature_NCE": 0.07,

    # Matcher
    "set_cost_class": 1,
    "set_cost_bbox": 5,
    "set_cost_giou": 2,

    # Loss coefficients
    "ce_loss_coef": 1,
    "mask_loss_coef": 1,
    "dice_loss_coef": 1,
    "bbox_loss_coef": 5,
    "giou_loss_coef": 2,
    "qa_loss_coef": 1,
    "eos_coef": 0.1,
    "contrastive_loss_coef": 0.1,
    "contrastive_align_loss_coef": 1,

    # Run specific
    "test": False,
    "test_type": "test",
    "output_dir": "",
    "device": "cuda",
    "seed": 42,
    "resume": "pretrained_resnet101_checkpoint.pth",
    "load": "",
    "start_epoch": 0,
    "eval": False,
    "num_workers": 0,

    # Distributed training parameters
    "world_size": 1,
    "dist_url": "env://"
}
mdetr_args = Args(**mdetr_args_dict)



medrpg_args_dict = {
    'lr': 1e-4,
    'lr_bert': 1e-5,
    'lr_visu_cnn': 1e-5,
    'lr_visu_tra': 1e-5,
    'batch_size': 8,
    'weight_decay': 1e-4,
    'epochs': 90,
    'lr_power': 0.9,
    'clip_max_norm': 0.0,
    'eval': False,
    'optimizer': 'adamw',
    'lr_scheduler': 'step',
    'lr_drop': 60,
    'model_name': 'TransVG_ca', 
    'memory': False,
    'm_threadHead': 4,
    'm_size': 2048,
    'm_dim': 256,
    'm_topK': 32,
    'm_linearMode': 'noLinear',
    'm_MMHA_outLinear': False,
    'm_resLearn': False,
    'btloss': False,
    'lossbt_type': 'l1',
    'lossbt_weight': 1.0,
    'GNpath': '/scratch/user/uqwzha16/MedRPG/data/MS_CXR/Genome',
    'GNClsType': 'mcls',
    'gnlossWeightBase': 0.1,
    'CAsampleType': 'random',
    'CAsampleNum': 5,
    'CAlossWeightBase': 0.05,
    'CATextPoolType': 'cls',
    'CATemperature': 0.1,
    'CAMode': 'max_image_lcpTriple', # to change
    'ConsLossWeightBase': 1.0,
    'ablation': 'none',
    'backbone': 'resnet50',
    'dilation': False, # to change, true -> 16, false-> 8
    'position_embedding': 'sine',
    'enc_layers': 6,
    'dec_layers': 6,
    'dim_feedforward': 2048,
    'hidden_dim': 256,
    'dropout': 0.1,
    'nheads': 8,
    'num_queries': 1,
    'pre_norm': False,
    'imsize': 640, # to change
    'emb_size': 512,
    'bert_enc_num': 12,
    'detr_enc_num': 6,
    'vl_dropout': 0.1,
    'vl_nheads': 8,
    'vl_hidden_dim': 256,
    'vl_dim_feedforward': 2048,
    'vl_enc_layers': 6,
    'contrastive_dim': 256, 
    'data_root': '/scratch/user/uqwzha16/MedRPG/ln_data/',
    'split_root': '/scratch/user/uqwzha16/MedRPG/data',
    'dataset': 'MS_CXR', # to change
    'max_query_len': 20,
    'output_dir': 'outputs/MIMIC', # to change
    'device': 'cuda',
    'seed': 13,
    'resume': '',
    'resume_model_only': False,
    'detr_model': '/scratch/user/uqwzha16/MedRPG/checkpoints/detr-r50-unc.pth',
    'bert_model': 'bert-base-uncased',
    'light': False,
    'start_epoch': 0,
    'start_batch': 0,
    'num_workers': 0,
    'world_size': 1,
    'dist_url': 'env://',
    'aug_blur': True,
    'aug_crop': True,
    'aug_scale': True,
    'aug_translate': True,
    'distributed': False,  # Assuming this parameter was intended but not included in argparse setup
    'freeze_text_encoder': True,
    'contrastive_loss': False,
}
medrpg_args = Args(**medrpg_args_dict)



def build_args(model_name, dataset_name):
    """
    Combines model and dataset arguments into a single Args object based on their names.

    Parameters:
        model_name (str): The name of the model (e.g., 'mdetr', 'transvg').
        dataset_name (str): The name of the dataset (e.g., 'ms_cxr', 'mimic_anatomy').

    Returns:
        Args: A combined Args object with merged arguments.
    """
    # Mapping of model and dataset names to their respective args objects
    models = {
        'mdetr': mdetr_args,
        'transvg': transvg_args,
        'medrpg': medrpg_args
    }

    datasets = {
        'ms_cxr': ms_args,
        'mimic_anatomy': mimic_anatomy_args,
        'mimic_pathology': mimic_pathology_args
    }

    # Get the model and dataset args objects
    model_args = models.get(model_name.lower())
    dataset_args = datasets.get(dataset_name.lower())

    # Raise an error if the names are not found
    if not model_args:
        raise ValueError(f"Unknown model name: {model_name}")
    if not dataset_args:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Merge the dictionaries of both arguments
    combined_dict = {**model_args.__dict__, **dataset_args.__dict__}

    # Return a new Args object with the merged dictionary
    return Args(**combined_dict)



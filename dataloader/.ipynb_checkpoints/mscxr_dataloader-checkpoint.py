import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from utils.box_utils import xywh2xyxy, xyxy2xywh

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import ast
from PIL import Image, UnidentifiedImageError
import numpy as np
import random
from torch.utils.data import Subset
from utils.misc import *



class MSCXRDataset(Dataset):
    def __init__(self, args, split, transform=None):
        self.args = args
        self.annotations = torch.load(args.anno_dir + '/MS_CXR_' + split + '.pth')
        self.img_dir = args.img_dir
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        self.model = AutoModel.from_pretrained(args.bert_model)
    
    def __len__(self):
        return len(self.annotations)

    def tokenize_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.args.max_query_len, truncation=True)
        return inputs["input_ids"], inputs["attention_mask"]


    def __getitem__(self, idx):
        _, _, category_id, img_path, bbox, height, width, phrase = self.annotations[idx]
        try:
            img_fn = self.img_dir + '/' + img_path
            image = Image.open(img_fn).convert("RGB")
            image = np.array(image)
            w, h, _ = image.shape
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {img_path}")
            return None  # Indicate a failure to load the image
        bbox = np.array(bbox, dtype=int)
        bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        if self.transform:
            transformed = self.transform(image=image, bboxes=[bbox], class_labels = [category_id])
            transformed_image = transformed['image']
            transformed_bbox = transformed['bboxes']
        bbox_coordinates_tensor = torch.tensor(transformed_bbox)
        bbox_coordinates_tensor = xyxy2xywh(bbox_coordinates_tensor)/torch.tensor([self.args.imsize, self.args.imsize, self.args.imsize, self.args.imsize])

        text_ids, text_masks = self.tokenize_text(phrase)
        results = {
            "image": transformed_image,
            "img_path": img_fn,
            "bbox_coordinates": bbox_coordinates_tensor,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "phrase": phrase,
            "category_id": category_id,
        }
        return results


def custom_collate_fn(batch):
    images = [item['image'] for item in batch]
    image_masks = [torch.zeros_like(image[0], dtype=torch.bool) for image in images]  # Create masks
    phrases = [item['phrase'] for item in batch]
    category_ids = [item['category_id'] for item in batch]
    images = torch.stack(images, dim=0)
    image_masks = torch.stack(image_masks, dim=0)
    text_ids_list = []
    text_masks_list = []
    for i, item in enumerate(batch):
        text_ids_list.extend(item['text_ids'])
        text_masks_list.extend(item['text_masks'])
    text_ids = torch.stack(text_ids_list, dim=0)
    text_masks = torch.stack(text_masks_list, dim=0)
    text_inds = torch.Tensor(np.arange(len(batch)))
    collated_batch = {
        "images": NestedTensor(images, image_masks),
        "img_paths": [item['img_path'] for item in batch],
        "bbox_coordinates": torch.stack([item['bbox_coordinates'] for item in batch], dim=0),
        "text": NestedTensor(text_ids, text_masks),
        "text_inds": text_inds,
        "phrases": phrases,
        "category_ids": category_ids,
    }
    return collated_batch


def get_transforms(args, split, with_bboxes=True):
    mean = 0.485
    std = 0.229
    if split == 'train':
        transforms = [
            A.LongestMaxSize(max_size=args.imsize),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            A.RandomCrop(height=args.imsize - 5, width=args.imsize - 5, always_apply=False, p=0.5),
            A.PadIfNeeded(min_height=args.imsize, min_width=args.imsize, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),]
    else:
        transforms = [
            A.LongestMaxSize(max_size=args.imsize),
            A.PadIfNeeded(min_height=args.imsize, min_width=args.imsize, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    if with_bboxes: 
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc', 
                label_fields=['class_labels']
            )
        )
    else:
        return A.Compose(transforms)

def get_mscxr_dataloaders(args):

    train_dataset = MSCXRDataset(args, 'train', transform=get_transforms(args, 'train'))
    val_dataset = MSCXRDataset(args, 'val', transform=get_transforms(args, 'val'))
    test_dataset = MSCXRDataset(args, 'test', transform=get_transforms(args, 'val'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

def get_viz_dataloader(args):
    test_dataset = MSCXRDataset(args, 'test', transform=get_transforms(args, 'val'))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    return test_loader


def get_optimizer(args, model):
    visu_cnn_param = [p for n, p in model.named_parameters() if (("visumodel" in n) and ("backbone" in n) and p.requires_grad)]
    visu_tra_param = [p for n, p in model.named_parameters() if (("visumodel" in n) and ("backbone" not in n) and p.requires_grad)]
    text_tra_param = [p for n, p in model.named_parameters() if (("textmodel" in n) and p.requires_grad)]
    rest_param = [p for n, p in model.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    param_list = [{"params": rest_param},
                {"params": visu_cnn_param, "lr": args.lr_visu_cnn},
                {"params": visu_tra_param, "lr": args.lr_visu_tra},
                {"params": text_tra_param, "lr": args.lr_bert},
                ]
    optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer



def get_subset_dataloader(args, percentage):
    dataset = MSCXRDataset(args, 'train', transform=get_transforms(args, 'train'))
    # Group indices by category
    category_to_indices = {}
    for idx, data in enumerate(dataset):
        category_id = data['category_id']
        if category_id not in category_to_indices:
            category_to_indices[category_id] = []
        category_to_indices[category_id].append(idx)
    
    # Sample a subset of indices for each category
    sampled_indices = []
    for category_id, indices in category_to_indices.items():
        sample_size = int(len(indices) * percentage)
        sampled_indices.extend(random.sample(indices, sample_size))
        
    # Create Subset dataset with sampled indices
    subset_dataset = Subset(dataset, sampled_indices)
    
    # Create DataLoader with the subset dataset
    subset_dataloader = DataLoader(
        subset_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    return subset_dataloader
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
    # Filter out None values from the batch (in case some items failed to load)
    batch = [item for item in batch if item is not None]

    # Images and their masks
    images = [item['image'] for item in batch]
    image_masks = [torch.zeros_like(image[0], dtype=torch.bool) for image in images]  # Create masks
    images = torch.stack(images, dim=0)
    image_masks = torch.stack(image_masks, dim=0)

    # Text data (phrases and tokenized representations)
    phrases = [item['phrase'] for item in batch]
    category_ids = [item['category_id'] for item in batch]

    # Check if negative phrases exist
    negative_phrases = None
    if "negative_phrases" in batch[0]:
        negative_phrases = [item['negative_phrases'] for item in batch]
        negative_phrases = list(map(list, zip(*negative_phrases)))
        
    text_ids_list = []
    text_masks_list = []
    for item in batch:
        text_ids_list.extend(item['text_ids'])
        text_masks_list.extend(item['text_masks'])
    text_ids = torch.stack(text_ids_list, dim=0)
    text_masks = torch.stack(text_masks_list, dim=0)

    # Bounding box coordinates
    bbox_coordinates = torch.stack([item['bbox_coordinates'] for item in batch], dim=0)

    # Additional indexing
    text_inds = torch.Tensor(np.arange(len(batch)))

    # Collate all items into a batch dictionary
    collated_batch = {
        "images": NestedTensor(images, image_masks),
        "img_paths": [item['img_path'] for item in batch],
        "bbox_coordinates": bbox_coordinates,
        "text": NestedTensor(text_ids, text_masks),
        "text_inds": text_inds,
        "phrases": phrases,
        "category_ids": category_ids,
    }

    # Include negative phrases if they exist
    if negative_phrases is not None:
        collated_batch["negative_phrases"] = negative_phrases

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
    """
    Create train, val, and test dataloaders. Includes negative phrases if `args.neg` is True.
    """
    # Select the appropriate dataset class based on the `neg` flag
    DatasetClass = MSCXRDatasetWithNegatives if getattr(args, "neg", False) else MSCXRDataset

    # Create datasets
    train_dataset = DatasetClass(args, 'train', transform=get_transforms(args, 'train'))
    val_dataset = DatasetClass(args, 'val', transform=get_transforms(args, 'val'))
    test_dataset = DatasetClass(args, 'test', transform=get_transforms(args, 'val'))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

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




class MSCXRDatasetWithNegatives(MSCXRDataset):
    def __init__(self, args, split, transform=None):
        super().__init__(args, split, transform)
        self.negative_phrases_per_image = args.negative_phrases_per_image
        self.image_to_categories, self.category_to_phrases = self._build_mappings()

    def _build_mappings(self):
        """
        Build two mappings:
        1. `image_to_categories`: Maps each image to the set of its associated categories.
        2. `category_to_phrases`: Maps each category to its associated phrases.
        """
        image_to_categories = defaultdict(set)
        category_to_phrases = defaultdict(set)

        for item in self.annotations:
            _, _, category_id, img_path, _, _, _, phrase = item
            dicom_id = img_path.split('/')[-1].split('.')[0]  # Extract DICOM ID from img_path
            image_to_categories[dicom_id].add(category_id)
            category_to_phrases[category_id].add(phrase)

        # Convert sets to lists for easy random sampling
        category_to_phrases = {k: list(v) for k, v in category_to_phrases.items()}
        return image_to_categories, category_to_phrases

    def get_negative_phrases(self, dicom_id):
        """
        Get `negative_phrases_per_image` random phrases from disease categories 
        that are not associated with the current image.
        """
        # Get all categories associated with the image
        current_categories = self.image_to_categories[dicom_id]

        # Get available categories by excluding current categories
        available_categories = [
            category for category in self.category_to_phrases.keys() 
            if category not in current_categories
        ]

        # Sample negative categories
        negative_categories = random.sample(available_categories, k=min(self.negative_phrases_per_image, len(available_categories)))

        # Select one random phrase per negative category
        negative_phrases = [random.choice(self.category_to_phrases[cat]) for cat in negative_categories]
        return negative_phrases

    def __getitem__(self, idx):
        # Fetch the original data
        item = super().__getitem__(idx)
        if item is None:  # Handle cases where the image couldn't be loaded
            return None

        # Extract DICOM ID from image path
        img_path = item["img_path"]
        dicom_id = img_path.split('/')[-1].split('.')[0]

        # Get negative phrases for the current image
        negative_phrases = self.get_negative_phrases(dicom_id)

        # Append negative phrases to the results
        item["negative_phrases"] = negative_phrases
        return item


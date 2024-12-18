/'import albumentations as A
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
from utils.misc import *
import random


class ProcessedMIMICDataset(Dataset):
    def __init__(self, args, split, transform=None):
        """
        Args:
            args: Arguments containing configurations.
            split (string): One of 'train', 'valid', or 'test'.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Load the annotations
        csv_file = args.anno_dir
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = args.img_dir
        self.transform = transform
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        self.model = AutoModel.from_pretrained(args.bert_model)
        self.pathology_labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]


    def __len__(self):
        return len(self.annotations)
    
    def tokenize_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.args.max_query_len, truncation=True, padding="max_length")
        ids = inputs['input_ids']
        masks = inputs['attention_mask']
        return ids, masks
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_fn = row['img_path']
        sentence = row['sentence']
        bbox = row['bounding_box']

        # Parse the bounding box string to a list
        try:
            bbox_coordinates = ast.literal_eval(bbox)
        except (ValueError, SyntaxError):
            print(f"Invalid bounding box at index {idx}")
            return None

        # Adjust the image path as per your specification
        img_fn = '/scratch/project/cxrdata/' + img_fn.split('mimic-cxr\\')[-1].replace('\\', '/')
        if not os.path.exists(img_fn):
            print(f"Image file not found: {img_fn}")
            return None


        try:
            image = Image.open(img_fn).convert("RGB")
            image = np.array(image)
            h, w, _ = image.shape

            # Ensure bounding box coordinates are within image bounds
            x_min = max(0, min(w, bbox_coordinates[0]))
            y_min = max(0, min(h, bbox_coordinates[1]))
            x_max = max(0, min(w, bbox_coordinates[2]))
            y_max = max(0, min(h, bbox_coordinates[3]))
            bbox_coordinates = [x_min, y_min, x_max, y_max]
            
            bbox_labels = [1]
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image, bboxes=[bbox_coordinates], bbox_labels=bbox_labels)
                image = transformed['image']
                bbox_coordinates = transformed['bboxes'][0]
            else:
                # If no transform is provided, convert the image to tensor
                transform = ToTensorV2()
                image = transform(image=image)['image']

            # Convert bounding box to tensor and normalize
            bbox_tensor = torch.tensor(bbox_coordinates)
            bbox_tensor = xyxy2xywh(bbox_tensor) / torch.tensor(
                [self.args.imsize, self.args.imsize, self.args.imsize, self.args.imsize]
            )
            text_ids, text_masks = self.tokenize_text(sentence)

            # Prepare the sample
            sample = {
                'image': image,
                'phrase': sentence,
                'bbox_coordinate': bbox_tensor,
                "text_ids": text_ids,
                "text_masks": text_masks,
                'img_fn': img_fn
            }
            return sample

        except Exception as e:
            print(f"Error processing image {img_fn}: {e}")
            return None
        
        
def custom_collate_fn(batch):
    # Remove None samples
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    images = [item['image'] for item in batch]
    image_masks = [torch.zeros_like(image[0], dtype=torch.bool) for image in images]  # Create masks

    images = torch.stack(images, dim=0)
    image_masks = torch.stack(image_masks, dim=0)
    phrases = [item['phrase'] for item in batch]
    bbox_coordinates = [item['bbox_coordinate'] for item in batch]
    img_fns = [item['img_fn'] for item in batch]
    text_ids_list = []
    text_masks_list = []
    for i, item in enumerate(batch):
        text_ids_list.extend(item['text_ids'])
        text_masks_list.extend(item['text_masks'])
    text_ids = torch.stack(text_ids_list, dim=0)
    text_masks = torch.stack(text_masks_list, dim=0)
    bbox_coordinates = torch.stack(bbox_coordinates, dim=0)

    return {
        "images": NestedTensor(images, image_masks),
        'phrases': phrases,
        'bbox_coordinates': bbox_coordinates,
        "text": NestedTensor(text_ids, text_masks),
        'img_paths': img_fns
    }

def get_transforms(args, split):
    mean = 0.485
    std = 0.229
    if split == 'train':
        return A.Compose([
            A.LongestMaxSize(max_size=args.imsize),  # Resize the image to have the longer side equal to imsize
            A.PadIfNeeded(min_height=args.imsize, min_width=args.imsize, border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad the shorter side to imsize
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            # A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            # A.RandomCrop(height=args.imsize - 5, width=args.imsize - 5, always_apply=False, p=0.5),
            A.PadIfNeeded(min_height=args.imsize, min_width=args.imsize, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels'], min_area=0.0, min_visibility=0.9)
        )
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=args.imsize),  # Resize the image to have the longer side equal to imsize
            A.PadIfNeeded(min_height=args.imsize, min_width=args.imsize, border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad the shorter side to imsize
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels'], min_area=0.0, min_visibility=0.9)
        )




def get_processed_mimic_dataloaders(args):
    train_dataset = ProcessedMIMICDataset(args, 'train', transform=get_transforms(args, 'train'))
    val_dataset = ProcessedMIMICDataset(args, 'valid', transform=get_transforms(args, 'val'))
    test_dataset = ProcessedMIMICDataset(args, 'test', transform=get_transforms(args, 'val'))

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


def get_small_val_loader(args, num_samples=100):
    """
    Get a smaller validation dataloader with a specified number of samples.
    
    Args:
        args: The arguments containing dataset paths and other configurations.
        num_samples (int): Number of samples to include in the validation set. Default is 100.
    
    Returns:
        DataLoader: A DataLoader object for the subset of the validation dataset.
    """
    # Load the full validation dataset
    val_dataset = ProcessedMIMICDataset(args, 'valid', transform=get_transforms(args, 'val'))
    
    # Check if the number of samples requested is greater than the dataset size
    num_samples = min(num_samples, len(val_dataset))
    
    # Create a subset of the validation dataset
    indices = list(range(num_samples))
    val_subset = torch.utils.data.Subset(val_dataset, indices)
    
    # Create the DataLoader for the subset
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    
    return val_loader
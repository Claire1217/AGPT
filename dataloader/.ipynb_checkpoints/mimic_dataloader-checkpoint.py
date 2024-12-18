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
from utils.misc import *

class MIMICDataset(Dataset):
    def __init__(self, args, split, transform=None, with_text=False):
        """                                                                                                                                                                                    
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        csv_file = args.anno_dir + '/' + split + '.csv'
        self.args = args
        self.img_dir = args.img_dir
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)
        self.with_text = with_text
        self.with_anatomy_encoding = args.anatomy
        if with_text:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.model = AutoModel.from_pretrained(args.bert_model)
        if args.anatomy:
            self.with_text = True
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.model = AutoModel.from_pretrained(args.bert_model)
            self.anatomies = self.get_anatomy_variations()
            self.frequencies = self.get_frequencies()
            self.anatomy_features, self.anatomy_masks = self.build_anatomy_encodings()



    def __len__(self):
        return len(self.annotations)
    
    
    def get_frequencies(self):
        """
        Define the frequencies of anatomical variations for each region.
        """
        return [
            [0.6, 0.3, 0.1],
            [0.4, 0.4, 0.2],
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.5, 0.3, 0.2],
            [0.6, 0.3, 0.1],
            [0.6, 0.3, 0.1],
            [0.4, 0.4, 0.2],
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.5, 0.3, 0.2],
            [0.6, 0.3, 0.1],
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.6, 0.3, 0.1],
            [0.6, 0.3, 0.1],
            [0.5, 0.3, 0.2],
            [0.6, 0.3, 0.1],
            [0.5, 0.3, 0.2],
            [0.6, 0.3, 0.1],
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
            [0.6, 0.3, 0.1],
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1]
        ]

    def get_anatomy_variations(self, num_variations=3):
        """
        Define the anatomical variations for each region.
        """
        anatomy_variations =  [
            ["right lung", "right pulmonary field", "right pulmonary parenchyma", "right lung zone", "right thoracic cavity"],
            ["right upper lung zone", "right upper lobe", "right apical lung region", "right apical region", "upper right lung"],
            ["right mid lung zone", "right middle lobe", "right mid lung field", "right central lung region", "middle right lung"],
            ["right lower lung zone", "right lower lobe", "right basal lung region", "right lung base", "right breast"],
            ["right hilar structures", "right hilum", "right perihilar region", "rigt central thoracic region", "right pulmonary hilum"],
            ["right apical zone", "right apex", "right apical lung region", "right uppermost lung region", "right lung tip"], 
            ["right costophrenic angle", "right CP angle", "right costo-phrenic sulcus", "right pleurodiaphragmatic recess", "right inferior lung corner"],
            ["right hemidiaphragm", "right diaphragm", "right diaphragmatic dome", "right subdiaphragmatic space", "right phrenic surface"],
            ["left lung", "left pulmonary field", "left pulmonary parenchyma", "left lung zone", "left breast"],
            ["left upper lung zone", "left upper lobe", "left apical lung region", "left apical region", "upper left lung"],
            ["left mid lung zone", "left middle lobe", "left mid lung field", "left central lung region", "middle left lung"],
            ["left lower lung zone", "left lower lobe", "left basal lung region", "left lung base", "left lower thoracic region"],
            ["left hilar structures", "left hilum", "left perihilar region", "left central thoracic region", "left pulmonary hilum"],
            ["left apical zone", "left apex", "left apical lung region", "left uppermost lung region", "left lung tip"],
            ["left costophrenic angle", "left CP angle", "left costo-phrenic sulcus", "left pleurodiaphragmatic recess", "left inferior lung corner"],
            ["left hemidiaphragm", "left diaphragm", "left diaphragmatic dome", "left subdiaphragmatic space", "left phrenic surface"],
            ["trachea", "bronchus", "tracheal shadow", "central airway", "tracheal passage"],
            ["spine", "vertebral column", "thoracic spine", "vertebral bodies", "spinal shadow"],
            ["right clavicle", "right collarbone", "right clavicular region", "right clavicle bone", "right shoulder girdle"],
            ["left clavicle", "left collarbone", "left clavicular region", "left clavicle bone", "left shoulder girdle"],
            ["aortic arch", "arch of the aorta", "aortic knob", "aortic shadow", "aortic contour"],
            ["mediastinum", "mediastinal structures", "mediastinal shadow", "central thoracic space", "mediastinal silhouette"],
            ["upper mediastinum", "superior mediastinum", "upper mediastinal region", "upper thoracic space", "superior mediastinal contour"],
            ["svc", "superior vena cava", "superior caval vein", "central venous shadow", "SVC shadow"],
            ["cardiac silhouette", "heart", "cardiac shadow", "ventricle", "atrium"],
            ["cavoatrial junction", "SVC-atrial junction", "superior vena cava and right atrium junction", "cavoatrial connection", "Cavo-atrial boundary"],
            ["right atrium", "right heart chamber", "right cardiac atrium", "right atrial silhouette", "right atrial contour"],
            ["carina", "carinal bifurcation", "tracheal bifurcation", "bronchial bifurcation", "carinal ridge"],
            ["abdomen", "abdominal cavity", "stomach", "liver", "abominal shadow"]]
        

        anatomy_slice = [sublist[:num_variations] for sublist in anatomy_variations]
        return anatomy_slice
    
    def build_anatomy_encodings(self):
        """
        Encode the anatomies,
        returning the feature tensors and mask tensors.
        """
        anatomy_features = []
        anatomy_masks = []
        
        for anatomy_list in self.get_anatomy_variations(self.args.anatomy_variation_count):
            feature_list = []
            mask_list = []
            for anatomy in anatomy_list:
                text_ids, text_masks = self.tokenize_text(anatomy)
                feature_list.append(text_ids)
                mask_list.append(text_masks)
            
            # Stack the features and masks for each anatomy list
            anatomy_features.append(torch.stack(feature_list))
            anatomy_masks.append(torch.stack(mask_list))
        anatomy_features = torch.stack(anatomy_features)  # Shape: (29, 3, feature_dim)
        anatomy_masks = torch.stack(anatomy_masks)  # Shape: (29, 3, mask_dim)
        return anatomy_features, anatomy_masks
            
    
    def get_random_anatomy_feature_mask(self):
        anatomy_features_selected = []
        anatomy_masks_selected = []
        anatomy_list = self.get_anatomy_variations()
        anatomy_names_selected = []
        for i in range(29):
            if self.args.anatomy_sampling == 'random':
                selected_index = np.random.choice(self.args.anatomy_variation_count)
                anatomy_names_selected.append(anatomy_list[i][selected_index])
                anatomy_features_selected.append(self.anatomy_features[i, selected_index])
                anatomy_masks_selected.append(self.anatomy_masks[i, selected_index])
            elif self.args.anatomy_sampling == 'frequency':
                selected_index = np.random.choice(self.args.anatomy_variation_count, p=self.frequencies[i])
                anatomy_names_selected.append(anatomy_list[i][selected_index])
                anatomy_features_selected.append(self.anatomy_features[i, selected_index])
                anatomy_masks_selected.append(self.anatomy_masks[i, selected_index])
        return anatomy_names_selected, torch.stack(anatomy_features_selected), torch.stack(anatomy_masks_selected)
    
    
    def pull_item(self, idx):
        row = self.annotations.iloc[idx]
        img_path = row['mimic_image_file_path'].split('images/')[-1]
        bbox_phrases = ast.literal_eval(row['bbox_phrases'])
        bbox_labels = ast.literal_eval(row['bbox_labels'])
        bbox_coordinates = ast.literal_eval(row['bbox_coordinates'])
        bbox_phrases_exists = ast.literal_eval(row['bbox_phrase_exists'])
        
        img_fn = os.path.join(self.img_dir, img_path)
        
        try:
            image = Image.open(img_fn).convert("RGB")
            image = np.array(image)
            print('opened sucess')
            w, h, _ = image.shape
        except Exception as e:
            print(f"Skipping image: {img_fn}")
            print(e)
            return None
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=bbox_coordinates, class_labels=bbox_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_labels = transformed['class_labels']
        
        bbox_coordinates_tensor = torch.tensor(transformed_bboxes)
        bbox_coordinates_tensor = torch.stack([xyxy2xywh(box)/torch.tensor([w, h, w, h]) for box in bbox_coordinates_tensor])

        return transformed_image, bbox_phrases, bbox_coordinates_tensor, bbox_phrases_exists, transformed_labels, img_fn
    
    def tokenize_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=50, truncation=True, padding="max_length")
        ids = inputs['input_ids']
        masks = inputs['attention_mask']
        return ids, masks
        
    def __getitem__(self, idx):
        item = self.pull_item(idx)
        if item is None:
            return self.__getitem__((idx + 1) % len(self.annotations))
        image, bbox_phrases, bbox_coordinates, bbox_phrases_exists, bbox_labels, img_path = item
        
        if len(bbox_coordinates) != 29:
            return self.__getitem__((idx + 1) % len(self.annotations))
        
        result = {
            "image": image,
            "img_path": img_path,
            "bbox_coordinates": bbox_coordinates,
        }

        if self.with_anatomy_encoding:
            anatomy_names, anatomy_features, anatomy_masks = self.get_random_anatomy_feature_mask()
            result.update({
                "anatomy_names": anatomy_names,
                "text_ids": anatomy_features,
                "text_masks": anatomy_masks, 
                
            })
            return result

        else:
            # Variable length depends on the number of pairs of region-phras (num_pairs)e for this idx
            # text_ids of shape [num_pairs, num_token]
            # text_masks of shape [num_pairs, num_token]
            # bbox_coordinates_exists of shape [num_pairs, 4]
            # bbox_labels_exists of shape [num_pairs], indicating the index of box
            phrases_exist = [phrase for phrase, exist in zip(bbox_phrases, bbox_phrases_exists) if exist]
            text_ids, text_masks = self.tokenize_text(phrases_exist)
            bbox_labels_exists = [label for label, exist in zip(bbox_labels, bbox_phrases_exists) if exist]
            bbox_coordinates_exists = [bbox for bbox, exist in zip(bbox_coordinates, bbox_phrases_exists) if exist]
            result.update({
                "text_ids": text_ids,
                "text_masks": text_masks,
                "bbox_coordinates_varibale_length": bbox_coordinates_exists,
                "bbox_labels": bbox_labels_exists
            })

            return result
        


def get_transforms(args, split, with_bboxes=True):
    mean = 0.485
    std = 0.229

    # Common transformations for all splits
    transform_list = [
        A.Resize(height=args.imsize, width=args.imsize),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    # Additional transformations for the training split
    if split == 'train':
        train_transforms = [
            A.Resize(height = args.imsize, width = args.imsize),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            A.Affine(
                mode=cv2.BORDER_CONSTANT, cval=0,
                translate_percent=(-0.02, 0.02),
                rotate=(-2, 2)
            ),
            A.RandomCrop(
                height=args.imsize - 5,
                width=args.imsize - 5,
                always_apply=False,
                p=0.5
            ),
            A.PadIfNeeded(
                min_height=args.imsize,
                min_width=args.imsize,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ]
        transform_list = train_transforms + transform_list

    if with_bboxes:
        return A.Compose(
            transform_list,
            bbox_params=A.BboxParams(
                format='pascal_voc', 
                label_fields=['class_labels']
            )
        )
    else:
        return A.Compose(transform_list)



def custom_collate_fn(batch):
    # Filter out None items and ensure bbox_coordinates has the correct shape
    batch = [item for item in batch if item is not None and item['bbox_coordinates'].shape[0] == 29]
    
    images = [item['image'] for item in batch]
    image_masks = [torch.zeros_like(image[0], dtype=torch.bool) for image in images]  # Create masks

    images = torch.stack(images, dim=0)
    image_masks = torch.stack(image_masks, dim=0)

    collated_batch = {
        "images": NestedTensor(images, image_masks),
        "img_paths": [item['img_path'] for item in batch],
        "bbox_coordinates": torch.stack([item['bbox_coordinates'] for item in batch], dim=0)
    }
    
    if 'text_ids' in batch[0]:
        text_ids_list = []
        text_masks_list = []
        text_inds = []
        for i, item in enumerate(batch):
            text_ids_list.extend(item['text_ids'])
            text_masks_list.extend(item['text_masks'])
            text_inds.extend([i] * 29)
        text_ids = torch.stack(text_ids_list, dim=0)
        text_masks = torch.stack(text_masks_list, dim=0)
        collated_batch.update({
            "text": NestedTensor(text_ids.squeeze(), text_masks.squeeze()),
            "text_inds": torch.Tensor(text_inds)
        })
    
    return collated_batch


def get_mimic_dataloaders(args):

    train_dataset = MIMICDataset(args, 'train', transform=get_transforms(args, 'train'))
    val_dataset = MIMICDataset(args, 'valid', transform=get_transforms(args, 'val'))
    test_dataset = MIMICDataset(args, 'test', transform=get_transforms(args, 'val'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader



def get_viz_dataloader(args):
    test_dataset = MIMICDataset(args, 'test', transform=get_transforms(args, 'val'))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    return test_loader
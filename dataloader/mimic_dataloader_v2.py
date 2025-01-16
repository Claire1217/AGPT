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
import random
from utils.box_utils import sampleNegBBox

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
        anatomy_list = self.get_anatomy_variations(self.args.anatomy_variation_count)
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
    
    
    def tokenize_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.args.max_query_len, truncation=True, padding="max_length")
        ids = inputs['input_ids']
        masks = inputs['attention_mask']
        return ids, masks
        
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_fn = row['mimic_image_file_path']
        bbox_coordinates = ast.literal_eval(row['bbox_coordinates'])
  
        if not os.path.exists(img_fn):
            img_fn = self.img_dir + '/' + img_fn.split('mimic-cxr\\')[-1]
        try:
            image = Image.open(img_fn).convert("RGB")
            image = np.array(image)
            h, w, _ = image.shape

            # bound bbox coordinates (x1, y1, x2, y2) to image size
            bbox_coordinates = [[max(0, min(w, bbox[0])), max(0, min(h, bbox[1])), max(0, min(w, bbox[2])), max(0, min(h, bbox[3]))] for bbox in bbox_coordinates]

        # skip all the errors and exceptions and return None
        except Exception as e:
            print(f"Skipping image: {img_fn}")
            print(e)
            return None
        if self.transform:
            # anatomy_ids to be a list from 0 to 28
            ids = list(range(29))
            try:
                transformed = self.transform(image=image, bboxes=bbox_coordinates, anatomy_ids = ids)
            except Exception as e:
                print(f"Error in image: {img_fn}")
                print(f"bbox: {bbox_coordinates}")
                print(e)
                return None
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']

        
        bbox_coordinates_tensor = torch.tensor(transformed_bboxes)
        bbox_coordinates_tensor = torch.stack([xyxy2xywh(box)/torch.tensor([self.args.imsize, self.args.imsize, self.args.imsize, self.args.imsize]) for box in bbox_coordinates_tensor])
        results = {
            'image': transformed_image,
            'bbox_coordinates': bbox_coordinates_tensor,
            'img_path': img_fn,
        }
        if self.with_anatomy_encoding:
            anatomy_names, anatomy_features, anatomy_masks = self.get_random_anatomy_feature_mask()
            results['anatomy_names'] = anatomy_names
            results['anatomy_ids'] = anatomy_features
            results['anatomy_masks'] = anatomy_masks
        
#         if self.taco:
#             NegBBoxs = sampleNegBBox(bbox, self.args.CAsampleType, self.args.CAsampleNum, self.args.imsize, self.args.imsize)  # negative bbox
            
        
        return results

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
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['anatomy_ids'], min_area=0.0, min_visibility=0.9)
        )
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=args.imsize),  # Resize the image to have the longer side equal to imsize
            A.PadIfNeeded(min_height=args.imsize, min_width=args.imsize, border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad the shorter side to imsize
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['anatomy_ids'], min_area=0.0, min_visibility=0.9)
        )

def custom_collate_fn(batch):
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
    if 'anatomy_ids' in batch[0]:
        anatomy_names_list = []
        anatomy_ids_list = []
        anatomy_masks_list = []
        text_inds = []
        for i, item in enumerate(batch):
            anatomy_names_list.append(item['anatomy_names'])
            anatomy_ids_list.extend(item['anatomy_ids'])
            anatomy_masks_list.extend(item['anatomy_masks'])
            text_inds.extend([i] * 29)
        anatomy_ids = torch.stack(anatomy_ids_list, dim=0)
        anatomy_masks = torch.stack(anatomy_masks_list, dim=0)
        collated_batch.update({
            "phrases" : anatomy_names_list,
            "text": NestedTensor(anatomy_ids.squeeze(), anatomy_masks.squeeze()),
            "text_inds": torch.Tensor(text_inds)
        })
    return collated_batch

def custom_collate_fn_negbox(batch):
    batch = [item for item in batch if item is not None]
    images = [item['image'] for item in batch]
    image_masks = [torch.zeros_like(image[0], dtype=torch.bool) for image in images]  # Create masks

    images = torch.stack(images, dim=0)
    image_masks = torch.stack(image_masks, dim=0)
    imsize = images[0].shape[-1]
    NegBBoxs = sampleNegBBox(bbox, 'random', 5, imsize, imsize)
    [np.array(negBBox, dtype=np.float32) for negBBox in NegBBoxs]

    collated_batch = {
        "images": NestedTensor(images, image_masks),
        "img_paths": [item['img_path'] for item in batch],
        "bbox_coordinates": torch.stack([item['bbox_coordinates'] for item in batch], dim=0)
    }
    if 'anatomy_ids' in batch[0]:
        anatomy_names_list = []
        anatomy_ids_list = []
        anatomy_masks_list = []
        text_inds = []
        for i, item in enumerate(batch):
            anatomy_names_list.append(item['anatomy_names'])
            anatomy_ids_list.extend(item['anatomy_ids'])
            anatomy_masks_list.extend(item['anatomy_masks'])
            text_inds.extend([i] * 29)
        anatomy_ids = torch.stack(anatomy_ids_list, dim=0)
        anatomy_masks = torch.stack(anatomy_masks_list, dim=0)
        collated_batch.update({
            "phrases" : anatomy_names_list,
            "text": NestedTensor(anatomy_ids.squeeze(), anatomy_masks.squeeze()),
            "text_inds": torch.Tensor(text_inds)
        })
    return collated_batch

def custom_collate_fn_repeat(batch):
    
    batch = [item for item in batch if item is not None and item['bbox_coordinates'].shape[0] == 29]
    
    images = [item['image'] for item in batch]
    image_masks = [torch.zeros_like(image[0], dtype=torch.bool) for image in images]  # Create masks
    h, w = images[0].shape[-2:]
    # Repeat each image 29 times to match the number of text entries
    images = torch.stack(images, dim=0)  # Shape: [batch, 3, 256, 256]
    images = images.unsqueeze(1).repeat(1, 29, 1, 1, 1)  # Shape: [batch, 29, 3, 256, 256]
    images = images.view(-1, 3, h, w)  # Flatten to [batch * 29, 3, 256, 256]

    # Similarly, repeat the image masks
    image_masks = torch.stack(image_masks, dim=0)
    image_masks = image_masks.unsqueeze(1).repeat(1, 29, 1, 1)  # Shape: [batch, 29, 256, 256]
    image_masks = image_masks.view(-1, h, w)  # Flatten to [batch * 29, 256, 256]

    collated_batch = {
        "images": NestedTensor(images, image_masks),
        "img_paths": [item['img_path'] for item in batch for _ in range(29)],  # Repeat img_paths 29 times
        "bbox_coordinates": torch.stack([item['bbox_coordinates'] for item in batch], dim=0).view(-1, 4)  # Flatten bbox coordinates
    }

    if 'anatomy_ids' in batch[0]:
        anatomy_names_list = []
        anatomy_ids_list = []
        anatomy_masks_list = []
        text_inds = []
        for i, item in enumerate(batch):
            anatomy_names_list.append(item['anatomy_names'])
            anatomy_ids_list.extend(item['anatomy_ids'])
            anatomy_masks_list.extend(item['anatomy_masks'])
            text_inds.extend([i * 29 + j for j in range(29)])  # Adjust text_inds for the repeated structure
        
        anatomy_ids = torch.stack(anatomy_ids_list, dim=0)
        anatomy_masks = torch.stack(anatomy_masks_list, dim=0)
        collated_batch.update({
            "phrases": anatomy_names_list,
            "text": NestedTensor(anatomy_ids.squeeze(), anatomy_masks.squeeze()),
            "text_inds": torch.Tensor(text_inds)
        })

    return collated_batch



def custom_collate_fn_repeat_random5(batch):
    ind = 3
    batch = [item for item in batch if item is not None and item['bbox_coordinates'].shape[0] == 29]
    
    images = [item['image'] for item in batch]
    # Dynamically get the height and width of the first image
    h, w = images[0].shape[-2:]
    
    image_masks = [torch.zeros_like(image[0], dtype=torch.bool) for image in images]  # Create masks
    
    # Stack images
    images = torch.stack(images, dim=0)  # Shape: [batch, 3, h, w]

    collated_batch = {
        "img_paths": [],
        "bbox_coordinates": [],
    }

    if 'anatomy_ids' in batch[0]:
        anatomy_names_list = []
        anatomy_ids_list = []
        anatomy_masks_list = []
        text_inds = []
        final_images = []
        final_image_masks = []
        for i, item in enumerate(batch):
            # Select 5 random anatomies
            selected_indices = random.sample(range(29), ind)

            # Repeat the image and mask 5 times for selected anatomies
            selected_images = images[i].unsqueeze(0).repeat(ind, 1, 1, 1)  # Shape: [5, 3, h, w]
            selected_image_masks = image_masks[i].unsqueeze(0).repeat(ind, 1, 1)  # Shape: [5, h, w]
            
            final_images.append(selected_images)
            final_image_masks.append(selected_image_masks)

            collated_batch["img_paths"].extend([item['img_path']] * ind)  # Repeat img_path 5 times
            
            # Select bbox_coordinates and anatomy details for the chosen anatomies
            collated_batch["bbox_coordinates"].extend([item['bbox_coordinates'][idx] for idx in selected_indices])

            anatomy_names_list.extend([item['anatomy_names'][idx] for idx in selected_indices])
            anatomy_ids_list.extend([item['anatomy_ids'][idx] for idx in selected_indices])
            anatomy_masks_list.extend([item['anatomy_masks'][idx] for idx in selected_indices])
            text_inds.extend([i * ind + j for j in range(ind)])  # Adjust text_inds for the repeated structure
        
        final_images = torch.cat(final_images, dim=0)  # Shape: [batch_size * 5, 3, h, w]
        final_image_masks = torch.cat(final_image_masks, dim=0)  # Shape: [batch_size * 5, h, w]
        collated_batch["images"] = NestedTensor(final_images, final_image_masks)

        anatomy_ids = torch.stack(anatomy_ids_list, dim=0)
        anatomy_masks = torch.stack(anatomy_masks_list, dim=0)
        collated_batch.update({
            "phrases": anatomy_names_list,
            "text": NestedTensor(anatomy_ids.squeeze(), anatomy_masks.squeeze()),
            "text_inds": torch.Tensor(text_inds)
        })
    
    # Convert bbox_coordinates to tensor
    collated_batch["bbox_coordinates"] = torch.stack(collated_batch["bbox_coordinates"], dim=0)

    return collated_batch


def get_mimic_dataloaders_repeat(args):
    train_dataset = MIMICDataset(args, 'train', transform=get_transforms(args, 'train'))
    val_dataset = MIMICDataset(args, 'valid', transform=get_transforms(args, 'val'))
    test_dataset = MIMICDataset(args, 'test', transform=get_transforms(args, 'val'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn_repeat)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn_repeat)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn_repeat)

    return train_loader, val_loader, test_loader

def get_mimic_dataloaders_repeat_random5(args):
    train_dataset = MIMICDataset(args, 'train', transform=get_transforms(args, 'train'))
    val_dataset = MIMICDataset(args, 'valid', transform=get_transforms(args, 'val'))
    test_dataset = MIMICDataset(args, 'test', transform=get_transforms(args, 'val'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn_repeat_random5)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn_repeat_random5)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn_repeat_random5)

    return train_loader, val_loader, test_loader


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
    val_dataset = MIMICDataset(args, 'valid', transform=get_transforms(args, 'val'))
    
    # Check if the number of samples requested is greater than the dataset size
    num_samples = min(num_samples, len(val_dataset))
    
    # Create a subset of the validation dataset
    indices = list(range(num_samples))
    val_subset = torch.utils.data.Subset(val_dataset, indices)
    
    # Create the DataLoader for the subset
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn_repeat)
    
    return val_loader


def sampleNegBBox(box, CAsampleType, CAsampleNum, w=640, h=640):
    assert CAsampleType in ['random', 'attention', 'crossImage', 'crossBatch']
    index = 0
    negBox_list = []
    # ori_center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
    ori_w, ori_h = box[2]-box[0], box[3]-box[1]
    flag=0
    while index < CAsampleNum:
        flag += 1
        # print(flag)
        if CAsampleType == 'random':
            xNeg = torch.randint(1, w, (1,))
            yNeg = torch.randint(1, h, (1,))
            wNeg = ori_w + random.randint(torch.round(-ori_w * 0.1), torch.round(ori_w * 0.1))
            hNeg = ori_h + random.randint(torch.round(-ori_h * 0.1), torch.round(ori_h * 0.1))
        elif CAsampleType == 'attention':
            pass

        negBox = torch.zeros([4])
        negBox[0], negBox[1], negBox[2], negBox[3] = xNeg - 0.5 * wNeg, yNeg - 0.5 * hNeg, xNeg + 0.5 * wNeg, yNeg + 0.5 * hNeg
        negBox = torch.round(negBox)
        # 加入越界条件筛选 invalid bbox
        if negBox[0] < 0 or negBox[1] < 0 or negBox[2] >= w or negBox[3] >= h:
            continue
        # 加入box冲突条件筛选 invalid bbox
        iou, union = box_iou(box.unsqueeze(0), negBox.unsqueeze(0))
        if iou > 0.25 and flag < 300:
            continue
        negBox_list.append(negBox)
        index += 1
        
    return negBox_list



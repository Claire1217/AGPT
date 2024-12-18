import numpy as np
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def visualBBox(image_path, pred_box, bbox, output_dir):
    real_im = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(real_im)
    color_gt = (255, 0, 0) # 红色代表gt
    color_method1 = (0, 255, 0) # 绿色代表预测
    if bbox is not None:
        draw.rectangle(bbox, outline=color_gt, width=2)
    draw.rectangle(pred_box, outline=color_method1, width=2)
    save_path = os.path.basename(image_path)
    real_im.save('{}/{}.png'.format(output_dir, save_path))
    del draw

def is_normalized_bbox(bbox):
    """
    Check if the bounding box coordinates are normalized (between 0 and 1).

    Parameters:
    bbox (list or tuple): Bounding box coordinates.

    Returns:
    bool: True if the coordinates are normalized, False otherwise.
    """
    return all(0 <= coord <= 1 for coord in bbox)

def convert_bbox_to_absolute(bbox, img_width, img_height, bbox_type):
    """
    Convert normalized bounding box coordinates to absolute coordinates.

    Parameters:
    bbox (list or tuple): Bounding box coordinates.
    img_width (int): Width of the image.
    img_height (int): Height of the image.
    bbox_type (str): Type of the bounding box format ('xyxy' or 'xywh').

    Returns:
    list: Bounding box coordinates in absolute format.
    """
    if bbox_type == 'xyxy':
        x1, y1, x2, y2 = bbox
        return [x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height]
    elif bbox_type == 'xywh':
        x, y, w, h = bbox
        return [x * img_width, y * img_height, w * img_width, h * img_height]


def visualPosNegBBoxes(image_path, pos_bbox, neg_bboxes=None, bbox_type='xyxy'):
    # Open the image file
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # Convert and plot the positive bounding box
    if is_normalized_bbox(pos_bbox):
        pos_bbox = convert_bbox_to_absolute(pos_bbox, img_width, img_height, bbox_type)
    
    if bbox_type == 'xyxy':
        x1, y1, x2, y2 = pos_bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none', label='Positive')
    elif bbox_type == 'xywh':
        x, y, w, h = pos_bbox
        rect = patches.Rectangle((x - 0.5 * w, y - 0.5 * h), w, h, linewidth=2, edgecolor='red', facecolor='none', label='Positive')
    ax.add_patch(rect)
    
    # Convert and plot the negative bounding boxes
    if neg_bboxes:
        for bbox in neg_bboxes:
            if is_normalized_bbox(bbox):
                bbox = convert_bbox_to_absolute(bbox, img_width, img_height, bbox_type)
            
            if bbox_type == 'xyxy':
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none', label='Negative')
            elif bbox_type == 'xywh':
                x, y, w, h = bbox
                rect = patches.Rectangle((x - 0.5 * w, y - 0.5 * h), w, h, linewidth=2, edgecolor='green', facecolor='none', label='Negative')
            ax.add_patch(rect)
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # Show the plot
    plt.axis('off')
    plt.show()
    

import matplotlib.pyplot as plt
import cv2
import numpy as np

def visual_anatomy_boxes(img_path, pred_boxes, gt_boxes):
    """
    Visualize the ground truth and predicted bounding boxes for each image.
    
    Parameters:
    - img_path: Path to the image file.
    - pred_boxes: Tensor of shape (29, 4) containing the predicted bounding boxes.
    - gt_boxes: Tensor of shape (29, 4) containing the ground truth bounding boxes.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(6, 5, figsize=(15, 18))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(pred_boxes):
            pred_box = pred_boxes[i].numpy()
            gt_box = gt_boxes[i].numpy()
            
            # Draw both boxes on the same image
            img_copy = img.copy()
            
            # Draw ground truth box in green
            cv2.rectangle(img_copy, 
                          (int(gt_box[0]), int(gt_box[1])), 
                          (int(gt_box[2]), int(gt_box[3])), 
                          (0, 255, 0), 2)
            
            # Draw predicted box in red
            cv2.rectangle(img_copy, 
                          (int(pred_box[0]), int(pred_box[1])), 
                          (int(pred_box[2]), int(pred_box[3])), 
                          (255, 0, 0), 2)
            
            ax.imshow(img_copy)
            ax.axis('off')
            ax.set_title(f'Box {i+1}')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
# pred_boxes and gt_boxes should be tensors or arrays of shape (29, 4)
# visual_anatomy_boxes('path_to_image.jpg', pred_boxes, gt_boxes)
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

def denormalize_image(image, mean, std):
    """
    Denormalizes the image using the provided mean and standard deviation.
    
    Parameters:
    - image: Normalized image tensor.
    - mean: List or tuple with the mean values for each channel.
    - std: List or tuple with the standard deviation values for each channel.
    
    Returns:
    - Denormalized image as a numpy array.
    """
    mean = np.array(mean)
    std = np.array(std)
    
    # Reverse the normalization process: (image * std) + mean
    image = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    image = (image * std) + mean
    
    # Convert image from range [0, 1] to [0, 255]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image

def visualize_dataloader_item(dataset, idx, color=(255, 0, 0), thickness=1):
    """
    Visualizes the image and bounding boxes for a given index from the dataset.

    Parameters:
    - dataset: The dataset object (your custom dataset class).
    - idx: The index of the image in the dataset.
    - color: The color of the bounding boxes. Default is red.
    - thickness: The thickness of the bounding box lines. Default is 2.
    """
    # Get the item from the dataset
    item = dataset[idx]

    # Check if item was successfully loaded
    if item is None:
        print(f"Item at index {idx} could not be loaded.")
        return
    
    # Extract the image and bounding boxes
    image = item['image']
    bbox_coordinates = item['bbox_coordinates']

    # Denormalize the image
    mean = [0.485, 0.456, 0.406]  # Example mean values used in normalization
    std = [0.229, 0.224, 0.225]   # Example std values used in normalization
    image_np = denormalize_image(image, mean, std)

    # Convert bbox from relative coordinates to absolute pixel values
    bboxes = bbox_coordinates * torch.tensor([args.imsize, args.imsize, args.imsize, args.imsize])

    # Convert the bbox format from xywh to xyxy
    bboxes = torch.stack([xywh2xyxy(box) for box in bboxes])

    # Draw each bounding box on the image
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)  # Ensure coordinates are integers
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, thickness)
    
    # Convert image from BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

# Example usage for visualizing train_dataset[0]
# visualize_dataloader_item(train_dataset, 1)

anatomy_names = ['right lung', 'right upper lung zone', 'right mid lung zone', 'right lower lung zone', 'right hilar structures', 'right apical zone', 'right costophrenic angle', 'right hemidiaphragm', 'left lung', 'left upper lung zone', 'left mid lung zone', 'left lower lung zone', 'left hilar structures', 'left apical zone', 'left costophrenic angle', 'left hemidiaphragm', 'trachea', 'spine', 'right clavicle', 'left clavicle', 'aortic arch', 'mediastinum', 'upper mediastinum', 'svc', 'cardiac silhouette', 'cavoatrial junction', 'right atrium', 'carina', 'abdomen']

def visualize_dataloader_item_multiple_plots(dataset, idx, color=(255, 0, 0), thickness=1):
    item = dataset[idx]

    # Check if item was successfully loaded
    if item is None:
        print(f"Item at index {idx} could not be loaded.")
        return
    
    # Extract the image and bounding boxes
    image = item['image']
    bbox_coordinates = item['bbox_coordinates']
    anatomy_names = item['anatomy_names']
    # Denormalize the image
    mean = [0.485, 0.456, 0.406]  # Example mean values used in normalization
    std = [0.229, 0.224, 0.225]   # Example std values used in normalization
    image_np = denormalize_image(image, mean, std)

    # Convert bbox from relative coordinates to absolute pixel values
    bboxes = bbox_coordinates * torch.tensor([args.imsize, args.imsize, args.imsize, args.imsize])
    bboxes = torch.stack([xywh2xyxy(box) for box in bboxes])
    fig, axes = plt.subplots(6, 5, figsize=(15, 18))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i != len(bboxes):
            ax.imshow(image_np)
            ax.axis('off')
            box = bboxes[i]
            x_min, y_min, x_max, y_max = map(int, box)
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red', lw=1))
            ax.set_title(anatomy_names[i])

        else:
            ax.axis('off')
        
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    pass

import torch
import numpy as np
import torch.nn.functional as F
import math

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size
from torch.autograd import Variable

def loss_bbox(pred_boxes, target):
    """
    Compute the combined L1 and GIoU loss for bounding boxes.

    Parameters:
    pred_boxes: tensor of shape [batch_size, 29, 4]
    target: tensor of shape [batch_size, 29, 4] in the format xc, yc, w, h

    Return:
    Combined loss_bbox and loss_giou
    """
    if len(pred_boxes.shape) == 2:
        pred_boxes = pred_boxes.unsqueeze(0)
    if len(target.shape) == 2:
        target = target.unsqueeze(0)
    pred_boxes_xyxy = xywh2xyxy(pred_boxes)
    target_boxes_xyxy = xywh2xyxy(target)
    # Compute L1 loss
    loss_bbox = F.l1_loss(pred_boxes_xyxy, target_boxes_xyxy, reduction='none').sum(-1).mean()
    # Compute GIoU loss
    giou_loss_list = []
    for i in range(pred_boxes_xyxy.shape[0]):  # Iterate over batch size
        giou_loss_list.append(1 - torch.diag(generalized_box_iou(pred_boxes_xyxy[i], target_boxes_xyxy[i])).mean())
    loss_giou = torch.stack(giou_loss_list).mean()
    combined_loss = 5 * loss_bbox + 2 * loss_giou
    return combined_loss




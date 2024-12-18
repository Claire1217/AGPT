# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


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
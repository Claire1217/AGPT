import torch
import numpy as np
import torch.nn.functional as F
import math

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size
from torch.autograd import Variable
from opt_einsum import contract

def build_target(args, gt_bbox, pred, device):
    batch_size = gt_bbox.size(0)
    num_scales = len(pred)
    coord_list, bbox_list = [], []
    for scale_ii in range(num_scales):
        this_stride = 32 // (2 ** scale_ii)
        grid = args.size // this_stride
        # Convert [x1, y1, x2, y2] to [x_c, y_c, w, h]
        center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        box_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        box_h = gt_bbox[:, 3] - gt_bbox[:, 1]
        coord = torch.stack((center_x, center_y, box_w, box_h), dim=1)
        # Normalized by the image size
        coord = coord / args.size
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch_size):
        anch_ious = []
        for scale_ii in range(num_scales):
            this_stride = 32 // (2 ** scale_ii)
            grid = args.size // this_stride
            # gi = coord_list[scale_ii][ii,0].long()
            # gj = coord_list[scale_ii][ii,1].long()
            # tx = coord_list[scale_ii][ii,0] - gi.float()
            # ty = coord_list[scale_ii][ii,1] - gj.float()
            gw = coord_list[scale_ii][ii,2]
            gh = coord_list[scale_ii][ii,3]

            anchor_idxs = [x + 3*scale_ii for x in [0,1,2]]
            anchors = [args.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            ## Get shape of gt box
            # gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # import pdb
            # pdb.set_trace()

            gt_box = torch.from_numpy(np.array([0, 0, gw.cpu().numpy(), gh.cpu().numpy()])).float().unsqueeze(0)
            ## Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))

            ## Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n // 3

        best_grid = args.size//(32/(2**best_scale))
        anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
        anchors = [args.anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/best_grid), \
            x[1] / (args.anchor_imsize/best_grid)) for x in anchors]

        gi = coord_list[best_scale][ii,0].long()
        gj = coord_list[best_scale][ii,1].long()
        tx = coord_list[best_scale][ii,0] - gi.float()
        ty = coord_list[best_scale][ii,1] - gj.float()
        gw = coord_list[best_scale][ii,2]
        gh = coord_list[best_scale][ii,3]
        tw = torch.log(gw / scaled_anchors[best_n%3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n%3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n%3, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).to(device).squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = bbox_list[ii].to(device)
    return bbox_list, best_gi, best_gj, best_n_list


def yolo_loss(pred_list, target, gi, gj, best_n_list, device, w_coord=5., w_neg=1./5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    num_scale = len(pred_list)
    batch_size = pred_list[0].size(0)

    pred_bbox = torch.zeros(batch_size, 4).to(device)
    gt_bbox = torch.zeros(batch_size, 4).to(device)
    for ii in range(batch_size):
        pred_bbox[ii, 0:2] = torch.sigmoid(pred_list[best_n_list[ii]//3][ii, best_n_list[ii]%3,0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = pred_list[best_n_list[ii]//3][ii, best_n_list[ii]%3, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii]//3][ii, best_n_list[ii]%3, :4, gj[ii], gi[ii]]
    loss_x = mseloss(pred_bbox[:,0], gt_bbox[:,0])
    loss_y = mseloss(pred_bbox[:,1], gt_bbox[:,1])
    loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
    loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])

    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(num_scale):
        pred_conf_list.append(pred_list[scale_ii][:,:,4,:,:].contiguous().view(batch_size,-1))
        gt_conf_list.append(target[scale_ii][:,:,4,:,:].contiguous().view(batch_size,-1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf


def trans_vg_loss(batch_pred, batch_target):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """
    batch_size = batch_pred.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    return losses

def trans_vg_cls_loss(batch_pred, batch_target):
    """Compute the losses related to the disease prediction,
       including the CE loss.
    """
    return F.cross_entropy(batch_pred, batch_target, reduction='elementwise_mean')


# 400 * 256
# 20 * 20 * 256 (2,3)

# [100,200, 30,30] (x,y,w,h) 256*256
# [2,3,1,1] 20 

def visuPooling(visu_src, target, att_weights=None):
    """pooling the visual features according to the target bbox, taking the mean at dimension h and w. 
    """
    visu_bboxs = []
    bs = target.shape[0]
    s1 = visu_src.shape
    # 20 * 20
    width = height = math.floor(math.sqrt(visu_src.shape[0]))
    visu_src = visu_src.transpose(0, 1).view(bs, height, width, -1).contiguous()  # b, h, w, d
    att_weights_batch = []
    for i in range(bs):
        visu = visu_src[i]
        bbox = target[i]
        if att_weights is not None:
            s2 = att_weights.shape # 8, 420/450, 420/450
            att_weight = att_weights[i][(s2[1] - s1[0]):] # 取视觉特征的att部分、
            att_weight = att_weight.view(height, width, -1).contiguous()
        bbox = xywh2xyxy(bbox)
        # 还原到size of attention weight 
        bbox = [max(math.floor(bbox[0]*width), 0), max(math.floor(bbox[1]*height), 0), math.floor(bbox[2]*width), math.floor(bbox[3]*height)]
        # 防止相等
        if bbox[0] == bbox[2]:
            bbox[0] = max(0, bbox[0] - 1)
            bbox[2] = min(20, bbox[2] + 1)
        if bbox[1] == bbox[3]:
            bbox[1] = max(0, bbox[1] - 1)
            bbox[3] = min(20, bbox[3] + 1)

        # visu_bbox = visu[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        # visu_bbox: visual features within bbox
        visu_bbox = visu[bbox[1]:bbox[3], bbox[0]:bbox[2], :]  # bbox是 w, h 的顺序，到了特征里是 h, w的顺序，需要注意 why??
        visu_bbox = visu_bbox.mean(dim=0).mean(dim=0).unsqueeze(0) # from (b,h,w,d) to (b,d)
        visu_bboxs.append(visu_bbox)
        if att_weights is not None:
            # att_weights of shape b,h,w,d
            att_weight_bbox = att_weight[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            att_weight_bbox = att_weight_bbox.mean(dim=0).mean(dim=0).unsqueeze(0)
            att_weights_batch.append(att_weight_bbox)
    visu_pool = torch.cat(visu_bboxs, dim=0)
    if att_weights is not None:
        att_weights_batch = torch.cat(att_weights_batch, dim=0)
        return visu_pool, att_weights_batch
    return visu_pool

def textPooling(text_src, text_mask, type='mask', att_weights=None, text_data=None, lcpTriple=None):
    """pooling the text features according to the text mask or cls
    """
    bs = text_src.shape[1]
    text_pools = []
    text_src = text_src.transpose(0, 1).contiguous()
    att_text_batch = []
    att_reg_batch = []
    if type == 'marker':
        text_ids_batch = text_data.tensors
    for i in range(bs):
        pdb.set_trace()
        text = text_src[i]
        mask = text_mask[i]
        word_count = (mask==False).int().sum()
        if type == 'mask':
            text_pool = text[:word_count, :].mean(dim=0).unsqueeze(0)
            if att_weights is not None:
                att_text = att_weights[i][1:21] # 取语言特征的att部分
                att_text = att_text[:word_count, :].mean(dim=0).unsqueeze(0)
                att_text_batch.append(att_text)
        elif type == 'all':
            text_pool = text.mean(dim=0).unsqueeze(0)
        elif type == 'cls':
            text_pool = text[0].unsqueeze(0)
            if att_weights is not None:
                if lcpTriple == 'lcpTriple':
                    att_reg = att_weights[i][0:1] # 取语言特征的att部分 8x421x421
                    att_reg_batch.append(att_reg) 
                att_text = att_weights[i][1:2] # 取语言特征的att部分 8x421x421
                att_text_batch.append(att_text)
        elif type == 'marker':
            # 找下标是1008的marker Token
            text_ids = text_ids_batch[i]
            marker_idx = (text_ids == 1008).nonzero().squeeze()
            # assert len(marker_idx.shape) > 1
            #### 取marker的部分做loss ####
            # first_marker_idx = marker_idx[0]
            # text_pool = text[first_marker_idx:first_marker_idx+1]
            # if att_weights is not None:
            #     att_text = att_weights[i][first_marker_idx:first_marker_idx+1] # 取marker语言特征的att部分 8x421x421 --> 1x421
            #     att_text_batch.append(att_text)
            #### 取marker的部分做loss ####

            #### 取marker中间的部分做loss ####
            id1 = marker_idx[0]
            id2 = marker_idx[1]
            assert id2-id1>1
            text_pool = text[id1+1:id2].mean(dim=0).unsqueeze(0)
            if att_weights is not None:
                att_text = att_weights[i][id1+1:id2].mean(dim=0).unsqueeze(0) # 取marker语言特征的att部分 8x421x421 --> 1x421
                att_text_batch.append(att_text)
            #### 取marker中间的部分做loss ####
        text_pools.append(text_pool)
    text_pools = torch.cat(text_pools, dim=0)
    if att_weights is not None:
        att_text_batch = torch.cat(att_text_batch, dim=0)
        if lcpTriple == 'lcpTriple':
            att_reg_batch = torch.cat(att_reg_batch, dim=0)
            return text_pools, att_text_batch, att_reg_batch
        return text_pools, att_text_batch
    return text_pools



def trans_vg_btloss(visu_pool, text_pool, type='l1'):
    if type == 'l1':
        return F.l1_loss(visu_pool, text_pool, reduction='elementwise_mean')
    elif type == 'l2':
        return F.mse_loss(visu_pool, text_pool)
    else:
        raise ValueError('loss type not supportted ')

# Minimizing the similarity between negative image pools and the text pools. Not maximizing the similarity between text and positive image region.
# perpendicular features between neg image and text gives zero loss
# local, anchor:text, pushaway neg
def trans_vg_caloss(pos_pool, neg_pools, text_pool, temperature=0.07, mode='max'):
    text_pool = text_pool.unsqueeze(1)  #8x1x256
    pos_pool = pos_pool.unsqueeze(1)  #8x1x256
    # projection
    if 'projection' in mode:
        pass
    visu_pools = torch.cat([pos_pool, neg_pools], dim=1)
    # normalize 
    visu_pools = F.normalize(visu_pools, p=2, dim=2)  # (b, 1+neg, d), each feature (d) is a unit vector
    text_pool = F.normalize(text_pool, p=2, dim=2) # (b, 1, d), each feature (d) is a unit vector
    anchor_dot_contrast = torch.div(torch.matmul(text_pool, visu_pools.transpose(1,2)), temperature)  # 8x1x6 # (b, 1, (1+neg))
    # use -max trick
    if 'max' in mode:
        logit_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
        logit = anchor_dot_contrast - logit_max.detach()
    else:
        logit = anchor_dot_contrast
    exp_total = torch.exp(logit).sum(dim=2).squeeze()
    logit_pos = logit[:, :, 0].squeeze()
    loss = torch.mean(exp_total - logit_pos)
    return loss

# Copy and modify from https://github.com/marshuang80/gloria/blob/main/gloria/loss/gloria_loss.py
# Cross entropy within a batch, high similarity in paired whole image-text feature and low similarity in unpaired ones in the batch giving low loss. Birectional: for each image feature, the paired text feature should be similar. For each text feature, the paired image feature should be similar.
# global, batch, anchor:text/image, pulltogether pos/pushaway neg
# label: 01234567
def trans_vg_caloss_crossbatch(cnn_code, _, rnn_code, eps=1e-8, temp=0.1):
    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0) #1,8,256
        rnn_code = rnn_code.unsqueeze(0) #1,8,256

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True) # (1, batch, 256)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True) # (1, batch, 256)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2)) # scores: (batch_i, batch_t)
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2)) 
    scores0 = scores0 / norm0.clamp(min=eps) / temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    scores1 = scores0.transpose(0, 1)
    loss0 = torch.nn.CrossEntropyLoss()(scores0, labels) # scores(batch_i, batch_t) labels: 01234567
    loss1 = torch.nn.CrossEntropyLoss()(scores1, labels) # scores(batch_t, batch_i) labels: 01234567
    loss = loss0 + loss1
    return loss


# Copy and modify from https://github.com/marshuang80/gloria/blob/main/gloria/loss/gloria_loss.py
# Contrastive between textural feature and regional image features, high similarity bwteen textural features and positive regional image feataures and low similarity between textural features and negative regional image features giving low loss
# Label: 00000000
# local, anchor: text, pulltogether pos & pushaway neg
# No cross-batch computation, 
def trans_vg_caloss_inimage(pos_pool, neg_pools, rnn_code, eps=1e-8, temp3=0.1):
    rnn_code = rnn_code.unsqueeze(1)  #8x1x256
    pos_pool = pos_pool.unsqueeze(1)  #8x1x256
    cnn_code = torch.cat([pos_pool, neg_pools], dim=1) #8，5，256
    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor([0]*batch_size)).to(cnn_code.device) # 8

    # if cnn_code.dim() == 2:
    #     cnn_code = cnn_code.unsqueeze(0)
    #     rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2)) #（8，5，256）* （8，256，1） -> (8,5,1)
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))  
    scores0 = scores0 / norm0.clamp(min=eps) / temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze() # (8,5) 
    loss = torch.nn.CrossEntropyLoss()(scores0, labels) # (8,5) batch_t, pos+4neg labels: (0,0,0,0,0,0,0,0)
    return loss


# ref to https://github.com/wzhouad/ATLOP/blob/main/model.py
# h_att: attention text, t_att: attention positive, g_att: attention reg
# emb of shape seq_len, batch_size, 256
def cal_lcp_triple(h_att, t_att, g_att, emb):
    bs = h_att.shape[0]
    # Element-wise multiplication
    ht_att = h_att * t_att * g_att
    ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
    rss = []
    # for each element in batch, take the embedding tensor of shape (seq_len, 256), mupltiply by the attention weights of shape (1, 256),resulting in tensor of shape (1,256) for rs. Then rss of shape (batch, 256)
    for i in range(bs):
        rs = contract("ld,rl->rd", emb[:, i, :], ht_att[i:i+1, :])
        rss.append(rs)
    rss = torch.cat(rss, dim=0)
    return rss

# copy and modify from https://github.com/marshuang80/gloria/blob/main/gloria/loss/gloria_loss.py
# element-wise addition! 
# textural embedding and regional visual embeddings(both pos and neg) are added with attention-based global embedding output['vg_hs'] 
# attention-baesd embedding is weighted of 
# pulling pos and text together, pulling neg and text further away
def trans_vg_caloss_inimage_lcp_triple(pos_pool, neg_pools, rnn_code, att_pos, att_negs, att_text, att_reg, emb, eps=1e-8, temp3=0.1):
    rnn_code = rnn_code.unsqueeze(1)  #8x1x256
    pos_pool = pos_pool.unsqueeze(1)  #8x1x256
    cnn_code = torch.cat([pos_pool, neg_pools], dim=1) # 8,5,256
    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor([0]*batch_size)).to(cnn_code.device) # 8 [0,0,0,0,0,0,0,0]
    # lcp: 通过 att_pos, att_negs, att_text 重新计算新的embedding 
    # att_pos, att_negs, att_text of shape (8,256), embedding of shape(8, 256)
    tp = cal_lcp_triple(att_text, att_pos, att_reg, emb) # seq_len, 8, 256
    tns = []
    neg_num = neg_pools.shape[1]
    for j in range(neg_num):
        tn = cal_lcp_triple(att_text, att_negs[:, j, :], att_reg, emb)
        tns.append(tn.unsqueeze(1))
    
    # compute the attension-weighted positive feature and the attention-weighted negative feature, then concat them
    c = torch.cat([tp.unsqueeze(1)] + tns, dim=1) # shape: batch_size, 1+ num_neg, 256, calculated attented features
    
    # 把 c 加到原本的 emb_pool 上去
    # repete text features for (1+neg_num) times
    rnn_code = rnn_code.repeat(1, neg_num+1, 1) # shape: batch_size, 1+ num_neg, 256
    # combining text features with attention-based features
    rnn_code = rnn_code + c
    # combining image features with attention based features
    cnn_code = cnn_code + c
    
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    
    # element wise multiplication between image feature and text feature
    scores0 = cnn_code * rnn_code # 8, 1+neg, 256
    norm0 = cnn_code_norm * rnn_code_norm
    # scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    # norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) / temp3

    # 8x6x256 --> 8x6x1
    
    scores0 = scores0.sum(2)
    
    loss = torch.nn.CrossEntropyLoss()(scores0, labels)
    return loss

# ref to https://github.com/wzhouad/ATLOP/blob/main/model.py
def cal_lcp(h_att, t_att, emb):
    bs = h_att.shape[0]
    ht_att = h_att * t_att
    ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
    rss = []
    for i in range(bs):
        rs = contract("ld,rl->rd", emb[:, i, :], ht_att[i:i+1, :])
        rss.append(rs)
    rss = torch.cat(rss, dim=0)
    return rss

# copy and modify from https://github.com/marshuang80/gloria/blob/main/gloria/loss/gloria_loss.py
# not using the attention of reg token in this case. 
def trans_vg_caloss_inimage_lcp(pos_pool, neg_pools, rnn_code, att_pos, att_negs, att_text, emb, ws=None, wo=None, wc1=None, wc2=None, eps=1e-8, temp3=0.1):
    rnn_code = rnn_code.unsqueeze(1)  #8x1x256
    pos_pool = pos_pool.unsqueeze(1)  #8x1x256
    cnn_code = torch.cat([pos_pool, neg_pools], dim=1)
    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor([0]*batch_size)).to(cnn_code.device) # 8
    # lcp: 通过 att_pos, att_negs, att_text 重新计算新的embedding
    tp = cal_lcp(att_text, att_pos, emb)
    tns = []
    neg_num = neg_pools.shape[1]
    for j in range(neg_num):
        tn = cal_lcp(att_text, att_negs[:, j, :], emb)
        tns.append(tn.unsqueeze(1))
    c = torch.cat([tp.unsqueeze(1)] + tns, dim=1)
    
    if wc1 is None:
        # 把 c 加到原本的 emb_pool 上去
        rnn_code = rnn_code.repeat(1, neg_num+1, 1)
        rnn_code = rnn_code + c
        cnn_code = cnn_code + c
    else:  # Do projection for text/image embeddings
        # 先projection，再把 c 加到原本的 emb_pool 上去
        rnn_code = rnn_code.repeat(1, neg_num+1, 1)
        rnn_code = rnn_code + wc1(c)
        cnn_code = cnn_code + wc1(c)
    
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = cnn_code * rnn_code
    norm0 = cnn_code_norm * rnn_code_norm
    # scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    # norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) / temp3

    # 8x6x256 --> 8x6x1
    scores0 = scores0.sum(2)
    loss = torch.nn.CrossEntropyLoss()(scores0, labels)
    return loss

# 仿照原本的box loss改写
def trans_vg_conBox(batch_pred, batch_target):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """
    batch_size = batch_pred.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))
    return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes


def CAlossFunc(epoch, max_epoch, type='poly'):
    if type == 'poly':
        power = 0.9
        return (epoch/max_epoch)**power

def trans_vg_gn_loss(batch_pred, batch_target):
    """
       including the Multi-BCE loss.
    """
    return F.binary_cross_entropy_with_logits(batch_pred, batch_target)



def inbatch_bceloss(img_pool, txt_pool, phrase_exist=None, eps = 1e-8, temp=0.1):
    '''
    Img_pool of shape [batch, 29, 256], txt_pool of shape [batch, 29, 256], phrase_exist of shape [batch, 29]
    Return: loss
    '''
    # get the features of img_pool and txt_pool where phrases exist. 
    batch_size = img_pool.shape[0]
    total_loss = 0
    for i in range(batch_size):
        phrase_mask = phrase_exist[i]
        pair_count = phrase_mask.sum().item()
        if pair_count == 0:
            continue
        labels = Variable(torch.LongTensor(range(pair_count))).to(img_pool.device)
        img_pool_selected = img_pool[i][phrase_mask].unsqueeze(0)
        txt_pool_selected = txt_pool[i][phrase_mask].unsqueeze(0)
        # Computer similarity scores
        score = torch.bmm(img_pool_selected, txt_pool_selected.transpose(1,2))
        # Compute norms
        img_norm = torch.norm(img_pool_selected, 2, dim=2, keepdim=True)
        txt_norm = torch.norm(txt_pool_selected, 2, dim=2, keepdim=True)
        score = torch.bmm(img_pool_selected, txt_pool_selected.transpose(1,2))
        norm = torch.bmm(img_norm, txt_norm.transpose(1,2)).clamp(min=eps)
        score = score / norm / temp
        print(score.shape, labels.shape)
        loss0 = torch.nn.CrossEntropyLoss()(score.squeeze(), labels)
        loss1 = torch.nn.CrossEntropyLoss()(score.squeeze().transpose(0,1), labels)
        loss = (loss0 + loss1) / 2
        total_loss += loss
    return total_loss/batch_size

     

def crossbatch_bceloss(img_pool, txt_pool, phrase_exist, eps=1e-8, temp=0.1):
    '''
    Img_pool of shape [batch, 29, 256], txt_pool of shape [batch, 29, 256], phrase_exist of shape [batch, 29]
    Return: loss
    '''
    batch_size, num_phrases, feature_dim = img_pool.shape
    total_loss = 0

    # Flatten and mask tensors
    img_pool_flat = img_pool.view(batch_size * num_phrases, feature_dim)
    txt_pool_flat = txt_pool.view(batch_size * num_phrases, feature_dim)
    phrase_exist_flat = phrase_exist.view(-1)

    # Filter based on phrase_exist mask
    img_pool_valid = img_pool_flat[phrase_exist_flat]
    txt_pool_valid = txt_pool_flat[phrase_exist_flat]

    # Compute norms
    img_norm = torch.norm(img_pool_valid, p=2, dim=1, keepdim=True)
    txt_norm = torch.norm(txt_pool_valid, p=2, dim=1, keepdim=True)
    norm = (img_norm * txt_norm.transpose(0, 1)).clamp(min=eps)

    # Compute similarity scores
    scores = torch.mm(img_pool_valid, txt_pool_valid.transpose(0, 1)) / norm / temp
    
    # Compute cross-entropy loss
    pair_count = img_pool_valid.size(0) # total amount of region-text pairs in the batch: e.g.8+8+9+12 = 
    labels = Variable(torch.LongTensor(range(pair_count))).to(img_pool.device)
    print(scores)
    loss0 = torch.nn.CrossEntropyLoss()(scores, labels)
    loss1 = torch.nn.CrossEntropyLoss()(scores.transpose(0, 1), labels)
    total_loss = (loss0 + loss1) / 2

    return total_loss



def crossbatch_positive_bcewithlogitloss(img_pool, txt_pool, disease_id, normality, phrase_exist, eps=1e-8, temp=0.1):
    """
    Computes the cross-batch loss with additional positive pairs for matching disease_id and normality.
    
    Parameters:
    img_pool (torch.Tensor): Tensor of shape [batch, 29, 256] representing image features.
    txt_pool (torch.Tensor): Tensor of shape [batch, 29, 256] representing text features.
    disease_id (torch.Tensor): Tensor of shape [batch, 29] representing disease IDs.
    normality (torch.Tensor): Tensor of shape [batch, 29] representing normality status.
    phrase_exist (torch.Tensor): Tensor of shape [batch, 29] indicating the presence of phrases.
    eps (float): Small constant to avoid division by zero.
    temp (float): Temperature scaling factor for similarity scores.
    
    Returns:
    torch.Tensor: The computed cross-batch extra positive loss.
    """
    
    # Validate input shapes
    assert img_pool.shape == txt_pool.shape, "img_pool and txt_pool must have the same shape."
    assert disease_id.shape == normality.shape == phrase_exist.shape, "disease_id, normality, and phrase_exist must have the same shape."
    
    batch_size, num_phrases, feature_dim = img_pool.shape

    # Flatten and mask tensors
    img_pool_flat = img_pool.view(batch_size * num_phrases, feature_dim)
    txt_pool_flat = txt_pool.view(batch_size * num_phrases, feature_dim)
    phrase_exist_flat = phrase_exist.view(-1)

    # Filter based on phrase_exist mask
    img_pool_valid = img_pool_flat[phrase_exist_flat]
    txt_pool_valid = txt_pool_flat[phrase_exist_flat]
    disease_id_valid = disease_id.view(-1)[phrase_exist_flat]
    normality_valid = normality.view(-1)[phrase_exist_flat]
    

    # Compute norms
    img_norm = torch.norm(img_pool_valid, p=2, dim=1, keepdim=True)
    txt_norm = torch.norm(txt_pool_valid, p=2, dim=1, keepdim=True)
    norm = (img_norm * txt_norm.transpose(0, 1)).clamp(min=eps)

    # Compute similarity scores
    scores = torch.mm(img_pool_valid, txt_pool_valid.transpose(0, 1)) / norm / temp

    pair_count = img_pool_valid.size(0)

    # Create binary labels matrix
    
    labels_matrix = torch.eye(pair_count).to(img_pool.device)

    # Mark additional positive pairs using broadcasting
    
    
    disease_id_matrix = disease_id_valid.unsqueeze(0) == disease_id_valid.unsqueeze(1)
    normality_matrix = normality_valid.unsqueeze(0) == normality_valid.unsqueeze(1)
    extra_positive_mask = disease_id_matrix & normality_matrix & (~torch.eye(pair_count, device=img_pool.device).bool())
    labels_matrix = labels_matrix + extra_positive_mask.float()

    # Compute BCEWithLogitsLoss
    bce_loss = torch.nn.BCEWithLogitsLoss()

    loss0 = bce_loss(scores, labels_matrix)
    loss1 = bce_loss(scores.transpose(0, 1), labels_matrix)
    total_loss = (loss0 + loss1) / 2

    return total_loss



def loss_bbox(pred_boxes, target):
    """
    Compute the combined L1 and GIoU loss for bounding boxes.

    Parameters:
    pred_boxes: tensor of shape [batch_size, 29, 4] pr [batch_size, 4]
    target: tensor of shape [batch_size, 29, 4] or [batch_size, 4] in the format xc, yc, w, h

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
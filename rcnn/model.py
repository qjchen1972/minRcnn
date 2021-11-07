import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops import misc, roi_align, MultiScaleRoIAlign
from .util import *
from adabelief_pytorch import AdaBelief
from collections import OrderedDict
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

logger = logging.getLogger(__name__)

class RcnnConfig:

    backbone_name = "resnet50"
    backbone_outchannels = 256
    backbone_pretrained = True
    
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    anchor_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    #anchor_sizes = ((128, 256, 512),) 
    #anchor_ratios = ((0.5, 1.0, 2.0),) 
    
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_num_samples = 256
    rpn_positive_fraction = 0.5
    rpn_reg_weights = (1., 1., 1., 1.)
    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    rpn_nms_thresh = 0.7
    
    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_num_samples = 512
    box_positive_fraction = 0.25
    box_reg_weights = (10., 10., 5., 5.)
    box_score_thresh = 0.7
    box_nms_thresh = 0.3
    box_num_detections = 100
    havemask = True
    
    num_classes = 2
    min_size = 1    
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
class ResBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50', out_channels=256, pretrained=True):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        self.body =nn.ModuleDict()
        for i, d in enumerate(body.named_children()):
            
            if i < 8: self.body.update({d})
            elif i == 8: 
                self.avgpool = d[-1]
            elif i == 9: self.classier = d[-1]
            else: pass

        
        returned_layers = [1, 2, 3, 4]
        self.return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
        in_channels_stage2 = body.inplanes // 8
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
    
        out = OrderedDict()
        for name, module in  self.body.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        x = self.fpn(out)                
        return x
        
        
class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.score = nn.Conv2d(in_channels, num_anchors, 1)
        self.loc = nn.Conv2d(in_channels, 4 * num_anchors, 1)
        
        for l in self.children():
            nn.init.normal_(l.weight, mean=0, std=0.01)
            nn.init.constant_(l.bias, 0)
            
    def forward(self, x):
        scores = []
        locs = []
        for feature in x:
            feature = F.relu(self.conv(feature))
            scores.append(self.score(feature))
            locs.append(self.loc(feature))
        return scores, locs

    

class RegionProposalNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.anchor_generator = AnchorGenerator(config.anchor_sizes, config.anchor_ratios)
        num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        self.head = RPNHead(config.backbone_outchannels, num_anchors)  
        self.proposal_matcher = Matcher(config.rpn_fg_iou_thresh, config.rpn_bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(config.rpn_num_samples, config.rpn_positive_fraction)
        self.box_coder = BoxCoder(config.rpn_reg_weights)
        
        
    def create_proposal(self, anchor, objectness, pred_bbox_delta, image_shapes):
        if self.training:
            pre_nms_top_n = self.config.rpn_pre_nms_top_n['training']
            post_nms_top_n = self.config.rpn_post_nms_top_n['training']
        else:
            pre_nms_top_n = self.config.rpn_pre_nms_top_n['testing']
            post_nms_top_n = self.config.rpn_post_nms_top_n['testing']
            
        rois = []
        for score, loc, image_shape in zip(objectness, pred_bbox_delta, image_shapes):
            
            pre_nms_top_n = min(score.shape[0], pre_nms_top_n)
            top_n_idx = score.topk(pre_nms_top_n)[1]
            score = score[top_n_idx]
            proposal = self.box_coder.decode(loc[top_n_idx], anchor[top_n_idx])
            proposal, score = process_box(proposal, score, image_shape, self.config.min_size)
            
            keep = torch.ops.torchvision.nms(proposal, score, self.config.rpn_nms_thresh)[:post_nms_top_n] 
            proposal = proposal[keep]
            rois.append(proposal)
            
        return rois
    
    def compute_loss(self, objectness, pred_bbox_delta, target, anchor):
    
        objectness_loss = torch.zeros(1).to(objectness)
        box_loss = torch.zeros(1).to(objectness)
       
        for score, loc, per_target in zip(objectness, pred_bbox_delta, target):
                        
            box = per_target['boxes']
            iou = box_iou(box, anchor) 
            label, matched_idx = self.proposal_matcher(iou)
            pos_idx, neg_idx = self.fg_bg_sampler(label)
            idx = torch.cat((pos_idx, neg_idx))
            regression_target = self.box_coder.encode(box[matched_idx[pos_idx]], anchor[pos_idx])
            
            objectness_loss += F.binary_cross_entropy_with_logits(score[idx], label[idx])
            box_loss += F.l1_loss(loc[pos_idx], regression_target, reduction='sum') / idx.numel()
            
        return objectness_loss, box_loss
        
    def forward(self, features, image_shape, modify_image_sizes, target=None):
    
        features = list(features.values())
        
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(features, image_shape)
        objectness = [t.permute(0, 2, 3, 1).reshape(t.shape[0], -1) for t in objectness]
        pred_bbox_deltas = [t.permute(0, 2, 3, 1).reshape(t.shape[0], -1, 4) for t in pred_bbox_deltas]
        
        objectness = torch.cat(objectness, dim=1)
        pred_bbox_deltas = torch.cat(pred_bbox_deltas, dim=1)
        proposal = self.create_proposal(anchors, objectness.detach(), pred_bbox_deltas.detach(), modify_image_sizes)
        
        if self.training:
            assert target is not None, "Cannot forward, gt_box is None."
            objectness_loss, box_loss = self.compute_loss(objectness, pred_bbox_deltas, target, anchors)
            return proposal, dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)
        
        return proposal, {}
        

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta        
    
    
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                

                
class RoIHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config        
        
        self.box_predictor = FastRCNNPredictor(config.backbone_outchannels * 7 *7, 1024, config.num_classes)      
        self.mask_predictor =  MaskRCNNPredictor(config.backbone_outchannels, (256, 256, 256, 256), 256, config.num_classes)
        self.proposal_matcher = Matcher(config.box_fg_iou_thresh, config.box_bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(config.box_num_samples, config.box_positive_fraction)
        self.box_coder = BoxCoder(config.box_reg_weights)        
        self.box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        self.mask_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],output_size=14,sampling_ratio=2)
        
    def fastrcnn_loss(self, class_logit, box_regression, label, regression_target):
        
        label = torch.cat(label, dim=0)
        classifier_loss = F.cross_entropy(class_logit, label)
        box_regression = box_regression[label > 0]
        if box_regression.shape[0] == 0:
            box_reg_loss = torch.tensor(0).to(box_regression)
            return classifier_loss, box_reg_loss
        
        label = label[label>0]
        regression_target =  torch.cat(regression_target, dim=0)
        box_regression = box_regression.reshape(box_regression.shape[0], -1, 4)
        box_idx = torch.arange(box_regression.shape[0], device=label.device)
        box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / class_logit.shape[0]

        return classifier_loss, box_reg_loss
    
    def maskrcnn_loss(self, mask_logit, proposal, matched_idx, label, target):
        
        M = mask_logit.shape[-1]
        mask_target = []
        for per_target, box, idx in zip(target, proposal, matched_idx):            
            idx = idx.to(box)
            rois = torch.cat([idx[:, None], box], dim=1)
            gt_masks = per_target['masks'].to(rois)
            gt_masks = gt_masks.reshape(-1, 1, gt_masks.shape[-2], gt_masks.shape[-1])
            mask_target.append(roi_align(gt_masks, rois, output_size=(M,M), spatial_scale=1., sampling_ratio=-1)[:,0])
        
        mask_target = torch.cat(mask_target, dim=0)

        if mask_target.numel() == 0:
            return mask_logit.sum() * 0
            
        label = torch.cat(label, dim=0)
        idx = torch.arange(label.shape[0], device=label.device)
        mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)

        return mask_loss

    def select_training_samples(self, Rois, target):
    
        proposals = []
        matched_idxs = []
        labels = []
        regression_targets = []
        
        for proposal, per_target in zip(Rois, target):
            gt_box = per_target['boxes']
            gt_label = per_target['labels']
            
            proposal = torch.cat((proposal, gt_box))
            iou = box_iou(gt_box, proposal)
            pos_neg_label, matched_idx = self.proposal_matcher(iou)
            pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
            idx = torch.cat((pos_idx, neg_idx))
        
            regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
            proposal = proposal[idx]
            matched_idx = matched_idx[idx]
            label = gt_label[matched_idx] + 1
            num_pos = pos_idx.shape[0]
            label[num_pos:] = 0
            proposals.append(proposal)
            matched_idxs.append(matched_idx)
            labels.append(label)
            regression_targets.append(regression_target)
            
        return proposals, matched_idxs, labels, regression_targets
    
    def fastrcnn_inference(self, class_logit, box_regression, proposals, image_shapes):
        N, num_classes = class_logit.shape
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)
        startpos = 0
        endpos = 0
        results = []
        for proposal, image_shape in zip(proposals, image_shapes):
            
            boxes = []
            labels = []
            scores = []
            startpos = endpos
            endpos += proposal.shape[0]
            for l in range(1, num_classes):
                score, box_delta = pred_score[startpos : endpos, l], box_regression[startpos : endpos, l]
                keep = score >= self.config.box_score_thresh
                box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
                box = self.box_coder.decode(box_delta, box)
                box, score = process_box(box, score, image_shape, self.config.min_size)
                keep = torch.ops.torchvision.nms(box, score, self.config.box_nms_thresh)[:self.config.box_num_detections]
                box, score = box[keep], score[keep]
                label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)
            
                boxes.append(box)
                labels.append(label)
                scores.append(score)
        
            results.append(dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores)))        
        return results
    
    def forward(self, features, proposal, modify_image_sizes, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)
           
        box_feature = self.box_roi_pool(features, proposal, modify_image_sizes)
        
        class_logit, box_regression = self.box_predictor(box_feature)
        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = self.fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, modify_image_sizes)
        
        if self.config.havemask:
            if self.training:
            
                mask_proposal = [] 
                pos_matched_idx = [] 
                mask_label = [] 
                mask_regression_target = [] 
                for one_proposal, one_pos_matched_idx, one_label, one_regression_target\
                                   in zip(proposal, matched_idx, label, regression_target):
                    num_pos = one_regression_target.shape[0]
                    mask_proposal.append(one_proposal[:num_pos])
                    pos_matched_idx.append(one_pos_matched_idx[:num_pos])
                    mask_label.append(one_label[:num_pos])
                    mask_regression_target.append(one_regression_target)
                '''
                mask_proposal_num = sum([t.shape[0] for t in mask_proposal])
                if mask_proposal_num == 0:
                    losses.update(dict(roi_mask_loss=torch.tensor(0).to(target[0]['boxes'])))
                    return result, losses
                '''
            else:
                mask_proposal = [r["boxes"] for r in result]
                '''
                mask_proposal_num = sum([t.numel() for t in mask_proposal])
                if mask_proposal_num == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28)).to(target[0]['boxes'])))
                    return result, losses
                '''
                
            mask_feature = self.mask_roi_pool(features, mask_proposal, modify_image_sizes)
            mask_logit = self.mask_predictor(mask_feature)
            if self.training:
                mask_loss = self.maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, target)
                losses.update(dict(roi_mask_loss=mask_loss))
            else:
                labels = [r["labels"] for r in result]
                boxes_per_image = [label.shape[0] for label in labels]
                labels = torch.cat(labels)
                idx = torch.arange(labels.shape[0], device=labels.device)
                mask_logit = mask_logit[idx, labels]
                mask_probs = mask_logit.sigmoid() #[:, None]
                mask_probs = mask_probs.split(boxes_per_image, dim=0)
                for mask_prob, r in zip(mask_probs, result):
                    r["masks"] = mask_prob
                
        return result, losses
        
class Rcnn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone = ResBackbone(config.backbone_name, config.backbone_outchannels, config.backbone_pretrained)
        self.rpn = RegionProposalNetwork(config)
        self.head = RoIHeads(config)
        
        logger.info("number of parameters: %e( %e )", sum(p.numel() for p in self.parameters()),
                                                      sum(p.numel() for p in self.parameters() if p.requires_grad))
        self.transformer = Transformer(min_size=512, max_size=512,
                               image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    def config_optimizers(self, train_config):

        lr = train_config.learning_rate
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': train_config.weight_decay}]
        #optimizer = torch.optim.Adam(params)
        #optimizer = AdaBelief(self.parameters(), lr=lr, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
        #optimizer = AdaBelief(params, eps=1e-8, betas=(0.9,0.999), weight_decouple = True, rectify = True)
        #optimizer = torch.optim.AdamW(params, betas=train_config.betas)
        optimizer = torch.optim.SGD(params, momentum=0.9)
        return optimizer
        
    def forward(self, images, target=None):
    
        original_image_sizes = [img.shape[-2:] for img in images]
        image, modify_image_sizes, target = self.transformer(images, target)
        image_shape = image.shape[-2:]
        features = self.backbone(image)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        
        proposals, rpn_losses = self.rpn(features, image_shape, modify_image_sizes, target)
        result, roi_losses = self.head(features, proposals, modify_image_sizes, target)
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            return self.transformer.postprocess(result, modify_image_sizes, original_image_sizes)
    
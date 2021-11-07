import math
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_box(box, score, image_shape, min_size):
    """
    Clip boxes in the image size and remove boxes which are too small.
    """
    
    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, image_shape[1]) 
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, image_shape[0]) 

    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    keep = torch.where((w >= min_size) & (h >= min_size))[0]
    box, score = box[keep], score[keep]
    return box, score
    
def box_iou(box_a, box_b):
    """
    Arguments:
        boxe_a (Tensor[N, 4])
        boxe_b (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in box_a and box_b
    """
    
    if box_a.shape[1] != 4 or box_b.shape[1] != 4:
        raise IndexError
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], 1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], 1)
    
    return inter / torch.clamp(area_a[:, None] + area_b - inter, min=torch.finfo(box_a.dtype).eps)

class BoxCoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_box, proposal):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor[N, 4]): reference boxes
            proposals (Tensor[N, 4]): boxes to be encoded
        """
        
        width = torch.clamp(proposal[:, 2] - proposal[:, 0], min=torch.finfo(proposal.dtype).eps) 
        height = torch.clamp(proposal[:, 3] - proposal[:, 1], min=torch.finfo(proposal.dtype).eps)
        ctr_x = proposal[:, 0] + 0.5 * width
        ctr_y = proposal[:, 1] + 0.5 * height

        gt_width = reference_box[:, 2] - reference_box[:, 0]
        gt_height = reference_box[:, 3] - reference_box[:, 1]
        gt_ctr_x = reference_box[:, 0] + 0.5 * gt_width
        gt_ctr_y = reference_box[:, 1] + 0.5 * gt_height

        dx = self.weights[0] * (gt_ctr_x - ctr_x) / width
        dy = self.weights[1] * (gt_ctr_y - ctr_y) / height
        dw = self.weights[2] * torch.log( torch.clamp(gt_width / width, min=torch.finfo(proposal.dtype).eps)) 
        dh = self.weights[3] * torch.log(torch.clamp(gt_height / height, min=torch.finfo(proposal.dtype).eps))

        delta = torch.stack((dx, dy, dw, dh), dim=1)
        return delta

    def decode(self, delta, box):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            delta (Tensor[N, 4]): encoded boxes.
            boxes (Tensor[N, 4]): reference boxes.
        """
        
        dx = delta[:, 0] / self.weights[0]
        dy = delta[:, 1] / self.weights[1]
        dw = delta[:, 2] / self.weights[2]
        dh = delta[:, 3] / self.weights[3]

        
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        ctr_x = box[:, 0] + 0.5 * width
        ctr_y = box[:, 1] + 0.5 * height

        pred_ctr_x = dx * width + ctr_x
        pred_ctr_y = dy * height + ctr_y
        pred_w = torch.exp(dw) * width
        pred_h = torch.exp(dh) * height

        xmin = pred_ctr_x - 0.5 * pred_w
        ymin = pred_ctr_y - 0.5 * pred_h
        xmax = pred_ctr_x + 0.5 * pred_w
        ymax = pred_ctr_y + 0.5 * pred_h

        target = torch.stack((xmin, ymin, xmax, ymax), dim=1)
        return target
        
class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
        
        value, matched_idx = iou.max(dim=0)
        label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device) 
        label[value >= self.high_threshold] = 1
        label[value < self.low_threshold] = 0
        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0]
            highest_quality = torch.clamp(highest_quality, min=torch.finfo(highest_quality.dtype).eps)
            _, gt_pred_pairs = torch.where(iou == highest_quality[:, None])
            label[gt_pred_pairs] = 1
       
        return label, matched_idx
    

class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx
    
class Transformer(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std, stride=32):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.stride = stride
        
    def forward(self, images, targets=None):
        
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
        
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batched_image(images, stride=self.stride)
        
        return images, image_sizes, targets        
        
    def normalize(self, image):
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        dtype, device = image.dtype, image.device
        mean = torch.tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        ori_image_shape = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        
        scale_factor = min(self.min_size / min_size, self.max_size / max_size)
        size = [round(s * scale_factor) for s in ori_image_shape]
        image = F.interpolate(image[None], size=size, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target
        
        box = target['boxes']
        box[:, [0, 2]] = box[:, [0, 2]] * image.shape[-1] / ori_image_shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] * image.shape[-2] / ori_image_shape[0]
        target['boxes'] = box
        
        if 'masks' in target:
            mask = target['masks']
            mask = torchvision.transforms.Resize(image.shape[-2:])(mask[None])
            mask[mask>0] = 1
            target['masks'] = mask
            
        return image, target
      
    def batched_image(self, images, stride):
        maxC = 0
        maxH = 0
        maxW = 0
        for img in images:            
            if maxH < img.shape[-2]: maxH = img.shape[-2]
            if maxW < img.shape[-1]: maxW = img.shape[-1]
            if maxC < img.shape[-3]: maxC = img.shape[-3]
            
        maxH = int(math.ceil(float(maxH) / stride) * stride)
        maxW = int(math.ceil(float(maxW) / stride) * stride)
        batch_shape = [len(images)] + [maxC, maxH, maxW]
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        return batched_imgs

    def _expand_detection(self, mask, box, padding):
        M = mask.shape[-1]
        scale = (M + 2 * padding) / M
        padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)

        w_half = (box[:, 2] - box[:, 0]) * 0.5
        h_half = (box[:, 3] - box[:, 1]) * 0.5
        x_c = (box[:, 2] + box[:, 0]) * 0.5
        y_c = (box[:, 3] + box[:, 1]) * 0.5

        w_half = w_half * scale
        h_half = h_half * scale

        box_exp = torch.zeros_like(box)
        box_exp[:, 0] = x_c - w_half
        box_exp[:, 2] = x_c + w_half
        box_exp[:, 1] = y_c - h_half
        box_exp[:, 3] = y_c + h_half
        return padded_mask, box_exp.to(torch.int64)

    def _paste_masks_in_image(self, mask, box, padding, image_shape):
        mask, box = self._expand_detection(mask, box, padding)
        N = mask.shape[0]
        size = (N,) + tuple(image_shape)
        im_mask = torch.zeros(size, dtype=mask.dtype, device=mask.device)

        for m, b, im in zip(mask, box, im_mask):
            b = b.tolist()
            w = max(b[2] - b[0], 1)
            h = max(b[3] - b[1], 1)

            m = F.interpolate(m[None, None], size=(h, w), mode='bilinear', align_corners=False)[0][0]

            x1 = max(b[0], 0)
            y1 = max(b[1], 0)
            x2 = min(b[2], image_shape[1])
            y2 = min(b[3], image_shape[0])

            im[y1:y2, x1:x2] = m[(y1 - b[1]):(y2 - b[1]), (x1 - b[0]):(x2 - b[0])]
        return im_mask

    def postprocess(self, result, image_shapes, original_image_sizes):

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            box = pred["boxes"]
            box[:, [0, 2]] = box[:, [0, 2]] * o_im_s[1] / im_s[1]
            box[:, [1, 3]] = box[:, [1, 3]] * o_im_s[0] / im_s[0]
            result[i]["boxes"] = box
            if "masks" in pred:
                masks = pred["masks"]
                masks = self._paste_masks_in_image(masks, box, 1, o_im_s)
                result[i]["masks"] = masks
        return result



class AnchorGenerator:

    def __init__(self, sizes=((128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),),):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [self.generate_anchors(size, aspect_ratio)
                             for size, aspect_ratio in zip(sizes, aspect_ratios)]

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
    
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device)
                             for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        cell_anchors = self.cell_anchors

        for size, stride, base_anchors in zip( grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            ) * stride_width
            shifts_y = torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )
        return anchors

    def __call__(self, feature_maps, image_size):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
        anchors = torch.cat(anchors_in_image)        
        return anchors




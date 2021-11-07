#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torchvision
import torch
import os
import torch.nn.functional as F
import math
import cv2

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        #filename= 'dat.log',
)

# make deterministic
from rcnn.util import set_seed, Transformer
#set_seed(42)


def mask2point(mask):
    img = (mask > 0.5).cpu().numpy()
    img = cv2.copyMakeBorder(img.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(img, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE, offset=(-1,-1))
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    area = np.array([cv2.contourArea(i) for i in polygons])
    return polygons[area.argmax()] if area.shape[0]>0  else None
    
def showimg(image, color, target=None, classes=None):
    
    for i, b in enumerate(target["boxes"]):
        if 'scores' in target:
            txt = "{} {:.3f}".format(classes[target["labels"][i].item()], target["scores"][i].item())
        else:
            txt = "{}".format(classes[target["labels"][i].item()])
        image = cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), (color, color, color), 2)
        image = cv2.putText(image, txt, (b[0], b[1]), cv2.FONT_HERSHEY_COMPLEX, 1., (color, color, color), 2)
        point = mask2point(target["masks"][i])
        if point is not None:
            cv2.polylines(image, [point], isClosed=True, color=(color,color, color), thickness=2)
    
def _filetolist(dir_str, fileList):
    abspath = os.path.abspath(dir_str)
    for x in os.listdir(abspath):        
        filepath = os.path.join(abspath,x)
        if os.path.isdir(filepath):
            continue
        else:
            fileList.append(filepath)

def dirtolist(dir_str):

    abspath = os.path.abspath(dir_str)
    labelList = []
    imgList = []
    for x in os.listdir(abspath):        
        filepath = os.path.join(abspath,x)
        if os.path.isdir(filepath):
            if x == 'label': 
                _filetolist(filepath, labelList)
            else:
                _filetolist(filepath, imgList)
    labelList.sort()
    imgList.sort()
    return labelList, imgList

def img2data(img_path, label_path):

    img = Image.open(img_path).convert('RGB')
    
    boxes = []
    labels = []
    masks = np.empty(0)
    if os.path.exists(label_path):
        with open(label_path, "r") as lf:            
            jason = json.load(lf)
            shapes = jason['shapes']
            for i in range(len(shapes)):
                points = np.array(shapes[i]['points'])
                mask = np.zeros(img.size)
                cv2.drawContours(mask, [np.trunc(points).astype(int)], 0, (1,1,1), cv2.FILLED)
                mask = mask[None]
                if masks.shape[0] == 0:
                    masks = mask
                else:
                    masks = np.concatenate((masks ,mask),axis=0)
                boxes.append([np.min(points[:,0]), np.min(points[:,1]), np.max(points[:,0]), np.max(points[:,1])])
                labels.append(0)

    else:
        boxes.append([0, 0, 0, 0])
        labels.append(0)
        mask = np.zeros(img.size)[None]
        masks = mask        

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels)
    return img, dict(boxes=boxes, labels=labels, masks=torch.from_numpy(masks))
        

from torch.utils.data import Dataset
from PIL import Image
import json

class MedDataset(Dataset):
    
    def __init__(self, path ):
        self.path = path
        _, self.imgList = dirtolist(path)
    
    def __getitem__(self, idx):
        img, target = self.get_example(idx)
        img = torchvision.transforms.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

    def get_example(self, idx):
        
        imgName = self.imgList[idx]
        dirs = imgName.split("/")        
        label = self.path + '/label/' + dirs[-1][:-3] + 'json'
        img, target= img2data(imgName, label)
        
        return img, target
        
    def get_imgName(self, idx):
        imgName = self.imgList[idx]
        dirs = imgName.split("/")
        return dirs[-1]
        
    
train_dir = '/home/data/train'
valid_dir = '/home/data/validate'
train_dataset = MedDataset(train_dir)
test_dataset = MedDataset(valid_dir)


from rcnn.model import Rcnn, RcnnConfig

# we'll do something a bit smaller
mconf = RcnnConfig(backbone_name = "resnet50", havemask=True)
model = Rcnn(mconf)

from rcnn.trainer import Trainer, TrainerConfig

batch_size = 4
tokens_per_epoch = len(train_dataset) // batch_size
train_epochs = 100 # todo run a bigger model and longer, this is tiny

# initialize a trainer instance and kick off training
train_conf = TrainerConfig(max_epochs=train_epochs, batch_size=batch_size, learning_rate=1e-2,
                      betas = (0.9, 0.95), weight_decay=0.0005,
                      lr_decay=True, warmup_tokens=tokens_per_epoch*5 , final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path='test_model.pt', last_path='models/model-4.pt',
                      num_workers=0)
trainer = Trainer(model, train_dataset, test_dataset, train_conf)

if train_conf.last_path is None:
    pass
elif os.path.exists(train_conf.last_path):
    trainer.load_lastpoint(train_conf.last_path)
else:
    pass
        
trainer.train()


# load the state of the best model we've seen based on early stopping
#checkpoint = torch.load(train_conf.ckpt_path)
#model.load_state_dict(checkpoint)

from rcnn.util import box_iou

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

BBOX_LABEL_NAMES = ('bg', 'TB',)

tp_path = '/home/data/tp'
fp_path = '/home/data/fp'

right_box = 0
total_box = 0
nontb = 0
tb = 0
fp = 0
fn = 0

#python -m torch.distributed.launch --nproc_per_node=4  --nnodes=1 --node_rank=0 play_medical.py 

for index, (img, target) in enumerate(test_dataset):
    
    with torch.no_grad():
        result = model([img.to(device)])
    
    image = img.numpy() * 255
    image = image.transpose((1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    showimg(image, 0, re, BBOX_LABEL_NAMES)
    cv2.imshow('ok', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(index)
    target['labels'] += 1
    
    if target['boxes'].shape[0] == 1 and target['boxes'].sum(dim=1) < 1e-7:
        nontb += 1
        if result[0]['boxes'].shape[0] > 0:
            fp += 1
            total_box += result[0]['boxes'].shape[0]
            showimg(image, 0, result[0], BBOX_LABEL_NAMES)
            #cv2.imwrite(os.path.join(fp_path, "{}.png".format(index)), image)
            cv2.imwrite(os.path.join(fp_path, test_dataset.get_imgName(index)), image)
        continue
        
    tb += 1
    if result[0]['boxes'].shape[0] == 0:
        fn += 1
        continue
        
    value, idx = box_iou(result[0]['boxes'], target['boxes'].to(result[0]['boxes'])).max(dim=1)
    gt_idx = []
    for i in range(result[0]['boxes'].shape[0]):
        if value[i] < 0.1: 
            gt_idx.append(-1)
        else:
            gt_idx.append(target['labels'][idx[i]].item())
            
    right_box += (np.array(gt_idx) == result[0]['labels'].cpu().numpy()).sum()
    total_box += len(gt_idx)
    showimg(image, 0, result[0], BBOX_LABEL_NAMES)
    showimg(image, 200, target, BBOX_LABEL_NAMES)
    cv2.imwrite(os.path.join(tp_path, test_dataset.get_imgName(index)), image)
    
print( total_box, right_box)
print(nontb, fp, tb, fn)

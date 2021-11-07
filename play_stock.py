#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torchvision
import torch
import os
import torch.nn.functional as F
import math
import cv2
from torch.utils.data import Dataset
from PIL import Image
import json
import shutil
import time

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,        
        #filename= 'dat.log',
)


def mask2point(mask):
    img = (mask > 0.5).cpu().numpy()
    img = cv2.copyMakeBorder(img.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(img, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE, offset=(-1,-1))
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]
    area = np.array([cv2.contourArea(i) for i in polygons])
    return polygons[area.argmax()] if area.shape[0]>0  else None
    
def showimg(image, color, target=None, classes=None):

    for i, b in enumerate(target["boxes"]):
        b = b.cpu().long().numpy()
        print(i,b)
        if 'scores' in target:
            txt = "{} {:.3f}".format(classes[target["labels"][i].item()], target["scores"][i].item())
        else:
            txt = "{}".format(classes[target["labels"][i].item()])
        print(txt)
        
        print(b[0], b[1])        
        image = cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), (color, color, color), 2)
        image = cv2.putText(image, txt, (b[0], b[1]), cv2.FONT_HERSHEY_COMPLEX, 1., (color, color, color), 2)
        
        '''
        t = target["masks"]
        p = t[torch.where(t>0)]
        #print(target["masks"], p.shape, p)
        '''        
        point = mask2point(target["masks"][i])
        print(point)
        if point is not None:
            cv2.polylines(image, [point], isClosed=True, color=(color, color, color), thickness=2)

    
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
    masks = [] #np.empty(0)
    if os.path.exists(label_path):
        with open(label_path, "r") as lf:            
            jason = json.load(lf)
            shapes = jason['shapes']
            for i in range(len(shapes)):
                points = np.array(shapes[i]['points'])
                mask = np.zeros(img.size)
                
                points = np.trunc(points).astype(int)
                #mask[points[0][0]:points[1][0], points[0][1]:points[1][1]] = 1
                cv2.drawContours(mask, [np.trunc(points).astype(int)], 0, (1,1,1), cv2.FILLED)
                #masks.append(torch.from_numpy(mask[None]))
                masks.append(mask)
                '''
                mask = mask[None]
                if masks.shape[0] == 0:
                    masks = mask
                else:
                    masks = np.concatenate((masks ,mask),axis=0)
                '''
                
                boxes.append([np.min(points[:,0]), np.min(points[:,1]), np.max(points[:,0]), np.max(points[:,1])])
                labels.append(0)  
                

    else:
        boxes.append([0, 0, 0, 0])
        labels.append(0)
        mask = np.zeros(img.size)        
        #masks.append(torch.from_numpy(mask[None]))
        '''
        mask = np.zeros(img.size)[None]
        masks = mask
        '''        
        masks.append(mask)
        
    #masks = torch.stack(masks, dim=0) #
    masks = torch.tensor(np.array(masks), dtype=torch.float32) #torch.cat(masks, dim=0)
    #print(masks.shape, masks.dtype)
    
    boxes = torch.tensor(boxes, dtype=torch.float32)
    #print(boxes.shape, boxes)
    
    labels = torch.tensor(labels)
    #print(labels.shape, labels.dtype)
    #exit()
    
    return img, dict(boxes=boxes, labels=labels, masks=masks) #torch.from_numpy(masks))
        



class StockDataset(Dataset):
    
    def __init__(self, path ):
        self.path = path
        _, self.imgList = dirtolist(path)
        self.imgList.sort(reverse = False)
        
    
    def __getitem__(self, idx):
        img, target = self.get_example(idx)
        img = torchvision.transforms.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

    def get_example(self, idx):
        
        imgName = self.imgList[idx]
        basename = os.path.basename(imgName)        
        #dirs = imgName.split("\\")
        label = self.path + '/label/' + basename[:-3] + 'json'
        img, target= img2data(imgName, label)
        #print(imgName, label)
        return img, target
        
    def get_imgName(self, idx):
        imgName = self.imgList[idx]
        basename = os.path.basename(imgName)
        return basename        
        #dirs = imgName.split("/")
        #return dirs[-1]
        
    
train_dir = 'e:/data'
valid_dir = 'e:/data'
train_dataset = StockDataset(train_dir)
test_dataset = StockDataset(valid_dir)


from rcnn.model import Rcnn, RcnnConfig

# we'll do something a bit smaller
mconf = RcnnConfig(backbone_name = "resnet50", havemask=True)
model = Rcnn(mconf)

from rcnn.trainer import Trainer, TrainerConfig

batch_size = 32
tokens_per_epoch = len(train_dataset) // batch_size
train_epochs = 60 # todo run a bigger model and longer, this is tiny

# initialize a trainer instance and kick off training
train_conf = TrainerConfig(max_epochs=train_epochs, batch_size=batch_size, learning_rate=1e-3,
                      betas = (0.9, 0.95), weight_decay=0.0005,
                      lr_decay=True, warmup_tokens=tokens_per_epoch*5 , final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path='test_model.pt', last_path='models/model-36.pt',
                      num_workers=0)
trainer = Trainer(model, train_dataset, test_dataset=None, config=train_conf)



if train_conf.last_path is not None and os.path.exists(train_conf.last_path):
    trainer.load_lastpoint(train_conf.last_path)
    
trainer.train()


# load the state of the best model we've seen based on early stopping
#checkpoint = torch.load(train_conf.ckpt_path)
#model.load_state_dict(checkpoint)

from rcnn.util import box_iou

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
BBOX_LABEL_NAMES = ('bg', 'TB',)

'''
{
  "version": "4.5.6",
  "flags": {},
  "shapes": [
    {
      "label": "1",
      "points": [
        [
          205.68783068783068,
          215.34391534391534
        ],
        [
          419.44444444444446,
          486.2433862433862
        ]
      ],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    }
  ],
  "imagePath": "000002_20001106.jpg",
  "imageData": null 
  "imageHeight": 512,
  "imageWidth": 512
}
'''
from copy import deepcopy



def createPostive(dataset):

    tb_path = '../tb'

    baseDict = {"version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": None,
            "imageData": None,
            "imageHeight": 512,
            "imageWidth": 512
            }
            
    print(len(dataset))

    start = time.time()
    for index, (img, target) in enumerate(dataset):
        with torch.no_grad():
            result = model([img.to(device)])
        print(index)
        if result[0]['boxes'].shape[0] == 0: continue
    
        json_dict = deepcopy(baseDict)
    
        imgName = os.path.basename(dataset.imgList[index])
        jsonName = imgName[:-3] + 'json'
        ansName = imgName[:-4] +'_ans.jpg'
    
        json_dict["imagePath"] = imgName
        for box, score, label in zip(result[0]['boxes'], result[0]['scores'], result[0]['labels']):
            if score.item() < 0.5: continue
            one = {"group_id": None, "shape_type": "rectangle", "flags": {}}
            one["label"] = str(label.item())
            one["points"] = []
            temp = box.cpu().numpy().tolist()
            one["points"].append(temp[:2])
            one["points"].append(temp[2:])
            json_dict["shapes"].append(one)
        if len(json_dict["shapes"]) == 0: continue
        with open(os.path.join(tb_path, jsonName), 'w') as f:
            json_str = json.dumps(json_dict, ensure_ascii = False,indent=4)
            f.write(json_str)        
        shutil.copyfile(dataset.imgList[index], os.path.join(tb_path, imgName))
        if index + 15 > len(dataset): ans_index = len(dataset) -1 
        else: ans_index = index + 15
    
        temp = os.path.basename(test_dataset.imgList[ans_index])
        if temp[:6] !=  imgName[:6]: continue
        shutil.copyfile(test_dataset.imgList[ans_index], os.path.join(tb_path, ansName))
    print('total time is %d' % (time.time() - start) )  

def labelme(dataset):

    tb_path = '../tb'

    baseDict = {"version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": None,
            "imageData": None,
            "imageHeight": 512,
            "imageWidth": 512
            }
            
    print(len(dataset))

    start = time.time()
    for index, (img, target) in enumerate(dataset):
        imgName = os.path.basename(dataset.imgList[index])
        if int(imgName[:6]) <= 554: 
            continue
        with torch.no_grad():
            result = model([img.to(device)])
        print(index)
        if result[0]['boxes'].shape[0] == 0: continue
    
        json_dict = deepcopy(baseDict)
    
        #imgName = os.path.basename(dataset.imgList[index])
        jsonName = imgName[:-3] + 'json'
        #ansName = imgName[:-4] +'_ans.jpg'
    
        json_dict["imagePath"] = imgName
        for box, score, label in zip(result[0]['boxes'], result[0]['scores'], result[0]['labels']):
            if score.item() < 0.5: continue
            one = {"group_id": None, "shape_type": "rectangle", "flags": {}}
            one["label"] = str(label.item())
            one["points"] = []
            temp = box.cpu().numpy().tolist()
            one["points"].append(temp[:2])
            one["points"].append(temp[2:])
            json_dict["shapes"].append(one)
        if len(json_dict["shapes"]) == 0: continue
        if int(imgName[11:15]) > 1225: continue
        with open(os.path.join(tb_path, jsonName), 'w') as f:
            json_str = json.dumps(json_dict, ensure_ascii = False,indent=4)
            f.write(json_str)        
        shutil.copyfile(dataset.imgList[index], os.path.join(tb_path, imgName))
        for num in range(1, 15):
            if index + num >= len(dataset): break
            temp = os.path.basename(dataset.imgList[index + num])
            if temp[:6] !=  imgName[:6]: break
            ansName = os.path.basename(dataset.imgList[index + num])
            shutil.copyfile(dataset.imgList[index + num], os.path.join(tb_path, ansName))
    print('total time is %d' % (time.time() - start) )  
    
    
def labelImg(dataset):


    baseDict = {"version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": None,
            "imageData": None,
            "imageHeight": 512,
            "imageWidth": 512
            }
            
    print(len(dataset))

    start = time.time()
    for index, (img, target) in enumerate(dataset):
        with torch.no_grad():
            result = model([img.to(device)])
        print(index)
        if result[0]['boxes'].shape[0] == 0: continue
    
        json_dict = deepcopy(baseDict)
        
        imgPath = os.path.dirname(dataset.imgList[index])
        imgName = os.path.basename(dataset.imgList[index])
        jsonName = imgName[:-3] + 'json'
     
        json_dict["imagePath"] = imgName
        for box, score, label in zip(result[0]['boxes'], result[0]['scores'], result[0]['labels']):
            if score.item() < 0.5: continue
            one = {"group_id": None, "shape_type": "rectangle", "flags": {}}
            one["label"] = str(label.item())
            one["points"] = []
            temp = box.cpu().numpy().tolist()
            one["points"].append(temp[:2])
            one["points"].append(temp[2:])
            json_dict["shapes"].append(one)
        if len(json_dict["shapes"]) == 0: continue
        with open(os.path.join(imgPath, jsonName), 'w') as f:
            json_str = json.dumps(json_dict, ensure_ascii = False,indent=4)
            f.write(json_str)        
    print('total time is %d' % (time.time() - start) )  


'''
m = [[img, target] for index, (img, target) in enumerate(test_dataset) if index <= 2]
#print(m)
print(m[0][0].shape)
print(m[1][0].shape)

with torch.no_grad():
    result = model([m[0][0].to(device), m[1][0].to(device)])
print(result)

image = m[0][0].numpy()
image = image.transpose((1, 2, 0))
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
showimg(image, 116, result[0], BBOX_LABEL_NAMES)
cv2.imshow('ok', image)
cv2.waitKey(0)

image = m[1][0].numpy()
image = image.transpose((1, 2, 0))
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
showimg(image, 116, result[0], BBOX_LABEL_NAMES)
cv2.imshow('ok', image)
cv2.waitKey(0)

exit()
'''

from torch.utils.data.dataloader import DataLoader

def showAnswer(dataset):


    loader = DataLoader(dataset, shuffle=True, collate_fn=lambda x: tuple(zip(*x)),  batch_size=4)
 
    for index, (imgs, target) in enumerate(loader):

        images = [img.to(device) for img in imgs]
        print([{k: v for k, v in t.items()} for t in target]) 
    
        with torch.no_grad():
            results = model(images)
    
        for img, result in zip(imgs, results):
            image = img.numpy()
            image = image.transpose((1, 2, 0))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            showimg(image, 116, result, BBOX_LABEL_NAMES)
            cv2.imshow('ok', image)
            cv2.waitKey(0)
#labelme(test_dataset)    
#labelImg(test_dataset)
#createPostive(test_dataset)
showAnswer(train_dataset)    
exit()
    

tp_path = '/home/data/tp'
fp_path = '/home/data/fp'

right_box = 0
total_box = 0
nontb = 0
tb = 0
fp = 0
fn = 0

#python -m torch.distributed.launch --nproc_per_node=1  --nnodes=1 --node_rank=0 play_stock.py 
#python -m torch.distributed.run --nproc_per_node=1  --nnodes=1 --node_rank=0 play_stock.py
for index, (img, target) in enumerate(test_dataset):
    
    with torch.no_grad():
        result = model([img.to(device)])
    
    image = img.numpy() * 255
    image = image.transpose((1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    showimg(image, 0, result, BBOX_LABEL_NAMES)
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

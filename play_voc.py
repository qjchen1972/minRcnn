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
    print(img, img.shape, img.dtype)
    exit()
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
        mask = target["masks"][i]
        print(mask, mask.shape)
        point = mask2point(target["masks"][i])
        if point is not None:
            cv2.polylines(image, [point], isClosed=True, color=(color,color, color), thickness=2)

'''
def mask2point(mask):
    #print(mask.shape)
    img = (mask > 0.5).cpu().numpy()
    #print(img.shape, (mask > 0.5).sum())
    
    img = cv2.copyMakeBorder(img.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(img, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE, offset=(-1,-1))
    #print(polygons)
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
        #print( target["masks"].shape, i)
        if target["masks"].shape[0] >i:
            point = mask2point(target["masks"][i])
        else:
            print('mask is out')
        #print(type(point))
        if point is not None:
            cv2.polylines(image, [point], isClosed=True, color=(color,color, color), thickness=1)
        else:
            print('mask is None')
            
'''
            
from torch.utils.data import Dataset
from PIL import Image
import json
import xml.etree.ElementTree as ET

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
)

class VocDataset(Dataset):
    
    def __init__(self, data_dir, split):
        
        self.data_dir = data_dir
        self.split = split
        id_file = os.path.join(data_dir, "ImageSets/Segmentation/{}.txt".format(split))
        self.ids = [id_.strip() for id_ in open(id_file)]
        self.classes = VOC_CLASSES
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img, target = self.get_example(img_id)
        img = torchvision.transforms.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.ids)
        
    def get_imgName(self, idx):
        
        return "{}.jpg".format(self.ids[idx])        
        
    
    def get_example(self, img_id):
    
        image = Image.open(os.path.join(self.data_dir, "JPEGImages/{}.jpg".format(img_id))).convert("RGB")

        masks = Image.open(os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_id)))
        masks = torchvision.transforms.ToTensor()(masks)        
        uni = masks.unique()
        uni = uni[(uni > 0) & (uni < 1)]
        masks = (masks == uni.reshape(-1, 1, 1)).to(torch.uint8)
        
        anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id)))
        boxes = []
        labels = []
        for obj in anno.findall("object"):
            bndbox = obj.find("bndbox")
            bbox = [int(bndbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
            name = obj.find("name").text
            label = self.classes.index(name)
            boxes.append(bbox)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)
        assert boxes.shape[0] == labels.shape[0]
        mask_padded = torch.full((boxes.shape[0], masks.shape[-2],  masks.shape[-1]), 0, dtype=torch.float)       
        if boxes.shape[0] > masks.shape[0]:
            mask_padded[:masks.shape[0]] = masks
        else:
            mask_padded = masks[:boxes.shape[0]]
            
        return image, dict(boxes=boxes, labels=labels, masks=mask_padded)
        
        '''
        boxes = np.stack(boxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)
        return image, boxes,labels, masks.numpy()
        '''

data_dir = '/home/VOCdevkit/VOC2012'
train_dataset = VocDataset(data_dir, 'trainval')
test_dataset = VocDataset(data_dir, 'val')

'''
from detection.mask_rcnn import maskrcnn_resnet50_fpn
from detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torch.onnx
import io
import onnx
import caffe2.python.onnx.backend as backend
from caffe2.python.predictor import mobile_exporter

#/root/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so
#/root/anaconda3/lib/python3.7/site-packages/torch/include/torch/csrc/api/include/

# we'll do something a bit smaller
#model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=len(VOC_CLASSES)+1, pretrained_backbone=True)


#script_module = torch.jit.script()
#print(model)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) #, progress=True, num_classes=len(VOC_CLASSES)+1, pretrained_backbone=True)

x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
#print(r.code)
model.eval()
scripted_module = torch.jit.script(model)
#scripted_module = torch.jit.trace(model, x)
torch.jit.save(scripted_module, 'mymodule.pt')
r = torch.jit.load('mymodule.pt')
#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
print(r.code)
t = r(x)
print(t)
exit()


#device = torch.device("cpu") #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# optionally, if you want to export the model to ONNX:
torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
exit()

def create_caffe2_model(onnx_model_path, out_init_path, out_pred_path):

    model = onnx.load(onnx_model_path)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
        # Print a human readable representation of the graph
        #print(onnx.helper.printable_graph(model.graph))
        #rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    rep = backend.prepare(model, device="CPU")
        # For the Caffe2 backend:
        # rep.predict_net is the Caffe2 protobuf for the network
        # rep.workspace is the Caffe2 workspace for the network
        #(see the class caffe2.python.onnx.backend.Workspace)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    outputs = rep.run(x)

    print(outputs[0])

    c2_workspace = rep.workspace
    c2_graph = rep.predict_net
    init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_graph, c2_graph.external_input)

    with io.open(out_init_path, 'wb') as f:
        f.write(init_net.SerializeToString())

    with open(out_pred_path, 'wb') as f:
        f.write(predict_net.SerializeToString())
        
create_caffe2_model("faster_rcnn.onnx", 'init.pb', 'pred.pb')

exit()

def create_onnx_model(model, model_path, input_names=["input"], output_names=["output"]):

    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    print(dummy_input[0].shape)
    script_module = torch.jit.trace(model, dummy_input)
    print(script_module)
    p1 = script_module(dummy_input)
    #print(p1, p1.shape, p2, p2.shape, p3, p3.shape)
    script_module.save(model_path)
    exit()
    
    torch.onnx.export(model,  
                      dummy_input,
                      model_path, 
                      verbose=True,
                      input_names=input_names, 
                      output_names=output_names)
                      #keep_initializers_as_inputs=True)
        
create_onnx_model(model, 'onnx_proto')        

def run_onnx():
    
    onnx_model = 'onnx_proto'
    out_init = 'init.pb'
    out_pred = 'pred.pb'
    
    onnx = OnnxProc(opt.load_path, 3, 512, 512)
    onnx.create_onnx_model(onnx_model)
    onnx.create_caffe2_model(onnx_model, out_init, out_pred)
    
exit()
'''


from rcnn.model import Rcnn, RcnnConfig

# we'll do something a bit smaller
mconf = RcnnConfig(backbone_name = "resnet50", num_classes=len(VOC_CLASSES)+1, havemask=True)
model = Rcnn(mconf)

    
    
    
from rcnn.trainer import Trainer, TrainerConfig

batch_size = 4
tokens_per_epoch = len(train_dataset) // batch_size
train_epochs = 100 # todo run a bigger model and longer, this is tiny

# initialize a trainer instance and kick off training
train_conf = TrainerConfig(max_epochs=train_epochs, batch_size=batch_size, learning_rate=1e-2,
                      betas = (0.9, 0.95), weight_decay=0.0005,
                      lr_decay=True, warmup_tokens=tokens_per_epoch*5 , final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path='test_model.pt', last_path='models/model-7.pt',
                      num_workers=8)
trainer = Trainer(model, train_dataset, test_dataset, train_conf)

if train_conf.last_path is None:
    pass
elif os.path.exists(train_conf.last_path):
    trainer.load_lastpoint(train_conf.last_path)
else:
    pass
        



#trainer.train()
        
# load the state of the best model we've seen based on early stopping
#checkpoint = torch.load(train_conf.ckpt_path)
#model.load_state_dict(checkpoint)
'''
model.eval()
scripted_module = torch.jit.script(model)
torch.jit.save(scripted_module, 'mymodule2.pt')
r = torch.jit.load('mymodule.pt')
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
print(r.code)
t = r(x)
print(t)
exit()
'''

from rcnn.util import box_iou

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

BBOX_LABEL_NAMES = ("bg",) + VOC_CLASSES

#indices =  torch.arange(len(test_dataset)).tolist()
#print(indices)
#loader = torch.utils.data.Subset(test_dataset, indices)
tp_path = '/home/data/tp'
fp_path = '/home/data/fp'
right_box = 0
total_box = 0

nontb = 0
tb = 0
fp = 0
fn = 0
for index, (img, target) in enumerate(test_dataset):
    
    with torch.no_grad():
        result = model([img.to(device)])
    
    image = img.numpy() * 1
    image = image.transpose((1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
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
    cv2.imshow('ok', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
    
    #showimg(image, 200, target, BBOX_LABEL_NAMES)
    cv2.imwrite(os.path.join(tp_path, test_dataset.get_imgName(index)), image)
    
print( total_box, right_box)
print(nontb, fp, tb, fn)

'''
for i, (img, ori_box, ori_label, ori_mask) in enumerate(test_dataset):

    ori_box, ori_label, ori_mask = torch.from_numpy(ori_box), torch.from_numpy(ori_label),  torch.from_numpy(ori_mask)
    ori_result = dict(boxes=ori_box, labels=ori_label + 1, masks=ori_mask)
    ori_data = torchvision.transforms.ToTensor()(ori_img)
    img, _,  _  = med_transformer(ori_data)
    img = img[None]
    with torch.no_grad():
        result = model(img.to(device))
        result = postprocess(result, img.shape[-2:], ori_data.shape[-2:])
    image = np.array(ori_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    print(i)
    if ori_result['boxes'].shape[0] == 1 and ori_result['boxes'].sum(dim=1) < 1e-7:
        nontb += 1
        if result['boxes'].shape[0] > 0:
            fp += 1
            p_tb += 1
            total += result['boxes'].shape[0]
            showimg(image, 0, result, BBOX_LABEL_NAMES)
            cv2.imwrite(os.path.join(fp_path, dirs[-1]), image)
        else:
            p_ntb += 1           
        continue
        
    tb += 1
    if result['boxes'].shape[0] == 0:
        fn += 1
        p_ntb += 1
        continue
    
    p_tb += 1
    
    value, idx = box_iou(result['boxes'], ori_result['boxes'].to(result['boxes'])).max(dim=1)
    gt_idx = []
    for i in range(result['labels'].shape[0]):
        if value[i] < 0.1: 
            gt_idx.append(-1)
        else:
            gt_idx.append(ori_result['labels'][idx[i]].item())
    right += (np.array(gt_idx) == result['labels'].cpu().numpy()).sum()
    total += len(gt_idx)
    showimg(image, 0, result, BBOX_LABEL_NAMES)
    #showimg(image, 200, ori_result, BBOX_LABEL_NAMES)
    cv2.imwrite(os.path.join(tp_path, "{}.png".format(img_id)), image)

print( total, right)
print(nontb, tb, p_tb, fp, p_ntb, fn)
'''




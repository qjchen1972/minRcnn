"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import math
import logging

from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data.dataloader import DataLoader
from rcnn.util import set_seed

logger = logging.getLogger(__name__)

#python -m torch.distributed.launch --nproc_per_node=1  --nnodes=1 --node_rank=0 play_medical.py

class ScheduledOptim(object):
    
    def __init__(self, optimizer, init_lr, n_warmup_steps, n_current_steps=0, final_steps=0):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        self.init_lr = init_lr
        self.final_steps = final_steps

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.power(self.n_warmup_steps, 0.5) * np.min([np.power(self.n_current_steps, -0.5),
                                                            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
    def _update_learning_rate(self):
        
        self.n_current_steps += 1
        if self.final_steps > 0:            
            if self.n_current_steps < self.n_warmup_steps:
                lr_mult = float(self.n_current_steps) / float(max(1, self.n_warmup_steps))
            else:
                progress = float(self.n_current_steps - self.n_warmup_steps) / float(max(1, self.final_steps - self.n_warmup_steps))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            
            '''    
            elif self.n_current_steps < 15000:
                progress = float(self.n_current_steps - self.n_warmup_steps) / float(max(1, self.final_steps - self.n_warmup_steps))
                lr_mult = 1.0 #max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            elif self.n_current_steps < 30000:
                progress = float(self.n_current_steps - self.n_warmup_steps) / float(max(1, self.final_steps - self.n_warmup_steps))
                lr_mult = 0.1 #max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
            else:
                lr_mult = 0.01
            '''
            
            self.lr = self.init_lr * lr_mult
        else:
            self.lr = self.init_lr * self._get_lr_scale()            
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr
            
class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    last_path = None
    num_workers = 0 # for DataLoader
    distributed = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    
class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
    
        if config.distributed:
            self.local_rank, self.rank, self.world_size = self.init_distributed_mode()
        else:
            self.rank = 0
            self.world_size = 1
            
        set_seed(42 + self.rank)
        
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        self.start_epoch = 0
        self.best_loss = float('inf')
       
       
        self.model = self.model.to(self.device)
        
        if config.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if test_dataset is not None else None
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            self.train_sampler = torch.utils.data.RandomSampler(train_dataset)
            self.test_sampler = torch.utils.data.SequentialSampler(test_dataset) if test_dataset is not None else None
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        
        # take over whatever gpus are on the system
        '''
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()            
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        '''
            
    def reduce_dict(self, input_dict, average=True):
        if self.world_size < 2:
            return input_dict
        with torch.no_grad():
            names = []
            values = []
            for k in sorted(input_dict.keys()):
                names.append(k)
                values.append(input_dict[k])
            values = torch.stack(values, dim=0)
            torch.distributed.all_reduce(values)
            if average:                                               
                values /= self.world_size
            reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict
    
    def init_distributed_mode(self):
    
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ["LOCAL_RANK"])
        #device = torch.device("cuda", local_rank)
        
        torch.cuda.set_device(local_rank)
        #torch.cuda.set_device(rank % torch.cuda.device_count())
        # windows下只支持gloo, linux下使用nccl
        torch.distributed.init_process_group(backend="gloo",#backend="nccl", 
                                             init_method='env://',
                                             world_size=world_size, 
                                             rank=rank)
        torch.distributed.barrier()
        
        return local_rank, rank, world_size
        
    
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        if self.rank != 0: return
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
        
    def save_lastpoint(self, epoch, best_loss=float('inf')):
        if self.rank != 0: return
        torch.save({'epoch':epoch, 'best_loss':best_loss, 'state_dict':self.model.state_dict()}, './models/'+'model-'+str(epoch)+'.pt')
        
    def load_lastpoint(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['state_dict'])
        self.start_epoch = state_dict['epoch'] + 1
        self.best_loss = state_dict['best_loss']
        #self.init_tokens = self.start_epoch * (len(self.train_dataset) // self.config.batch_size)
        
    def collate_fn(self, batch):
        
        return tuple(zip(*batch))
        return [val[0] for val in batch], [val[1] for val in batch]
    
    def train(self):
        
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.config_optimizers(config)
        init_tokens = self.start_epoch * (len(self.train_dataset) // self.config.batch_size)
        optim_schedule = ScheduledOptim(optimizer, init_lr=config.learning_rate, 
                                        n_warmup_steps=config.warmup_tokens, n_current_steps=init_tokens,
                                        final_steps=config.final_tokens)

        def run_epoch(split):
            is_train = split == 'train'
            #model.train(is_train)
            model.train()
            (data, sampler)  = (self.train_dataset, self.train_sampler)  if is_train else (self.test_dataset, self.test_sampler)         
            
            '''
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                collate_fn = self.collate_fn,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            '''
            loader = DataLoader(data, pin_memory=True,
                                collate_fn = self.collate_fn,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                sampler=sampler)
         
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            lossName = []
            
            for it, (images, target) in pbar:

                # place data on the correct device
                images = [img.to(self.device) for img in images]
                target = [{k: v.to(self.device) for k, v in t.items()} for t in target]
                #print([{k: v for k, v in t.items()} for t in target])                
                
                # forward the model
                with torch.set_grad_enabled(is_train):
                    loss_dict = model(images, target)
                    loss = sum(loss_dict.values())
                    if it == 0: lossName = loss_dict.keys()
                    
                    loss_dict_reduced = self.reduce_dict(loss_dict)
                    losses.append([loss_dict_reduced[k].item() for k in loss_dict_reduced.keys()])
                    
                    '''
                    losses.append([loss_dict_reduced['rpn_objectness_loss'].item(),
                                   loss_dict_reduced['rpn_box_loss'].item(), 
                                   loss_dict_reduced['roi_classifier_loss'].item(),
                                   loss_dict_reduced['roi_box_loss'].item(),
                                   loss_dict_reduced['roi_mask_loss'].item(), 
                                   sum(loss_dict_reduced.values()).item()])
                    #loss = loss.mean() #collapse all losses if they are scattered on multiple gpus
                    #losses.append(loss.item())
                    '''

                if is_train:
                    # backprop and update the parameters
                    optim_schedule.zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optim_schedule.step_and_update_lr()                    
                    # report progress
                    pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {optim_schedule.lr:e}")

            #if not is_train:
            #print(losses, np.array(losses).shape, np.mean(losses, axis=0))
            mean_loss = np.mean(losses, axis=0)
            dict_loss = dict(zip(lossName, mean_loss))
            if not is_train:   
                if self.rank == 0:
                    logger.info('test loss = %.3f  %s' %( np.sum(mean_loss), str(dict_loss)))
                    '''                   
                    logger.info("test loss: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f ", 
                                         mean_loss[5], mean_loss[0], mean_loss[1],mean_loss[2],mean_loss[3],mean_loss[4])
                    '''                                
            else:
                if self.rank == 0:
                    logger.info('train loss = %.3f %s' %( np.sum(mean_loss), str(dict_loss)))
                    '''
                    logger.info("train loss: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f ", 
                                         mean_loss[5], mean_loss[0], mean_loss[1],mean_loss[2],mean_loss[3],mean_loss[4])   
                    '''                                         
            return np.sum(mean_loss)

        for epoch in range(self.start_epoch, config.max_epochs):
            
            if config.distributed:
                self.train_sampler.set_epoch(epoch)

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
               
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < self.best_loss            
            if self.config.ckpt_path is not None and good_model:                
                self.save_checkpoint()
                if self.test_dataset is not None: self.best_loss = test_loss
                
            self.save_lastpoint(epoch, self.best_loss)

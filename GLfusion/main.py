import glob
import os
import sys
import argparse
import time
import math
import random
import logging
import nibabel as nib
import numpy
import numpy as np
import  re
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate
from monai.data import DataLoader
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
from datasets.loader import *
from utils.tools import get_world_size, get_global_rank, get_local_rank, get_master_ip
from utils.metrics import DiceScore
from models.unet import U_Net, R2AttU_Net, AttU_Net, R2U_Net
from models.segmentation import deeplabv3_resnet50_iekd
from models.res3dunet import ResUNet
from torch.utils.data import Dataset,ConcatDataset
from models.ours import *
from models.CEN import refinenet
from models.ResNet_AVSModel import *
from utils.PCGrad import PCGrad
import copy
torch.autograd.set_detect_anomaly(True)
matplotlib.use('Agg')
# os.environ['CUDA_ENABLE_DEVICES'] = '0,1,2,3'
parts_num = {'1':2, '2':1, '3':2, '4':4}
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def create_segmenter(num_layers, num_classes, num_parallel, bn_threshold, gpu):
    """Create Encoder; for now only ResNet [50,101,152]"""
    segmenter = refinenet(num_layers, num_classes, num_parallel, bn_threshold)
    assert(torch.cuda.is_available())
    segmenter.to(gpu[0])
    segmenter = torch.nn.DataParallel(segmenter, gpu)
    return segmenter
def L1_penalty(var):
    return torch.abs(var).sum()
def create_optimisers(lr_enc, lr_dec, mom_enc, mom_dec, wd_enc, wd_dec, param_enc, param_dec, optim_dec):
    """Create optimisers for encoder, decoder and controller"""
    optim_enc = torch.optim.SGD(param_enc, lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc)
    if optim_dec == 'sgd':
        optim_dec = torch.optim.SGD(param_dec, lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec)
    elif optim_dec == 'adam':
        optim_dec = torch.optim.Adam(param_dec, lr=lr_dec, weight_decay=wd_dec, eps=1e-3)

    return optim_enc, optim_dec

class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        torch.backends.cudnn.benchmark = config['train']['cudnn']
        self.distributed = config['train']['distributed']
        self.device = config['train']['device']
        self.local_rank = config['train']['local_rank']
        self.seg_parts = config['train']['seg_parts']
        self.view_num = config['train']['view_num']
        self.test_view = config['train']['test_view']
        self.device_ids = config['train']['device_ids']
        self.is_load = config['train']['is_load']
        self.dense_cyc = config['train']['dense_cyc']
        self.data_list_path = config['train']['data_list_path']
        self.outchannel_list = parts_num
        self.out_channels = parts_num[self.view_num[0]] if self.seg_parts else 1

        self.init_model_opt()

        self.loss_weight = []

        self.latest_eopch = 0

        #self.load()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.alpha = config['train']['alpha']

        self.print_val = True if self.local_rank == config['train']['enable_GPUs_id'][0] else False

        root = '/home/xmli/zyzheng/dataset_pa_iltrasound_nii_files_3rdcenters'

        discrete_infos = np.load(
            f'./infos/save_infos_reg_v2.npy',
            allow_pickle=True).item()
        infos = np.load(f'./infos/infos_unlab.npy', allow_pickle=True).item()

        self.infos = infos
        # self.infos_cyc = infos_cyc
        self.train_dataset = {}
        self.train_cyc_dataset = {}
        self.valid_dataset = {}
        self.test_dataset = {}
        self.train_gnd_dataset = {}
        self.train_pseudo_dataset = {}
        self.train_list = np.load(
            './data_list/train_list.npy')
        self.test_list = np.load(
            './data_list/test_list.npy')
        self.val_list = np.load(
            './data_list/val_list.npy')
        for view in self.view_num:
            self.train_dataset[view] = Seg_PAHDataset(discrete_infos, root, is_train=True, set_select=self.config['train']['use_data'],
                                                      view_num=[view], seg_parts=self.seg_parts,data_list=self.train_list,require_id=False)

            self.train_pseudo_dataset[view] = Aligned_Video_Seg_PAHDataset(infos, root, is_train=True,
                                                      set_select=self.config['train']['use_data'],
                                                      view_num=[view], seg_parts=self.seg_parts,clip_length=self.config['train']['clip_length'],
                                                      data_list=self.train_list, require_id=False,random_sample=False)
            self.valid_dataset[view] = Seg_PAHDataset(discrete_infos, root, is_train=False, data_list=self.val_list,
                                                set_select=self.config['train']['use_data'], view_num=[view], seg_parts=self.seg_parts,require_id=False)

            self.test_dataset[view] = Seg_PAHDataset(infos, root, is_train=False, data_list=self.test_list,
                                                set_select=self.config['train']['use_data'], view_num=[view], seg_parts=self.seg_parts,require_id=False)

        self.train_loader = {}
        self.train_gnd_loader = {}
        self.train_pseudo_loader = {}
        self.valid_loader = {}
        self.test_loader = {}
        self.train_cyc_loader={}
        for view in self.view_num:
            self.train_cyc_loader[view] = DataLoader(self.train_cyc_dataset[view], batch_size=1,
                                                 shuffle=False, num_workers=config['train']['num_workers'],
                                                 drop_last=True)
            self.train_loader[view] = DataLoader(self.train_dataset[view], batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['train']['num_workers'],drop_last=True)
            # self.train_gnd_loader[view] = DataLoader(self.train_gnd_dataset[view],
            #                                          batch_size=1, shuffle=False,
            #                                          num_workers=config['train']['num_workers'],drop_last=True)
            self.train_pseudo_loader[view] = DataLoader(self.train_pseudo_dataset[view],
                                                     batch_size=1, shuffle=False,
                                                     num_workers=config['train']['num_workers'], drop_last=True)
        if self.print_val:
            self.writer = SummaryWriter(os.path.join(config['train']['log_dir']))

    def init_model_opt(self):
        self.model = Global_and_Local(view_num=self.view_num)
        config = self.config

        if self.is_load:
            self.load()
        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model = self.model.to(self.device)

        if config['net']['opt']['opt_name'] == 'SGD':
            self.optimizer = torch.optim.SGD(list(self.model.parameters()),
                                             lr=config['net']['opt']['lr'],
                                             weight_decay=config['net']['opt']['weight_decay'])
        elif config['net']['opt']['opt_name'] == 'Adam':
            self.optimizer = torch.optim.Adam(list(self.model.parameters()),
                                              lr=config['net']['opt']['lr'],
                                              weight_decay=config['net']['opt']['weight_decay'])

        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['net']['opt']['step_size'], gamma=0.8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.config['train']['num_epochs'])


    def train(self,is_backbone=False,is_cycle=False):
        count = 0
        min_val_index = -1
        min_val_loss = 99999999999

        for self.epoch in range(self.latest_eopch,self.config['train']['num_epochs']):
            y_true, y_pred = [], []

            if self.print_val:
                print('Start Epoch / Total Epoch: {} / {}'.format(self.epoch, self.config['train']['num_epochs']))

            self.model.train()

            total_itr_num = len(self.train_loader[self.view_num[0]])
            train_loader = {}
            train_pseudo_loader = {}

            for view in self.view_num:
                train_loader[view]=iter(self.train_loader[view])
                train_pseudo_loader[view]=iter(self.train_cyc_loader[view])

            for i in range(total_itr_num):
                imgs = {}
                masks = {}
                loss4view = {}

                seg_loss = 0
                cyc_loss = 0

                #multi-view supervised
                for view in self.view_num:
                    img,mask,index = train_loader[view].next()
                    imgs[view] = img.to(self.device)
                    masks[view] = mask.to(self.device) / 1.0

                pred_frames,pred_bb,_,_ = self.model(imgs)

                for view in self.test_view:
                    loss4view[view] = self.bce_loss(pred_frames[view], masks[view])
                    seg_loss += loss4view[view]
                if is_cycle:
                    cyc_segfeed = {}
                    for view in self.view_num:
                        (cyc_imgs, _, _,_) = train_pseudo_loader[view].next()
                        cyc_imgs = cyc_imgs.to(self.device).permute(0, 4, 1, 2, 3)
                        cyc_segfeed[view] = cyc_imgs.reshape(-1, cyc_imgs.shape[2], cyc_imgs.shape[3],
                                                             cyc_imgs.shape[4])

                    _,_,cyc_feat_out,_ = self.model(cyc_segfeed)#tchw

                    cyc_w = {
                        '1':1,
                        '3':1,
                        '4':1,
                    }

                    for view in self.view_num:
                        cyc_feat_out[view] = cyc_feat_out[view].sum(dim=(2, 3))
                        if self.dense_cyc == False:
                            cyc_loss += self.seg_cycle(cyc_feat_out[view], target_region=16, cyc_off=2, chunk_size=3,
                                                       temperature=10)
                        else:
                            cyc_loss += cyc_w[view] * self.dense_seg_cycle(cyc_feat_out[view], target_region=16, cyc_off=2, chunk_size=3,
                                                       temperature=10,soft_label=False,is_overlap=True)

                total_loss = seg_loss + 1e-2 * cyc_loss

                self.optimizer.zero_grad()

                total_loss.backward(retain_graph=False)

                self.optimizer.step()
                if self.print_val:
                    self.add_summary(self.writer, 'train/net_loss', total_loss.item(), count)
                    count += 1
                    if count % total_itr_num == 0:
                        pixel_acc, dice, precision, specificity, recall = {},{},{},{},{}
                        for view in self.test_view:
                            pixel_acc[view], dice[view], precision[view], specificity[view], recall[view] = self._calculate_overlap_metrics(masks[view], torch.where(nn.Sigmoid()(pred_frames[view]) > 0.5, 1, 0))

                    if self.config['train']['record_params']:
                        for tag, value in self.model.named_parameters():
                            tag = tag.replace('.', '/').replace('module', '')
                            self.add_summary(self.writer, tag, value.data.cpu().numpy(), sum_type='histogram')

            self.scheduler.step()

            if self.print_val:

                for view in self.test_view:
                    print('------Training Result for view {view}------\n \
                               Loss : {loss:.4f} \
                               Seg Loss : {seg_loss:.4f} \
                               Pixel Acc : {pixel_acc:.4f} \
                               Dice : {dice:.4f} \
                               Precision : {pre:.4f} \
                               Specificity : {specificity:.4f} \
                               Recall : {recall:.4f}'. \
                          format(view=view,loss=loss4view[view].item(), seg_loss=seg_loss.item(),
                                 pixel_acc=pixel_acc[view], dice=dice[view], pre=precision[view], specificity=specificity[view], recall=recall[view]))


                self.validation_and_test(net_root=None, is_fuse=True, raw_data=True)
                self.save(self.epoch)

                print('End Training Epoch: {}'.format(self.epoch))

    def validation_and_test(self,net_root=None,is_fuse = True,raw_data = True):

        if raw_data==True:
            test_info = np.load(f'./infos/test_infos.npy', allow_pickle=True).item()
        else:
            test_info = np.load(f'./rmyy_test_align_dataset/infos_discrete_unlab.npy',allow_pickle=True).item()

        val_list = ['0_0','0_2']
        test_list = ['0_1','0_3','0_4','0_5','0_6','0_7','0_8','0_9']
        test_dataset = {}
        val_dataset = {}
        test_loader = {}
        val_loader = {}
        val_dice_list = []
        for view in self.view_num:
            if raw_data==False:
                test_dataset[view] = Aligned_Video_Seg_PAHDataset(test_info, root=None, is_train=False, data_list=test_list,
                                             set_select=self.config['train']['use_data'], view_num=[view],
                                             clip_length=self.config['train']['clip_length'],
                                             seg_parts=self.seg_parts, require_id=False)

                val_dataset[view] = Aligned_Video_Seg_PAHDataset(test_info, root=None, is_train=False,
                                                                  data_list=val_list,
                                                                  set_select=self.config['train']['use_data'],
                                                                  view_num=[view],
                                                                  clip_length=self.config['train']['clip_length'],
                                                                  seg_parts=self.seg_parts, require_id=False)
            else:
                test_dataset[view] = Test_Seg_PAHDataset(test_info, is_train=False,data_list=test_list,
                                              set_select=['rmyy'], clip_length=40,
                                              view_num=[view], seg_parts=True)
                val_dataset[view] = Test_Seg_PAHDataset(test_info, is_train=False,data_list=val_list,
                                                         set_select=['rmyy'], clip_length=40,
                                                         view_num=[view], seg_parts=True)
            test_loader[view] = DataLoader(test_dataset[view], batch_size=1,shuffle=False, num_workers=config['train']['num_workers'],drop_last=True)
            val_loader[view] = DataLoader(val_dataset[view], batch_size=1, shuffle=False,num_workers=config['train']['num_workers'], drop_last=True)

        for net_num in range(100):
            if net_root!=None:
                print('Start Epoch / Total Epoch: {} / {}'.format(net_num, self.config['train']['num_epochs']))
                net_path = os.path.join(
                    net_root, 'net_{}.pth'.format(str(net_num).zfill(5)))
                data = torch.load(net_path, map_location=self.device)
                data['network'] = {'module.' + k: v for k, v in data['network'].items()}
                self.model.load_state_dict(data['network'], strict=True)
            self.model.eval()
            for dataset_type in ['Inner-val','Inner-test']:
                with torch.no_grad():
                    dataloader = {}
                    all_pred_frame = {}
                    all_mask = {}
                    pixel_acc, dice, precision, specificity, recall = {}, {}, {}, {}, {}
                    for view in self.view_num:
                        pixel_acc[view] = 0.
                        dice[view] = 0.
                        precision[view] = 0.
                        specificity[view] = 0.
                        recall[view] = 0.
                    for view in self.view_num:
                        all_pred_frame[view] = []
                        all_mask[view] = []
                    for view in self.view_num:
                        if dataset_type=='Inner-val':
                            dataloader[view] = iter(val_loader[view])
                        if dataset_type=='Inner-test':
                            dataloader[view] = iter(test_loader[view])
                    total_itr_num = len(dataloader[self.view_num[0]])
                    loss = 0
                    loss4view = {}
                    for view in self.view_num:
                        loss4view[view] = 0
                    for i in range(total_itr_num):
                        imgs = {}
                        masks = {}
                        mk=0
                        for view in self.view_num:
                            try:
                                img, mask, index = dataloader[view].next()
                            except EOFError:
                                mk=1
                                break

                            img = img.to(self.device).permute(0, 4, 1, 2, 3).float()
                            mask = mask.to(self.device).permute(0, 4, 1, 2, 3).float()

                            imgs[view] = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
                            masks[view] = mask.reshape(-1, mask.shape[2], mask.shape[3], mask.shape[4])

                        if mk==1:
                            continue
                        if is_fuse:
                            pred_frames, _ = self.model(imgs)
                        else:
                            _,pred_frames, _,_ = self.model(imgs)
                        for view in self.view_num:
                            all_pred_frame[view].append(pred_frames[view].unsqueeze(-1))
                            all_mask[view].append(masks[view].unsqueeze(-1))
                        for view in self.test_view:
                            loss4view[view]+=self.bce_loss(pred_frames[view], masks[view])
                            loss += self.bce_loss(pred_frames[view], masks[view])

                for view in self.view_num:
                    all_pred_frame[view] = torch.cat(all_pred_frame[view],dim=-1)
                    all_mask[view] = torch.cat(all_mask[view], dim=-1)

                for view in self.test_view:
                    pixel_acc[view], dice[view], precision[view], specificity[view], recall[view] = self._calculate_overlap_metrics(all_mask[view], torch.where(nn.Sigmoid()(all_pred_frame[view]) > 0.5, 1, 0))
                if dataset_type=="Inner-val":
                    avg_dice = 0
                    for view in self.view_num:
                        avg_dice+=dice[view]
                    avg_dice/=len(self.view_num)
                    val_dice_list.append(avg_dice)

                for view in self.test_view:
                    print('------Validation Result . {dataset_type} for view{view} ------\n \
                           Loss : {loss:.4f} \
                           Pixel Acc : {pixel_acc:.4f} \
                           Dice : {dice:.4f} \
                           Precision : {pre:.4f} \
                           Specificity : {specificity:.4f} \
                           Recall : {recall:.4f}'.\
                           format(dataset_type=dataset_type ,view=view, loss=loss4view[view].item(), pixel_acc=pixel_acc[view], dice=dice[view], pre=precision[view], specificity=specificity[view], recall=recall[view]))

                    for part in range(5):
                        pred_view = all_pred_frame[view][:, part]
                        select_masks = all_mask[view][:, part]
                        _, part_dice, _, _, _ = self._calculate_overlap_metrics(select_masks, torch.where(nn.Sigmoid()(pred_view) > 0.5, 1, 0))
                        print('Part Result for view{view} . ------ {part_num} ------ . \
                           Dice : {dice:.4f} '.\
                           format(view=view, part_num=part, dice=part_dice))

            if net_root==None:
                return val_dice_list[0]

        val_dice_list = torch.tensor(val_dice_list[50:])
        if net_root!=None:
            print(f'best val epoch:{50+torch.argmax(val_dice_list)},best val dice:{torch.max(val_dice_list)}')
    def eval(self,net_path=None,is_fuse = True,raw_data = True):
        if raw_data==True:
            test_info = np.load(f'./infos/test_infos.npy', allow_pickle=True).item()
        else:
            test_info = np.load(f'./rmyy_test_align_dataset/infos_discrete_unlab.npy',allow_pickle=True).item()

        val_list = ['0_0','0_2']
        test_list = ['0_1','0_3','0_4','0_5','0_6','0_7','0_8','0_9']
        test_dataset = {}
        val_dataset = {}
        test_loader = {}
        val_loader = {}
        val_dice_list = []
        for view in self.view_num:
            if raw_data==False:
                test_dataset[view] = Aligned_Video_Seg_PAHDataset(test_info, root=None, is_train=False, data_list=test_list,
                                             set_select=self.config['train']['use_data'], view_num=[view],
                                             clip_length=self.config['train']['clip_length'],
                                             seg_parts=self.seg_parts, require_id=False)

                val_dataset[view] = Aligned_Video_Seg_PAHDataset(test_info, root=None, is_train=False,
                                                                  data_list=val_list,
                                                                  set_select=self.config['train']['use_data'],
                                                                  view_num=[view],
                                                                  clip_length=self.config['train']['clip_length'],
                                                                  seg_parts=self.seg_parts, require_id=False)
            else:
                test_dataset[view] = Test_Seg_PAHDataset(test_info, is_train=False,data_list=test_list,
                                              set_select=['rmyy'], clip_length=40,
                                              view_num=[view], seg_parts=True)
                val_dataset[view] = Test_Seg_PAHDataset(test_info, is_train=False,data_list=val_list,
                                                         set_select=['rmyy'], clip_length=40,
                                                         view_num=[view], seg_parts=True)
            test_loader[view] = DataLoader(test_dataset[view], batch_size=1,shuffle=False, num_workers=config['train']['num_workers'],drop_last=True)
            val_loader[view] = DataLoader(val_dataset[view], batch_size=1, shuffle=False,num_workers=config['train']['num_workers'], drop_last=True)


        data = torch.load(net_path, map_location=self.device)

        data['network'] = {'module.' + k: v for k, v in data['network'].items()}
        self.model.load_state_dict(data['network'], strict=True)
        self.model.eval()
        for dataset_type in ['Inner-val','Inner-test']:
            with torch.no_grad():
                dataloader = {}
                all_pred_frame = {}
                all_mask = {}
                pixel_acc, dice, precision, specificity, recall = {}, {}, {}, {}, {}
                for view in self.view_num:
                    pixel_acc[view] = 0.
                    dice[view] = 0.
                    precision[view] = 0.
                    specificity[view] = 0.
                    recall[view] = 0.
                for view in self.view_num:
                    all_pred_frame[view] = []
                    all_mask[view] = []
                for view in self.view_num:
                    if dataset_type=='Inner-val':
                        dataloader[view] = iter(val_loader[view])
                    if dataset_type=='Inner-test':
                        dataloader[view] = iter(test_loader[view])
                total_itr_num = len(dataloader[self.view_num[0]])
                loss = 0
                loss4view = {}
                for view in self.view_num:
                    loss4view[view] = 0
                for i in range(total_itr_num):
                    imgs = {}
                    masks = {}
                    mk=0
                    for view in self.view_num:
                        try:
                            img, mask, index = dataloader[view].next()
                        except EOFError:
                            mk=1
                            break

                        img = img.to(self.device).permute(0, 4, 1, 2, 3).float()
                        mask = mask.to(self.device).permute(0, 4, 1, 2, 3).float()

                        imgs[view] = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
                        masks[view] = mask.reshape(-1, mask.shape[2], mask.shape[3], mask.shape[4])

                    if mk==1:
                        continue
                    if is_fuse:
                        pred_frames, _ = self.model(imgs)
                    else:
                        _,pred_frames, _,_ = self.model(imgs)
                    for view in self.view_num:
                        all_pred_frame[view].append(pred_frames[view].unsqueeze(-1))
                        all_mask[view].append(masks[view].unsqueeze(-1))
                    for view in self.test_view:
                        loss4view[view]+=self.bce_loss(pred_frames[view], masks[view])
                        loss += self.bce_loss(pred_frames[view], masks[view])

            for view in self.view_num:
                all_pred_frame[view] = torch.cat(all_pred_frame[view],dim=-1) #40 5 112 112 n
                all_mask[view] = torch.cat(all_mask[view], dim=-1)

            for view in self.test_view:
                pixel_acc[view], dice[view], precision[view], specificity[view], recall[view] = self._calculate_overlap_metrics(all_mask[view], torch.where(nn.Sigmoid()(all_pred_frame[view]) > 0.5, 1, 0))
            if dataset_type=="Inner-val":
                avg_dice = 0
                for view in self.view_num:
                    avg_dice+=dice[view]
                avg_dice/=len(self.view_num)
                val_dice_list.append(avg_dice)

            for view in self.test_view:
                print('------Validation Result . {dataset_type} for view{view} ------\n \
                       Loss : {loss:.4f} \
                       Pixel Acc : {pixel_acc:.4f} \
                       Dice : {dice:.4f} \
                       Precision : {pre:.4f} \
                       Specificity : {specificity:.4f} \
                       Recall : {recall:.4f}'.\
                       format(dataset_type=dataset_type ,view=view, loss=loss4view[view].item(), pixel_acc=pixel_acc[view], dice=dice[view], pre=precision[view], specificity=specificity[view], recall=recall[view]))

                for part in range(5):
                    pred_view = all_pred_frame[view][:, part]
                    select_masks = all_mask[view][:, part]
                    _, part_dice, _, _, _ = self._calculate_overlap_metrics(select_masks, torch.where(nn.Sigmoid()(pred_view) > 0.5, 1, 0))
                    print('Part Result for view{view} . ------ {part_num} ------ . \
                       Dice : {dice:.4f} '.\
                       format(view=view, part_num=part, dice=part_dice))


    def test_visualize(self,method_name,net_path = './result/rmyy_latefusion_bbloss/net_00088.pth'):

        test_info = np.load(f'./infos/test_infos.npy', allow_pickle=True).item()
        test_list = ['0_0', '0_1','0_2', '0_3', '0_4', '0_5', '0_6', '0_7', '0_8', '0_9']
        test_dataset = {}
        test_loader = {}
        for view in self.view_num:
            test_dataset[view] = Test_Seg_PAHDataset(test_info, is_train=False, data_list=test_list,
                                                     set_select=['rmyy'], clip_length=40,
                                                     view_num=[view], seg_parts=True,require_id=True)
            test_loader[view] = DataLoader(test_dataset[view], batch_size=1, shuffle=False,
                                           num_workers=config['train']['num_workers'], drop_last=True)
        if net_path != None:
            data = torch.load(net_path, map_location=self.device)
            data['network'] = {'module.' + k: v for k, v in data['network'].items()}
            self.model.load_state_dict(data['network'], strict=True)
        self.model.eval()
        if os.path.exists(f'./visualze_for_ppt/{method_name}') == False:
            os.mkdir(f'./visualze_for_ppt/{method_name}')
        if os.path.exists(f'./visualze_for_ppt/{method_name}/192_data') == False:
            os.mkdir(f'./visualze_for_ppt/{method_name}/192_data')
        if os.path.exists(f'./visualze_for_ppt/{method_name}/img') == False:
            os.mkdir(f'./visualze_for_ppt/{method_name}/img')
        if os.path.exists(f'./visualze_for_ppt/{method_name}/local_mask') == False:
            os.mkdir(f'./visualze_for_ppt/{method_name}/local_mask')
        for dataset_type in ['Inner-test']:
            with torch.no_grad():
                total_itr_num = len(test_loader[self.view_num[0]])
                dataloader = {}
                all_pred_frame = {}
                all_mask = {}
                pixel_acc, dice, precision, specificity, recall = {}, {}, {}, {}, {}
                for view in self.view_num:
                    pixel_acc[view] = 0.
                    dice[view] = 0.
                    precision[view] = 0.
                    specificity[view] = 0.
                    recall[view] = 0.
                for view in self.view_num:
                    all_pred_frame[view] = []
                    all_mask[view] = []
                for view in self.view_num:
                    if dataset_type == 'Inner-test':
                        dataloader[view] = iter(test_loader[view])
                loss4view = {}
                for view in self.view_num:
                    loss4view[view] = 0
                for i in range(total_itr_num):
                    imgs = {}
                    masks = {}
                    visual_pred_frames = {}
                    visual_gnd = {}
                    for view in self.view_num:
                        img, mask, _, idx = dataloader[view].next()
                        img = img.to(self.device).permute(0, 4, 1, 2, 3).float()
                        mask = mask.to(self.device).permute(0, 4, 1, 2, 3)
                        imgs[view] = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
                        masks[view] = mask.reshape(-1, mask.shape[2], mask.shape[3], mask.shape[4])
                    pred_frames,_,local_mask,_ = self.model(imgs)
                    for view in self.view_num:
                        pred_frames[view] = torch.where(nn.Sigmoid()(pred_frames[view]) > 0.5, 1, 0)
                        visual_pred_frames[view] = torch.argmax(torch.cat(
                            [0.5 + torch.zeros_like(pred_frames[view][:, 0, :, :]).unsqueeze(1), pred_frames[view]],
                            dim=1), dim=1)
                        visual_gnd[view] = torch.argmax(torch.cat(
                            [0.5 + torch.zeros_like(masks[view][:, 0, :, :]).unsqueeze(1), masks[view]],
                            dim=1), dim=1)
                    idx = idx[0]
                    # print(i)
                    for view in self.view_num:
                        # print(view)
                        for frame_i, pred in enumerate(visual_pred_frames[view]):
                            pred = pred.detach().cpu().numpy()
                            pred_shape = pred.shape
                            pred = pred.tolist()
                            for pred_i in range(pred_shape[0]):
                                for pred_j in range(pred_shape[1]):
                                    if pred[pred_i][pred_j] == 0:
                                        pred[pred_i][pred_j] = [0, 0, 0,255]
                                    elif pred[pred_i][pred_j] == 1:
                                        pred[pred_i][pred_j] = [55, 255, 254, 255]
                                    elif pred[pred_i][pred_j] == 2:#ra
                                        pred[pred_i][pred_j] = [27, 255, 46, 255]
                                    elif pred[pred_i][pred_j] == 3:#rv
                                        pred[pred_i][pred_j] = [45, 0, 251, 255]
                                    elif pred[pred_i][pred_j] == 4:#la
                                        pred[pred_i][pred_j] =  [251, 13, 15, 255]
                                    elif pred[pred_i][pred_j] == 5:#lv
                                        pred[pred_i][pred_j] = [223, 48, 236, 255]
                            pred = numpy.array(pred)
                            frame = imgs[view][frame_i].squeeze(0).detach().cpu().numpy()

                            plt.imshow(pred)

                            plt.axis('off')
                            if os.path.exists(os.path.join(f'./visualze_for_ppt/{method_name}/192_data', idx)) == False:
                                os.mkdir(os.path.join(f'./visualze_for_ppt/{method_name}/192_data', idx))
                            if os.path.exists(os.path.join(f'./visualze_for_ppt/{method_name}/192_data', idx, view)) == False:
                                os.mkdir(os.path.join(f'./visualze_for_ppt/{method_name}/192_data', idx,view))
                            plt.savefig(os.path.join(f'./visualze_for_ppt/{method_name}/192_data', idx, view, f'pred_{frame_i}.png'),dpi=600,bbox_inches='tight', pad_inches = -0.1)
                            plt.close()

                    print(f'patient {idx} pred finished')

    def seg_cycle(self, feat_out, target_region, cyc_off, chunk_size, temperature):
        feat_out_query = feat_out[:target_region]
        feat_out_query_cyc = feat_out[cyc_off:target_region]
        feat_out_key = feat_out[target_region:]

        target_strtpt = np.random.choice(target_region - (chunk_size + cyc_off) + 1)
        target_strtpt_1ht = torch.eye(target_region - (chunk_size + cyc_off) + 1)[target_strtpt] 
        target_strtpt_1ht = target_strtpt_1ht.to(self.device)
        
        query_feat = feat_out_query[target_strtpt:target_strtpt + chunk_size, ...] 

        key_size = feat_out_key.shape[0] 
        feat_size = feat_out.shape[1]

        ### distance calculation 
        dist_mat = feat_out_key.unsqueeze(1).repeat((1,chunk_size, 1)) - query_feat.unsqueeze(1).transpose(0,1).repeat(key_size, 1, 1) 
        dist_mat_sq = dist_mat.pow(2) 
        dist_mat_sq_ftsm = dist_mat_sq.sum(dim = -1)
        
        
        indices_ftsm = torch.arange(chunk_size)
        gather_indx_ftsm = torch.arange(key_size).view((key_size, 1)).repeat((1,chunk_size)) 
        gather_indx_shft_ftsm = (gather_indx_ftsm + indices_ftsm) % (key_size)
        gather_indx_shft_ftsm = gather_indx_shft_ftsm.to(self.device)
        dist_mat_sq_shft_ftsm = torch.gather(dist_mat_sq_ftsm, 0, gather_indx_shft_ftsm)[:key_size - (chunk_size + cyc_off) + 1] 
        dist_mat_sq_total_ftsm = dist_mat_sq_shft_ftsm.sum(dim=(1))   
        similarity = - dist_mat_sq_total_ftsm

        similarity_averaged = similarity / feat_size / chunk_size * temperature
        beta_raw = torch.nn.functional.softmax(similarity_averaged, dim = 0)
        beta_weights = beta_raw.unsqueeze(1).unsqueeze(1).repeat([1, chunk_size, feat_size])
        

        #### calculate weighted key features
        indices_beta = torch.arange(chunk_size).view((1, chunk_size, 1)).repeat((key_size,1, feat_size))
        gather_indx_beta = torch.arange(key_size).view((key_size, 1, 1)).repeat((1,chunk_size, feat_size))
        gather_indx_beta_shft = (gather_indx_beta + indices_beta) % (key_size)
        gather_indx_beta_shft = gather_indx_beta_shft.to(self.device)
        feat_out_key_beta = torch.gather(feat_out_key.unsqueeze(1).repeat(1, chunk_size, 1), 0, gather_indx_beta_shft)[cyc_off:key_size - chunk_size + 1] 

        weighted_features = beta_weights * feat_out_key_beta 
        weighted_features_averaged = weighted_features.sum(dim=0)


        #### calculate sim of query feats
        q_dist_mat = feat_out_query_cyc.unsqueeze(1).repeat((1,chunk_size, 1)) - weighted_features_averaged.unsqueeze(1).transpose(0,1).repeat((target_region - cyc_off), 1, 1)
        q_dist_mat_sq = q_dist_mat.pow(2)
        q_dist_mat_sq_ftsm = q_dist_mat_sq.sum(dim = -1)

        indices_query_ftsm = torch.arange(chunk_size)
        gather_indx_query_ftsm = torch.arange(target_region - cyc_off).view((target_region - cyc_off, 1)).repeat((1,chunk_size))
        gather_indx_query_shft_ftsm = (gather_indx_query_ftsm + indices_query_ftsm) % (target_region - cyc_off)
        gather_indx_query_shft_ftsm = gather_indx_query_shft_ftsm.to(self.device)
        q_dist_mat_sq_shft_ftsm = torch.gather(q_dist_mat_sq_ftsm, 0, gather_indx_query_shft_ftsm)[:(target_region - cyc_off) - chunk_size + 1]
        
        
        q_dist_mat_sq_total_ftsm = q_dist_mat_sq_shft_ftsm.sum(dim=(1))
        q_similarity = - q_dist_mat_sq_total_ftsm

        q_similarity_averaged = q_similarity / feat_size / chunk_size * temperature

        frm_prd = torch.argmax(q_similarity_averaged)
        frm_lb = torch.argmax(target_strtpt_1ht)


        loss_cyc_raw = torch.nn.functional.binary_cross_entropy_with_logits(q_similarity_averaged, target_strtpt_1ht)
        
        return loss_cyc_raw

    def dense_seg_cycle(self, feat_out, target_region, cyc_off, chunk_size, temperature,soft_label=False,is_overlap=True):
        feat_out_query = feat_out[:target_region]
        feat_out_query_cyc = feat_out[cyc_off:target_region]
        feat_out_key = feat_out[target_region:]
        cyc_loss = 0

        if is_overlap:
            step_size = 1
        else:
            step_size = chunk_size
        for target_strtpt in range(0,target_region - (chunk_size + cyc_off) + 1,step_size):
            target_strtpt_1ht = torch.eye(target_region - (chunk_size + cyc_off) + 1)[target_strtpt]
            target_strtpt_1ht = target_strtpt_1ht.to(self.device)

            query_feat = feat_out_query[target_strtpt:target_strtpt + chunk_size, ...]

            key_size = feat_out_key.shape[0]
            feat_size = feat_out.shape[1]

            ### distance calculation
            dist_mat = feat_out_key.unsqueeze(1).repeat((1, chunk_size, 1)) - query_feat.unsqueeze(1).transpose(0,
                                                                                                                1).repeat(
                key_size, 1, 1)
            dist_mat_sq = dist_mat.pow(2)
            dist_mat_sq_ftsm = dist_mat_sq.sum(dim=-1)

            indices_ftsm = torch.arange(chunk_size)
            gather_indx_ftsm = torch.arange(key_size).view((key_size, 1)).repeat((1, chunk_size))
            gather_indx_shft_ftsm = (gather_indx_ftsm + indices_ftsm) % (key_size)
            gather_indx_shft_ftsm = gather_indx_shft_ftsm.to(self.device)
            dist_mat_sq_shft_ftsm = torch.gather(dist_mat_sq_ftsm, 0, gather_indx_shft_ftsm)[
                                    :key_size - (chunk_size + cyc_off) + 1]
            dist_mat_sq_total_ftsm = dist_mat_sq_shft_ftsm.sum(dim=(1))
            similarity = - dist_mat_sq_total_ftsm

            similarity_averaged = similarity / feat_size / chunk_size * temperature
            beta_raw = torch.nn.functional.softmax(similarity_averaged, dim=0)
            beta_weights = beta_raw.unsqueeze(1).unsqueeze(1).repeat([1, chunk_size, feat_size])

            #### calculate weighted key features
            indices_beta = torch.arange(chunk_size).view((1, chunk_size, 1)).repeat((key_size, 1, feat_size))
            gather_indx_beta = torch.arange(key_size).view((key_size, 1, 1)).repeat((1, chunk_size, feat_size))
            gather_indx_beta_shft = (gather_indx_beta + indices_beta) % (key_size)
            gather_indx_beta_shft = gather_indx_beta_shft.to(self.device)
            feat_out_key_beta = torch.gather(feat_out_key.unsqueeze(1).repeat(1, chunk_size, 1), 0, gather_indx_beta_shft)[
                                cyc_off:key_size - chunk_size + 1]

            weighted_features = beta_weights * feat_out_key_beta
            weighted_features_averaged = weighted_features.sum(dim=0)

            #### calculate sim of query feats
            q_dist_mat = feat_out_query_cyc.unsqueeze(1).repeat((1, chunk_size, 1)) - weighted_features_averaged.unsqueeze(
                1).transpose(0, 1).repeat((target_region - cyc_off), 1, 1)
            q_dist_mat_sq = q_dist_mat.pow(2)
            q_dist_mat_sq_ftsm = q_dist_mat_sq.sum(dim=-1)

            indices_query_ftsm = torch.arange(chunk_size)
            gather_indx_query_ftsm = torch.arange(target_region - cyc_off).view((target_region - cyc_off, 1)).repeat(
                (1, chunk_size))
            gather_indx_query_shft_ftsm = (gather_indx_query_ftsm + indices_query_ftsm) % (target_region - cyc_off)
            gather_indx_query_shft_ftsm = gather_indx_query_shft_ftsm.to(self.device)
            q_dist_mat_sq_shft_ftsm = torch.gather(q_dist_mat_sq_ftsm, 0, gather_indx_query_shft_ftsm)[
                                      :(target_region - cyc_off) - chunk_size + 1]

            q_dist_mat_sq_total_ftsm = q_dist_mat_sq_shft_ftsm.sum(dim=(1))
            q_similarity = - q_dist_mat_sq_total_ftsm

            q_similarity_averaged = q_similarity / feat_size / chunk_size * temperature
            # print(f'-------------sim shape:f{q_similarity_averaged.shape}-------------')
            frm_prd = torch.argmax(q_similarity_averaged)
            frm_lb = torch.argmax(target_strtpt_1ht)

            ##### soft label
            if soft_label:
                target_strtpt_1ht = torch.where(target_strtpt_1ht == 1, 0.8, 0.2 / (target_strtpt_1ht.shape[0] - 1))
            # print(target_strtpt_1ht)

            loss_cyc_raw = torch.nn.functional.binary_cross_entropy_with_logits(q_similarity_averaged, target_strtpt_1ht)
            cyc_loss += loss_cyc_raw
        return cyc_loss/(target_region - (chunk_size + cyc_off) + 1)

    def _calculate_overlap_metrics(self, gt, pred, eps=1e-5):
        output = pred.reshape(-1, )
        target = gt.reshape(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        precision = (tp) / (tp + fp + eps)
        recall = (tp) / (tp + fn + eps)
        specificity = (tn) / (tn + fp + eps)

        return pixel_acc, dice, precision, specificity, recall

    def adjust_learning_rate(self, optimizer, epoch, args):
        """Decay the learning rate based on schedule"""
        cur_lr = self.config['net']['opt']['lr'] * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

    def load(self):
        model_path = self.config['train']['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None


        self.latest_eopch = latest_epoch+1
        if latest_epoch is not None:
            net_path = os.path.join(
                model_path, 'net_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))

            if self.local_rank == self.config['train']['enable_GPUs_id'][0]:
                print('Loading model from {}...'.format(net_path))

            data = torch.load(net_path, map_location=self.device)
            # data['network'] = {k.replace('module.', ''): v for k, v in data['network'].items() if k.replace('module.', '') in self.model.state_dict()}
            self.model.load_state_dict(data['network'])

            data = torch.load(opt_path, map_location=self.device)
            self.optimizer.load_state_dict(data['optimizer'])

        else:
            if self.local_rank == config['train']['enable_GPUs_id'][0] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    def save(self, it):
        if self.local_rank == self.config['train']['enable_GPUs_id'][0]:
            net_path = os.path.join(
                self.config['train']['save_dir'], 'net_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(
                self.config['train']['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(net_path))
            if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.utils.data.distributed.DistributedSampler):
                network = self.model.module
            else:
                network = self.model.module
            torch.save({'network': network.state_dict()}, net_path)
            # torch.save({'epoch': self.epoch,
            #             'optimizer': self.optimizer.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['train']['save_dir'], 'latest.ckpt')))
    # add summary
    def add_summary(self, writer, name, val, count, sum_type = 'scalar'):
        def writer_in(writer, name, val, sum_type, count):
            if sum_type == 'scalar':
                writer.add_scalar(name, val, count)
            elif sum_type == 'image':
                writer.add_image(name, val, count)
            elif sum_type == 'histogram':
                writer.add_histogram(name, val, count)

        writer_in(writer, name, val, sum_type, count)

def main(rank, config):

    if 'local_rank' not in config:
        config['train']['local_rank'] = config['train']['global_rank'] = rank

    if torch.cuda.is_available(): 
        config['train']['device'] = torch.device("cuda:{}".format(config['train']['local_rank']))
    else: 
        config['train']['device'] = 'cpu'


    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required = True)
    args = parser.parse_args()

    Train_ = Trainer(config)

    if args.mode == 'train':
        Train_.train(is_backbone=False,is_cycle=True)
    if args.mode == 'val':
        Train_.eval(net_path='./path/to/ckpt',is_fuse = True,raw_data = True)
    if args.mode == 'visual':
        Train_.test_visualize(method_name='GLfusion',net_path='./path/to/ckpt')
if __name__ == "__main__":
    config = {
                "train":{
                        "cudnn": True,
                        "enable_GPUs_id": [3],
                        "device_ids":[3,2,1,0],
                        "batch_size": 8,
                        "num_workers": 8,
                        "num_epochs": 100,
                        "clip_length":40,
                        "view_num": ['1','3','4'],#'1','2','3','4'
                        "test_view":['1','3','4'],#'1','3','4'
                        "dense_cyc": False,
                        "seg_parts": True,
                        "record_params": False,
                        "save_dir": './result/ckpt',
                        "log_dir": './result/log_info/log_01',
                        'data_list_path':'./',
                        "use_data":['rmyy'],#'gy','rmyy','szfw'
                        "alpha":0.8,
                        "is_load":False
                        # "device":torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        },

                "net":  {
                        "opt"  :{
                                "opt_name": 'Adam',
                                "lr": 3e-4,
                                "step_size": 50,
                                "params": (0.9, 0.999),
                                "weight_decay": 1e-5,
                                },
                        'opt_loss_weight':{
                                "opt_name": 'Adam',
                                "lr": 1e-3,
                                "step_size": 50,
                                "params": (0.9, 0.999),
                                "weight_decay": 1e-5,
                        }
                        },

              }

    # setting distributed configurations
    config['train']['world_size'] = len(config['train']['enable_GPUs_id'])
    config['train']['init_method'] = f"tcp://{get_master_ip()}:{23455}"
    config['train']['distributed'] = True if config['train']['world_size'] > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1" and config['train']['distributed']:
        # manually launch distributed processes 
        torch.multiprocessing.spawn(main, nprocs=config['train']['world_size'], args=(config,))
    else:
        # multiple processes have been launched by openmpi
        config['train']['local_rank'] = config['train']['enable_GPUs_id'][0]
        config['train']['global_rank'] = config['train']['enable_GPUs_id'][0]

    main(config['train']['local_rank'], config)
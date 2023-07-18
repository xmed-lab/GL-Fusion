import glob
import os
import sys
import argparse
import time
import math
import random
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorboardX import SummaryWriter

from torch.utils.data import Dataset

from .segmentation import deeplabv3_resnet50_iekd
from models.unet import U_Net, R2AttU_Net, AttU_Net, R2U_Net
import copy

class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """
    def __init__(self, n_embd, n_head, attn_pdrop=0, resid_pdrop=0):
        super(SelfAttention,self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        #B:batch_size T:channel C:feature_dim
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # print(B,' ',T,' ',C)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class transformer(nn.Module):

    def __init__(self, n_embd, n_head, view_num,attn_pdrop=0, resid_pdrop=0):
        super(transformer,self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.attn = SelfAttention(n_embd=n_embd, n_head=n_head,attn_pdrop = attn_pdrop,resid_pdrop = resid_pdrop)
        self.view_num = view_num

    def forward(self, x):
        #batch_size * channel * h * w * len(view_num)
        batch_size,channel,h,w = x[self.view_num[0]].shape
        x = torch.cat([x[view] for view in self.view_num],dim=1)
        x = x.reshape([batch_size,channel*len(self.view_num),-1])
        x = x+self.attn(x)
        x = x.reshape([batch_size,channel,h,w,len(self.view_num)])
        y = {}
        for i,view in enumerate(self.view_num):
            y[view] = x[...,i]
        return y

class concat_fusion(nn.Module):

    def __init__(self, dims,view_num):
        super(concat_fusion,self).__init__()

        # key, query, value projections for all heads
        self.dims = dims
        self.view_num = view_num
        self.fc = nn.Linear(self.dims*len(self.view_num),self.dims)

    def forward(self, x):
        #batch_size * channel * h * w * len(view_num)
        batch_size,channel,h,w = x[self.view_num[0]].shape
        for view in self.view_num:
            x[view] = x[view].reshape((batch_size,channel,h*w))
        x = torch.cat([x[view] for view in self.view_num],dim=2)
        x = self.fc(x)
        x = x.reshape([batch_size,channel,h,w])

        return x

class MLP(nn.Module):

    def __init__(self, dims):
        super(MLP,self).__init__()

        self.dims = dims
        self.fc = nn.Linear(self.dims,self.dims)
        self.relu = nn.ReLU()
    def forward(self, x):
        batch_size,channel,h,w = x.shape

        x = x.reshape((batch_size,channel,h*w))
        x = self.fc(x)
        x = self.relu(x)
        x = x.reshape([batch_size,channel,h,w])

        return x

class Mutiview_Model(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(Mutiview_Model,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block = nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.decouple = nn.ModuleDict()

        for view in view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            # early fusion
            # self.init_block[view] = copy.deepcopy(nn.Sequential(
            #     nn.Conv2d(in_channels=len(self.view_num),out_channels=self.network.backbone['conv1'].out_channels,kernel_size = self.network.backbone['conv1'].kernel_size,padding=self.network.backbone['conv1'].padding,stride=self.network.backbone['conv1'].stride),
            #     self.network.backbone['bn1'],
            #     self.network.backbone['relu'],
            #     self.network.backbone['maxpool']
            # ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            # self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
            #                                             self.outchannel_list[view],
            #                                             kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1
            #MLP解耦器
            # self.decouple[view] = MLP(dims=21*21)

        self.attn0 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
        self.attn1 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
        self.attn2 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
        self.attn3 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
        self.attn4 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        input_shape = x[self.view_num[0]].shape[-2:]
        f0 = {}
        f1 = {}
        f2 = {}
        f3 = {}
        f4 = {}
        f4_decouple={}
        f4_out = {}
        mask = {}

        # x_early_fusion = torch.cat([x[view] for view in self.view_num],dim=1)
        # for view in self.view_num:
        #     f0[view] = self.init_block[view](x_early_fusion)
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view]) #2 * 64 * 41 * 41
        # f0 = self.attn0(f0)
        for view in self.view_num:
            f1[view] = self.layer1[view](f0[view]) #2 * 256 * 41 * 41
        # f1 = self.attn1(f1)
        for view in self.view_num:
            f2[view] = self.layer2[view](f1[view]) #2 * 512 * 21 * 21
        # f2 = self.attn2(f2)
        for view in self.view_num:
            f3[view] = self.layer3[view](f2[view]) #2 * 1024 * 21 * 21
        # f3 = self.attn3(f3)
        for view in self.view_num:
            f4[view] = self.layer4[view](f3[view]) #2 * 2048 * 21 * 21
        # # late fusion 试试concat
        # # f4 = self.attn4(f4)
        # for view in self.view_num:
        #     f4_out[view] = copy.deepcopy(f4[view].detach())
        # f4_fusion = self.mlp(f4)

        # for view in self.view_num:
        #     mask[view] = self.classifier[view](f4[view])
        #     # mask[view] = self.classifier['1'](f4[view])
        #     mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        # for view in self.test_view:
        #     f4_decouple[view] = self.decouple[view](f4)
        #     mask[view] = self.classifier[view](f4_decouple[view])
        #     mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        for view in self.test_view:
            mask[view] = self.classifier[view](f4[view])
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)
        return mask,f4


# class model1(nn.Module):
#     def __init__(self,view_num,test_view=['1','2','3','4']):
#         super(model1,self).__init__()
#         self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
#         self.view_num = view_num
#         self.test_view = test_view
#         self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
#         self.init_block = nn.ModuleDict()
#         self.layer1 = nn.ModuleDict()
#         self.layer2 = nn.ModuleDict()
#         self.layer3 = nn.ModuleDict()
#         self.layer4 = nn.ModuleDict()
#         self.classifier = nn.ModuleDict()
#         self.decouple = nn.ModuleDict()
#
#         for view in view_num:
#             self.init_block[view]=copy.deepcopy(nn.Sequential(
#                 self.network.backbone['conv1'],
#                 self.network.backbone['bn1'],
#                 self.network.backbone['relu'],
#                 self.network.backbone['maxpool']
#             ))
#             self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
#             self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
#             self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
#             self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
#             self.classifier[view]=copy.deepcopy(self.network.classifier)
#
#             #全种类输出
#             self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
#                                                         5,
#                                                         kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1
#             #MLP解耦器
#             self.decouple[view] = MLP(dims=21*21)
#
#         self.attn0 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
#         self.attn1 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
#         self.attn2 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
#         self.attn3 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
#         self.attn4 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
#
#     def forward(self,x):
#         """
#         输入为字典,view -> img
#         返回为字典,view -> mask
#         """
#         input_shape = x[self.view_num[0]].shape[-2:]
#         f0 = {}
#         f1 = {}
#         f2 = {}
#         f3 = {}
#         f4 = {}
#         mask = {}
#
#         for view in self.view_num:
#             f0[view] = self.init_block['1'](x[view])
#
#         for view in self.view_num:
#             f1[view] = self.layer1['1'](f0[view])
#
#         for view in self.view_num:
#             f2[view] = self.layer2['1'](f1[view])
#
#         for view in self.view_num:
#             f3[view] = self.layer3['1'](f2[view])
#
#         for view in self.view_num:
#             f4[view] = self.layer4['1'](f3[view])
#
#
#         for view in self.test_view:
#             mask[view] = self.classifier[view](f4[view])
#             mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)
#         return mask,f4


# class model2(nn.Module):
#     def __init__(self,view_num,test_view=['1','2','3','4']):
#         super(model2,self).__init__()
#         self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
#         self.view_num = view_num
#         self.test_view = test_view
#         self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
#         self.init_block = nn.ModuleDict()
#         self.layer1 = nn.ModuleDict()
#         self.layer2 = nn.ModuleDict()
#         self.layer3 = nn.ModuleDict()
#         self.layer4 = nn.ModuleDict()
#         self.classifier = nn.ModuleDict()
#         self.decouple = nn.ModuleDict()
#         for view in view_num:
#             # early fusion
#             self.init_block[view] = copy.deepcopy(nn.Sequential(
#                 nn.Conv2d(in_channels=len(self.view_num),out_channels=self.network.backbone['conv1'].out_channels,kernel_size = self.network.backbone['conv1'].kernel_size,padding=self.network.backbone['conv1'].padding,stride=self.network.backbone['conv1'].stride),
#                 self.network.backbone['bn1'],
#                 self.network.backbone['relu'],
#                 self.network.backbone['maxpool']
#             ))
#             self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
#             self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
#             self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
#             self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
#             self.classifier[view]=copy.deepcopy(self.network.classifier)
#             #全种类输出
#             self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
#                                                         5,
#                                                         kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1
#             #MLP解耦器
#             self.decouple[view] = MLP(dims=21*21)
#
#         self.attn0 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
#         self.attn1 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
#         self.attn2 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
#         self.attn3 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
#         self.attn4 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
#     def forward(self,x):
#         """
#         输入为字典,view -> img
#         返回为字典,view -> mask
#         """
#         input_shape = x[self.view_num[0]].shape[-2:]
#         f4_decouple={}
#         mask = {}
#
#         x_early_fusion = torch.cat([x[view] for view in self.view_num],dim=1)
#
#         f0 = self.init_block['1'](x_early_fusion)
#         f1 = self.layer1['1'](f0)
#         f2 = self.layer2['1'](f1)
#         f3 = self.layer3['1'](f2)
#         f4 = self.layer4['1'](f3)
#
#         for view in self.test_view:
#             f4_decouple[view] = self.decouple[view](f4)
#             mask[view] = self.classifier[view](f4_decouple[view])
#             mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)
#
#         return mask,f4_decouple


class model3(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model3,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block = nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.decouple = nn.ModuleDict()

        for view in view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1



        self.attn0 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
        self.attn1 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
        self.attn2 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
        self.attn3 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
        self.attn4 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)

        # self.mlp = concat_fusion(dims=21*21,view_num=self.view_num)


    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        input_shape = x[self.view_num[0]].shape[-2:]
        f0 = {}
        f1 = {}
        f2 = {}
        f3 = {}
        f4 = {}
        f4_decouple={}
        f4_out = {}
        mask = {}

        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
        # f0 = self.attn0(f0)
        for view in self.view_num:
            f1[view] = self.layer1[view](f0[view])
        # f1 = self.attn1(f1)
        for view in self.view_num:
            f2[view] = self.layer2[view](f1[view])
        # f2 = self.attn2(f2)
        for view in self.view_num:
            f3[view] = self.layer3[view](f2[view])
        # f3 = self.attn3(f3)
        for view in self.view_num:
            f4[view] = self.layer4[view](f3[view])
        f4 = self.attn4(f4)

        #test for balance
        for view in self.test_view:
            mask[view] = self.classifier[view](f4[view])
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)
        return mask,f4

class model6(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model6,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)

        self.init_block=copy.deepcopy(nn.Sequential(
            self.network.backbone['conv1'],
            self.network.backbone['bn1'],
            self.network.backbone['relu'],
            self.network.backbone['maxpool']
        ))
        self.layer1=copy.deepcopy(self.network.backbone['layer1'])
        self.layer2=copy.deepcopy(self.network.backbone['layer2'])
        self.layer3=copy.deepcopy(self.network.backbone['layer3'])
        self.layer4=copy.deepcopy(self.network.backbone['layer4'])
        self.classifier=copy.deepcopy(self.network.classifier)
        #全种类输出
        self.classifier[-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                    5,
                                                    kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1


    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        input_shape = x.shape[-2:]

        f0 = self.init_block(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        mask = self.classifier(f4)
        mask = F.interpolate(mask, size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4

class model7(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model7,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)

        self.init_block=nn.Sequential(
            network.backbone['conv1'],
            network.backbone['bn1'],
            network.backbone['relu'],
            network.backbone['maxpool']
        )
        self.layer1=network.backbone['layer1']
        self.layer2=network.backbone['layer2']
        self.layer3=network.backbone['layer3']
        self.layer4=network.backbone['layer4']
        self.classifier=network.classifier
        #全种类输出
        self.classifier[-1] = torch.nn.Conv2d(network.classifier[-1].in_channels,
                                                    5,
                                                    kernel_size=network.classifier[-1].kernel_size)  # change number of outputs to 1


    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block(x[view])
            f1[view] = self.layer1(f0[view])
            f2[view] = self.layer2(f1[view])
            f3[view] = self.layer3(f2[view])
            f4[view] = self.layer4(f3[view])

            mask[view] = self.classifier(f4[view])
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,None,f4,f4
class model_CPS(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model_CPS,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)

        self.init_block=nn.Sequential(
            network.backbone['conv1'],
            network.backbone['bn1'],
            network.backbone['relu'],
            network.backbone['maxpool']
        )
        self.layer1=network.backbone['layer1']
        self.layer2=network.backbone['layer2']
        self.layer3=network.backbone['layer3']
        self.layer4=network.backbone['layer4']
        self.classifier=network.classifier
        #全种类输出
        self.classifier[-1] = torch.nn.Conv2d(network.classifier[-1].in_channels,
                                                    5,
                                                    kernel_size=network.classifier[-1].kernel_size)  # change number of outputs to 1
        self.init_block2 = copy.deepcopy(nn.Sequential(
            network.backbone['conv1'],
            network.backbone['bn1'],
            network.backbone['relu'],
            network.backbone['maxpool']
        ))
        self.layer1_2 = copy.deepcopy(network.backbone['layer1'])
        self.layer2_2 = copy.deepcopy(network.backbone['layer2'])
        self.layer3_2 = copy.deepcopy(network.backbone['layer3'])
        self.layer4_2 = copy.deepcopy(network.backbone['layer4'])
        self.classifier_2 = copy.deepcopy(network.classifier)
        # 全种类输出
        self.classifier[-1] = torch.nn.Conv2d(network.classifier[-1].in_channels,
                                              5,
                                              kernel_size=network.classifier[
                                                  -1].kernel_size)  # change number of outputs to 1

        self.classifier_2[-1] = torch.nn.Conv2d(self.classifier_2[-1].in_channels,
                                              5,
                                              kernel_size=self.classifier_2[
                                                  -1].kernel_size)  # cha
    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        mask={}
        f0_2 = {}
        f1_2 = {}
        f2_2 = {}
        f3_2 = {}
        f4_2 = {}
        mask_2 = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block(x[view])
            f1[view] = self.layer1(f0[view])
            f2[view] = self.layer2(f1[view])
            f3[view] = self.layer3(f2[view])
            f4[view] = self.layer4(f3[view])

            mask[view] = self.classifier(f4[view])
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            f0_2[view] = self.init_block2(x[view])
            f1_2[view] = self.layer1_2(f0_2[view])
            f2_2[view] = self.layer2_2(f1_2[view])
            f3_2[view] = self.layer3_2(f2_2[view])
            f4_2[view] = self.layer4_2(f3_2[view])

            mask_2[view] = self.classifier_2(f4_2[view])
            mask_2[view] = F.interpolate(mask_2[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_2,f4,f4

class model8(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model8,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block = nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()

        for view in view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))

            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier=copy.deepcopy(self.network.classifier)
            # self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
            #                                             self.outchannel_list[view],
            #                                             kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            #全种类输出
            self.classifier[-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.attn0 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
        self.attn1 = transformer(n_embd=41*41, n_head=1,view_num=self.view_num)
        self.attn2 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
        self.attn3 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
        self.attn4 = transformer(n_embd=21*21, n_head=3,view_num=self.view_num)
    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        input_shape = x[self.view_num[0]].shape[-2:]
        f0 = {}
        f1 = {}
        f2 = {}
        f3 = {}
        f4 = {}
        mask = {}


        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])

        for view in self.view_num:
            f1[view] = self.layer1[view](f0[view])

        for view in self.view_num:
            f2[view] = self.layer2[view](f1[view])

        for view in self.view_num:
            f3[view] = self.layer3[view](f2[view])

        for view in self.view_num:
            f4[view] = self.layer4[view](f3[view])

        f4 = self.attn4(f4)


        for view in self.view_num:
            mask[view] = self.classifier(f4[view])
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4


class model12(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model12,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)

        self.init_block=copy.deepcopy(nn.Sequential(
            self.network.backbone['conv1'],
            self.network.backbone['bn1'],
            self.network.backbone['relu'],
            self.network.backbone['maxpool']
        ))
        self.layer1=copy.deepcopy(self.network.backbone['layer1'])
        self.layer2=copy.deepcopy(self.network.backbone['layer2'])
        self.layer3=copy.deepcopy(self.network.backbone['layer3'])
        self.layer4=copy.deepcopy(self.network.backbone['layer4'])
        self.classifier=copy.deepcopy(self.network.classifier)
        #全种类输出
        self.classifier[-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                    5,
                                                    kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1
        self.attn = transformer(n_embd=21 * 21, n_head=3, view_num=self.view_num)

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block(x[view])
            f1[view] = self.layer1(f0[view])
            f2[view] = self.layer2(f1[view])
            f3[view] = self.layer3(f2[view])
            f4[view] = self.layer4(f3[view])
        f4 = self.attn(f4)
        for view in self.view_num:
            mask[view] = self.classifier(f4[view])
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4


class TPAVIModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot',
                 dimension=3, bn_layer=True):
        """
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(TPAVIModule, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        ## add align channel
        self.align_channel = nn.Linear(128, in_channels)
        self.norm_layer = nn.LayerNorm(in_channels)

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x, audio=None):
        """
        args:
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
            audio: (N, T, C)
        """
        # bs * 2048 * 3 * h * w
        # x : batch_size,channel,modal_num,h,w
        audio_temp = 0
        batch_size, C = x.size(0), x.size(1)
        if audio is not None:
            # print('==> audio.shape', audio.shape)
            H, W = x.shape[-2], x.shape[-1]
            audio_temp = self.align_channel(audio)  # [bs, T, C]
            audio = audio_temp.permute(0, 2, 1)  # [bs, C, T]
            audio = audio.unsqueeze(-1).unsqueeze(-1)  # [bs, C, T, 1, 1]
            audio = audio.repeat(1, 1, 1, H, W)  # [bs, C, T, H, W]
        else:
            audio = x

        # (N, C, THW)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [bs, C, THW]
        # print('g_x.shape', g_x.shape)
        # g_x = x.view(batch_size, C, -1)  # [bs, C, THW]
        g_x = g_x.permute(0, 2, 1)  # [bs, THW, C]

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = audio.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # [bs, C', THW]
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, -1)  # [bs, C', THW]
            theta_x = theta_x.permute(0, 2, 1)  # [bs, THW, C']
            f = torch.matmul(theta_x, phi_x)  # [bs, THW, THW]

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N  # [bs, THW, THW]

        y = torch.matmul(f_div_C, g_x)  # [bs, THW, C]

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()  # [bs, C, THW]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # [bs, C', T, H, W]

        W_y = self.W_z(y)  # [bs, C, T, H, W]
        # residual connection
        z = W_y + x  # # [bs, C, T, H, W]

        # add LayerNorm
        z = z.permute(0, 2, 3, 4, 1)  # [bs, T, H, W, C]
        z = self.norm_layer(z)
        z = z.permute(0, 4, 1, 2, 3)  # [bs, C, T, H, W]

        return z, audio_temp

class model18(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model18,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)

        self.init_block=copy.deepcopy(nn.Sequential(
            self.network.backbone['conv1'],
            self.network.backbone['bn1'],
            self.network.backbone['relu'],
            self.network.backbone['maxpool']
        ))
        self.layer1=copy.deepcopy(self.network.backbone['layer1'])
        self.layer2=copy.deepcopy(self.network.backbone['layer2'])
        self.layer3=copy.deepcopy(self.network.backbone['layer3'])
        self.layer4=copy.deepcopy(self.network.backbone['layer4'])
        self.classifier=copy.deepcopy(self.network.classifier)
        #全种类输出
        self.classifier[-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                    5,
                                                    kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block(x[view])
            f1[view] = self.layer1(f0[view])
            f2[view] = self.layer2(f1[view])
            f3[view] = self.layer3(f2[view])
            f4[view] = self.layer4(f3[view])
        concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num]
        concat_featuer_map = torch.cat(concat_featuer_map, dim=2)
        conv_feat, _ = self.non_local(concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_fusion[view] = conv_feat[:, :,i, :, :]
        for view in self.view_num:
            mask[view] = self.classifier(f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4


class model19(nn.Module):
    def __init__(self,view_num,local_attn=False,test_view=['1','2','3','4']):
        super(model19,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.local_attn = local_attn
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')
        # self.local_att = TPAVIModule(in_channels=2048, mode='dot')
    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        mask_bb={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w

        concat_featuer_map = torch.cat(concat_featuer_map, dim=2)
        conv_feat, _ = self.non_local(concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_fusion[view] = conv_feat[:, :, i, :, :]
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            mask_bb[view] = self.classifier[view](f4[view].contiguous())
            mask_bb[view] = F.interpolate(mask_bb[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_bb,f4,f4_fusion

class MLP_fusion(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(MLP_fusion,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            # self.fc[view] = nn.Linear(2048 * len(self.view_num),2048)
            self.fc[view] = nn.Conv2d(kernel_size=1,in_channels=2048 * len(self.view_num), out_channels=2048)
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        bs,channel,h,w = f4[self.view_num[0]].shape
        concat_featuer_map = [f4[view] for view in self.view_num] #bs * 2048  * h * w
        concat_featuer_map = torch.cat(concat_featuer_map, dim=1)
        for i,view in enumerate(self.view_num):
            f4_fusion[view] = self.fc[view](concat_featuer_map)
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4,None,None

class model20(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model20,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)

        self.init_block = nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local0 = TPAVIModule(in_channels=64, mode='dot')
        self.non_local1= TPAVIModule(in_channels=256, mode='dot')
        self.non_local2 = TPAVIModule(in_channels=512, mode='dot')
        self.non_local3 = TPAVIModule(in_channels=1024, mode='dot')
        self.non_local4 = TPAVIModule(in_channels=2048, mode='dot')

    def non_local_attn(self,f,stage):
        f_fusion = {}
        non_local_block = getattr(self, f'non_local{stage}')
        concat_featuer_map = [f[view].unsqueeze(2) for view in self.view_num]
        concat_featuer_map = torch.cat(concat_featuer_map, dim=2)
        conv_feat, _ = non_local_block(concat_featuer_map)
        # for view in self.view_num:
        #     f_fusion[view] = conv_feat[:, :, int(view) - 1, :, :]
        f_fusion['1'] = conv_feat[:, :, 0, :, :]
        f_fusion['3'] = conv_feat[:, :, 1, :, :]
        f_fusion['4'] = conv_feat[:, :, 2, :, :]
        return f_fusion

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
        # f0 = self.non_local_attn(f0,0)
        for view in self.view_num:
            f1[view] = self.layer1[view](f0[view])
        f1 = self.non_local_attn(f1, 1)
        for view in self.view_num:
            f2[view] = self.layer2[view](f1[view])
        f2 = self.non_local_attn(f2, 2)
        for view in self.view_num:
            f3[view] = self.layer3[view](f2[view])
        f3 = self.non_local_attn(f3, 3)
        for view in self.view_num:
            f4[view] = self.layer4[view](f3[view])
        f4_fusion = self.non_local_attn(f4, 4)
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4

class model21(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model21,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)

        self.init_block=copy.deepcopy(nn.Sequential(
            self.network.backbone['conv1'],
            self.network.backbone['bn1'],
            self.network.backbone['relu'],
            self.network.backbone['maxpool']
        ))
        self.layer1=copy.deepcopy(self.network.backbone['layer1'])
        self.layer2=copy.deepcopy(self.network.backbone['layer2'])
        self.layer3=copy.deepcopy(self.network.backbone['layer3'])
        self.layer4=copy.deepcopy(self.network.backbone['layer4'])
        self.classifier=copy.deepcopy(self.network.classifier)
        #全种类输出
        self.classifier[-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                    5,
                                                    kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')
        self.consistent_cov = nn.ModuleDict()
        self.complentary_conv = nn.ModuleDict()
        for view in self.view_num:
            self.consistent_cov[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
            self.complentary_conv[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        consistent_feat = {}
        complentary_feat = {}
        complentary_feat_fusion = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block(x[view])
            f1[view] = self.layer1(f0[view])
            f2[view] = self.layer2(f1[view])
            f3[view] = self.layer3(f2[view])
            f4[view] = self.layer4(f3[view])
        for view in self.view_num:
            consistent_feat[view] = self.consistent_cov[view](f4[view])
            complentary_feat[view] = self.complentary_conv[view](f4[view])
        concat_featuer_map = [complentary_feat[view].unsqueeze(2) for view in self.view_num]
        concat_featuer_map = torch.cat(concat_featuer_map, dim=2)
        conv_feat, _ = self.non_local(concat_featuer_map)
        for view in self.view_num:
            complentary_feat_fusion[view] = conv_feat[:, :, int(view) - 1, :, :]
        for view in self.view_num:
            f4_fusion[view] = complentary_feat_fusion[view] + consistent_feat[view]

        for view in self.view_num:
            mask[view] = self.classifier(f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,complentary_feat_fusion,consistent_feat

class model21_for_specific_view(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model21_for_specific_view,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        # self.fusion_view = ['1','3','4']
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)

        self.init_block=copy.deepcopy(nn.Sequential(
            self.network.backbone['conv1'],
            self.network.backbone['bn1'],
            self.network.backbone['relu'],
            self.network.backbone['maxpool']
        ))
        self.layer1=copy.deepcopy(self.network.backbone['layer1'])
        self.layer2=copy.deepcopy(self.network.backbone['layer2'])
        self.layer3=copy.deepcopy(self.network.backbone['layer3'])
        self.layer4=copy.deepcopy(self.network.backbone['layer4'])
        self.classifier=copy.deepcopy(self.network.classifier)
        #全种类输出
        self.classifier[-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                    5,
                                                    kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')
        self.consistent_cov = nn.ModuleDict()
        self.complentary_conv = nn.ModuleDict()
        for view in self.view_num:
            self.consistent_cov[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
            self.complentary_conv[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
    def forward(self,x):
        """
        只融合view1 3 4，同时对view4 解耦，以提取适合融合的RV LV特征
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        consistent_feat = {}
        complentary_feat = {}
        complentary_feat_fusion = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block(x[view])
            f1[view] = self.layer1(f0[view])
            f2[view] = self.layer2(f1[view])
            f3[view] = self.layer3(f2[view])
            f4[view] = self.layer4(f3[view])
        for view in self.view_num:
            consistent_feat[view] = self.consistent_cov[view](f4[view])
            complentary_feat[view] = self.complentary_conv[view](f4[view])
        #这里只解耦view4，融合view134
        concat_featuer_map = [complentary_feat[view].unsqueeze(2) for view in ['1','3','4']]
        concat_featuer_map = torch.cat(concat_featuer_map, dim=2)
        conv_feat, _ = self.non_local(concat_featuer_map)

        complentary_feat_fusion['1'] = conv_feat[:, :, 0, :, :]
        complentary_feat_fusion['2'] = f4['2'] #view 2 不做融合
        complentary_feat_fusion['3'] = conv_feat[:, :, 1, :, :]
        complentary_feat_fusion['4'] = conv_feat[:, :, 2, :, :]

        f4_fusion['1'] = complentary_feat_fusion['1']
        f4_fusion['2'] = complentary_feat_fusion['2']
        f4_fusion['3'] = complentary_feat_fusion['3']
        f4_fusion['4'] = complentary_feat_fusion['4'] + consistent_feat['4']

        for view in self.view_num:
            mask[view] = self.classifier(f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,complentary_feat,complentary_feat_fusion,consistent_feat#投影后未融合的特征、融合后的特征、不用于融合的特征

class model21_for_specific_view_parallel(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model21_for_specific_view_parallel,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        # self.fusion_view = ['1','3','4']
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block = nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')
        self.consistent_cov = nn.ModuleDict()
        self.complentary_conv = nn.ModuleDict()
        for view in self.view_num:
            self.consistent_cov[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
            self.complentary_conv[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
    def forward(self,x):
        """
        只融合view1 3 4，同时对view4 解耦，以提取适合融合的RV LV特征
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        consistent_feat = {}
        complentary_feat = {}
        complentary_feat_fusion = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        for view in self.view_num:
            consistent_feat[view] = self.consistent_cov[view](f4[view])
            complentary_feat[view] = self.complentary_conv[view](f4[view])
        #这里只解耦view4，融合view134
        concat_featuer_map = [complentary_feat[view].unsqueeze(2) for view in ['1','3','4']]
        concat_featuer_map = torch.cat(concat_featuer_map, dim=2)
        conv_feat, _ = self.non_local(concat_featuer_map)

        complentary_feat_fusion['1'] = conv_feat[:, :, 0, :, :]
        complentary_feat_fusion['2'] = f4['2'] #view 2 不做融合
        complentary_feat_fusion['3'] = conv_feat[:, :, 1, :, :]
        complentary_feat_fusion['4'] = conv_feat[:, :, 2, :, :]

        f4_fusion['1'] = complentary_feat_fusion['1']
        f4_fusion['2'] = complentary_feat_fusion['2']
        f4_fusion['3'] = complentary_feat_fusion['3']
        f4_fusion['4'] = complentary_feat_fusion['4'] + consistent_feat['4']

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,complentary_feat,complentary_feat_fusion,consistent_feat#投影后未融合的特征、融合后的特征、不用于融合的特征

class model21_for_specific_view_parallel_alldecouple(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model21_for_specific_view_parallel_alldecouple,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        # self.fusion_view = ['1','3','4']
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block = nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')
        self.consistent_cov = nn.ModuleDict()
        self.complentary_conv = nn.ModuleDict()
        for view in self.view_num:
            self.consistent_cov[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
            self.complentary_conv[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
    def forward(self,x):
        """
        只融合view1 3 4，同时对view4 解耦，以提取适合融合的RV LV特征
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        consistent_feat = {}
        complentary_feat = {}
        complentary_feat_fusion = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        for view in self.view_num:
            consistent_feat[view] = self.consistent_cov[view](f4[view])
            complentary_feat[view] = self.complentary_conv[view](f4[view])
        #这里只解耦view4，融合view134
        concat_featuer_map = [complentary_feat[view].unsqueeze(2) for view in ['1','3','4']]
        concat_featuer_map = torch.cat(concat_featuer_map, dim=2)
        conv_feat, _ = self.non_local(concat_featuer_map)

        complentary_feat_fusion['1'] = conv_feat[:, :, 0, :, :]
        complentary_feat_fusion['2'] = f4['2'] #view 2 不做融合
        complentary_feat_fusion['3'] = conv_feat[:, :, 1, :, :]
        complentary_feat_fusion['4'] = conv_feat[:, :, 2, :, :]

        f4_fusion['1'] = complentary_feat_fusion['1'] + consistent_feat['1']
        f4_fusion['2'] = complentary_feat_fusion['2']
        f4_fusion['3'] = complentary_feat_fusion['3'] + consistent_feat['3']
        f4_fusion['4'] = complentary_feat_fusion['4'] + consistent_feat['4']

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,complentary_feat,complentary_feat_fusion,consistent_feat#投影后未融合的特征、融合后的特征、不用于融合的特征

class model21_for_specific_view_parallel_nodecouple(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model21_for_specific_view_parallel_nodecouple,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        # self.fusion_view = ['1','3','4']
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block = nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')
        self.consistent_cov = nn.ModuleDict()
        self.complentary_conv = nn.ModuleDict()
        for view in self.view_num:
            self.consistent_cov[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
            self.complentary_conv[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
    def forward(self,x):
        """
        只融合view1 3 4，同时对view4 解耦，以提取适合融合的RV LV特征
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        consistent_feat = {}
        complentary_feat = {}
        complentary_feat_fusion = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        # for view in self.view_num:
        #     consistent_feat[view] = self.consistent_cov[view](f4[view])
        #     complentary_feat[view] = self.complentary_conv[view](f4[view])
        #这里只解耦view4，融合view134
        # batch_size,channel,h,w -> batch_size,channel,1,h,w
        concat_featuer_map = [f4[view].unsqueeze(2) for view in ['1','3','4']]
        #batch_size,channel,modal_num,h,w
        concat_featuer_map = torch.cat(concat_featuer_map, dim=2)
        conv_feat, _ = self.non_local(concat_featuer_map)

        complentary_feat_fusion['1'] = conv_feat[:, :, 0, :, :]
        complentary_feat_fusion['2'] = f4['2'] #view 2 不做融合
        complentary_feat_fusion['3'] = conv_feat[:, :, 1, :, :]
        complentary_feat_fusion['4'] = conv_feat[:, :, 2, :, :]

        f4_fusion = complentary_feat_fusion

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,complentary_feat,complentary_feat_fusion,consistent_feat#投影后未融合的特征、融合后的特征、不用于融合的特征

class model21_for_specific_view_parallel_nofusion(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(model21_for_specific_view_parallel_nofusion,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        # self.fusion_view = ['1','3','4']
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block = nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')
        self.consistent_cov = nn.ModuleDict()
        self.complentary_conv = nn.ModuleDict()
        for view in self.view_num:
            self.consistent_cov[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
            self.complentary_conv[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                nn.BatchNorm2d(2048)
            )
    def forward(self,x):
        """
        只融合view1 3 4，同时对view4 解耦，以提取适合融合的RV LV特征
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        consistent_feat = {}
        complentary_feat = {}
        complentary_feat_fusion = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        for view in self.view_num:
            consistent_feat[view] = self.consistent_cov[view](f4[view])
            complentary_feat[view] = self.complentary_conv[view](f4[view])
        #这里只解耦view4，融合view134

        complentary_feat_fusion['1'] = complentary_feat['1']
        complentary_feat_fusion['2'] = f4['2'] #view 2 不做融合
        complentary_feat_fusion['3'] = complentary_feat['3']
        complentary_feat_fusion['4'] = complentary_feat['4']

        f4_fusion['1'] = complentary_feat_fusion['1'] + consistent_feat['1']
        f4_fusion['2'] = complentary_feat_fusion['2']
        f4_fusion['3'] = complentary_feat_fusion['3'] + consistent_feat['3']
        f4_fusion['4'] = complentary_feat_fusion['4'] + consistent_feat['4']

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,complentary_feat,complentary_feat_fusion,consistent_feat#投影后未融合的特征、融合后的特征、不用于融合的特征

class Global_and_Local(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4'],center_aware_weight=20):
        super(Global_and_Local,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        # self.local_attn = local_attn
        self.center_aware_weight = center_aware_weight
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.centerness = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness[view] = copy.deepcopy(self.network.classifier)
            self.centerness[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.global_attn = TPAVIModule(in_channels=2048, mode='dot')
        self.local_attn = TPAVIModule(in_channels=2048, mode='dot')

    def backbone(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
            f4_fusion[view] = copy.deepcopy(f4[view])
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_global_fusion = {}
        f4_local_fusion = {}
        f4_local = {}
        f4_fusion = {}
        mask={}
        mask_bb={}
        ctr = {}
        atten_map = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        #feature extractor
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        #M_cls
        for view in self.view_num:
            mask_bb[view] = self.classifier[view](f4[view].clone().contiguous())
            mask_bb[view] = nn.Sigmoid()(mask_bb[view])
            n, c, h, w = mask_bb[view].shape
            maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
            mask_bb[view] = maxpooling(mask_bb[view])
        #M_ctr
        for view in self.view_num:
            ctr[view] = self.centerness[view](f4[view].clone().contiguous())
            ctr[view] = nn.Sigmoid()(ctr[view])

        #get local feature
        for view in self.view_num:
            atten_map[view] = (self.center_aware_weight * mask_bb[view] * ctr[view]).sigmoid()
            f4_local[view] = f4[view].clone() * atten_map[view]

        #global fusion
        global_concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)#bs * 2048 * 3 * h * w
        global_conv_feat, _ = self.global_attn(global_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_global_fusion[view] = global_conv_feat[:, :, i, :, :]

        #local fusion
        local_concat_featuer_map = [f4_local[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        local_concat_featuer_map = torch.cat(local_concat_featuer_map, dim=2)
        local_conv_feat, _ = self.local_attn(local_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_local_fusion[view] = local_conv_feat[:, :, i, :, :]

        #fuse global feat and local feat
        for view in self.view_num:
            f4_fusion[view] = f4_global_fusion[view] + f4_local_fusion[view]

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            mask_bb[view] = self.classifier[view](f4[view].contiguous())
            mask_bb[view] = F.interpolate(mask_bb[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_bb,f4_global_fusion,f4_local_fusion


class Global_and_Local_Temporal(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4'],center_aware_weight=20):
        super(Global_and_Local_Temporal,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        # self.local_attn = local_attn
        self.center_aware_weight = center_aware_weight
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.centerness = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness[view] = copy.deepcopy(self.network.classifier)
            self.centerness[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.global_attn = TPAVIModule(in_channels=2048, mode='dot')
        self.local_attn = TPAVIModule(in_channels=2048, mode='dot')

    def backbone(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
            f4_fusion[view] = copy.deepcopy(f4[view])
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4

    def forward(self,x,is_video):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_global_fusion = {}
        f4_local_fusion = {}
        f4_local = {}
        f4_fusion = {}
        mask={}
        mask_bb={}
        ctr = {}
        atten_map = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        #feature extractor
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        #M_cls
        for view in self.view_num:
            mask_bb[view] = self.classifier[view](f4[view].clone().contiguous())
            mask_bb[view] = nn.Sigmoid()(mask_bb[view])
            n, c, h, w = mask_bb[view].shape
            maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
            mask_bb[view] = maxpooling(mask_bb[view])
        #M_ctr
        for view in self.view_num:
            ctr[view] = self.centerness[view](f4[view].clone().contiguous())
            ctr[view] = nn.Sigmoid()(ctr[view])

        #get local feature
        for view in self.view_num:
            atten_map[view] = (self.center_aware_weight * mask_bb[view] * ctr[view]).sigmoid()
            f4_local[view] = f4[view].clone() * atten_map[view]

        #global fusion
        global_concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)#bs * 2048 * 3 * h * w
        if is_video:
            global_concat_featuer_map = global_concat_featuer_map.permute(1,0,2,3,4) # c t v h w
            c, t, v, h, w = global_concat_featuer_map.shape
            global_concat_featuer_map = global_concat_featuer_map.shape(c,t*v,h,w).unsqueeze(0)# 1 c (tv) h w
        global_conv_feat, _ = self.global_attn(global_concat_featuer_map)
        if is_video:
            global_conv_feat = global_conv_feat.squeeze(0).reshape(c,t,v,h,w).permute(1,0,2,3,4)
        for i,view in enumerate(self.view_num):
            f4_global_fusion[view] = global_conv_feat[:, :, i, :, :]

        #local fusion
        local_concat_featuer_map = [f4_local[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        local_concat_featuer_map = torch.cat(local_concat_featuer_map, dim=2)
        if is_video:
            local_concat_featuer_map = local_concat_featuer_map.permute(1,0,2,3,4) # c t v h w
            c, t, v, h, w = local_concat_featuer_map.shape
            local_concat_featuer_map = local_concat_featuer_map.shape(c,t*v,h,w).unsqueeze(0)# 1 c (tv) h w

        local_conv_feat, _ = self.local_attn(local_concat_featuer_map)

        if is_video:
            local_conv_feat = local_conv_feat.squeeze(0).reshape(c,t,v,h,w).permute(1,0,2,3,4)

        for i,view in enumerate(self.view_num):
            f4_local_fusion[view] = local_conv_feat[:, :, i, :, :]

        #fuse global feat and local feat
        for view in self.view_num:
            f4_fusion[view] = f4_global_fusion[view] + f4_local_fusion[view]

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            mask_bb[view] = self.classifier[view](f4[view].contiguous())
            mask_bb[view] = F.interpolate(mask_bb[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_bb,f4_global_fusion,f4_local_fusion


class Global_only(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4'],center_aware_weight=20):
        super(Global_only,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        # self.local_attn = local_attn
        self.center_aware_weight = center_aware_weight
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.centerness = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness[view] = copy.deepcopy(self.network.classifier)
            self.centerness[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.global_attn = TPAVIModule(in_channels=2048, mode='dot')
        # self.local_attn = TPAVIModule(in_channels=2048, mode='dot')

    def backbone(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
            f4_fusion[view] = copy.deepcopy(f4[view])
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_global_fusion = {}
        f4_local_fusion = {}
        f4_local = {}
        f4_fusion = {}
        mask={}
        mask_bb={}
        ctr = {}
        atten_map = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        #feature extractor
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])

        #global fusion
        global_concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)#bs * 2048 * 3 * h * w
        global_conv_feat, _ = self.global_attn(global_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_global_fusion[view] = global_conv_feat[:, :, i, :, :]

        #fuse global feat and local feat
        for view in self.view_num:
            f4_fusion[view] = f4_global_fusion[view]

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            mask_bb[view] = self.classifier[view](f4[view].contiguous())
            mask_bb[view] = F.interpolate(mask_bb[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_bb,f4_global_fusion,None

class Local_only(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4'],center_aware_weight=20):
        super(Local_only,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        # self.local_attn = local_attn
        self.center_aware_weight = center_aware_weight
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.centerness = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness[view] = copy.deepcopy(self.network.classifier)
            self.centerness[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        # self.global_attn = TPAVIModule(in_channels=2048, mode='dot')
        self.local_attn = TPAVIModule(in_channels=2048, mode='dot')

    def backbone(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
            f4_fusion[view] = copy.deepcopy(f4[view])
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_global_fusion = {}
        f4_local_fusion = {}
        f4_local = {}
        f4_fusion = {}
        mask={}
        mask_bb={}
        ctr = {}
        atten_map = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        #feature extractor
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        #M_cls
        for view in self.view_num:
            mask_bb[view] = self.classifier[view](f4[view].clone().contiguous())
            mask_bb[view] = nn.Sigmoid()(mask_bb[view])
            n, c, h, w = mask_bb[view].shape
            maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
            mask_bb[view] = maxpooling(mask_bb[view])
        #M_ctr
        for view in self.view_num:
            ctr[view] = self.centerness[view](f4[view].clone().contiguous())
            ctr[view] = nn.Sigmoid()(ctr[view])

        #get local feature
        for view in self.view_num:
            atten_map[view] = (self.center_aware_weight * mask_bb[view] * ctr[view]).sigmoid()
            f4_local[view] = f4[view].clone() * atten_map[view]

        #global fusion
        # global_concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        # global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)
        # global_conv_feat, _ = self.global_attn(global_concat_featuer_map)
        # for i,view in enumerate(self.view_num):
        #     f4_global_fusion[view] = global_conv_feat[:, :, i, :, :]

        #local fusion
        local_concat_featuer_map = [f4_local[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        local_concat_featuer_map = torch.cat(local_concat_featuer_map, dim=2)
        local_conv_feat, _ = self.local_attn(local_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_local_fusion[view] = local_conv_feat[:, :, i, :, :]

        #fuse global feat and local feat
        for view in self.view_num:
            # f4_fusion[view] = f4_global_fusion[view] + f4_local_fusion[view]
            f4_fusion[view] = f4_local_fusion[view]

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            mask_bb[view] = self.classifier[view](f4[view].contiguous())
            mask_bb[view] = F.interpolate(mask_bb[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_bb,atten_map,f4_fusion

class early_fusion(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(early_fusion,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            # self.fc[view] = nn.Linear(2048 * len(self.view_num),2048)
            self.fc[view] = nn.Conv2d(kernel_size=1,in_channels=1 * len(self.view_num), out_channels=1)
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        concat_input = [x[view] for view in self.view_num] #bs * 2048  * h * w
        concat_input = torch.cat(concat_input, dim=1)
        for i,view in enumerate(self.view_num):
            x[view] = self.fc[view](concat_input)
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        # bs,channel,h,w = f4[self.view_num[0]].shape

        for view in self.view_num:
            mask[view] = self.classifier[view](f4[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4,None,None

class late_fusion(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4']):
        super(late_fusion,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            # self.fc[view] = nn.Linear(2048 * len(self.view_num),2048)
            self.fc[view] = nn.Conv2d(kernel_size=1,in_channels=5 * len(self.view_num), out_channels=5)
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.non_local = TPAVIModule(in_channels=2048, mode='dot')

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]

        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        # bs,channel,h,w = f4[self.view_num[0]].shape

        for view in self.view_num:
            mask[view] = self.classifier[view](f4[view].contiguous())
        concat_output = [mask[view] for view in self.view_num] #bs * 5 * h * w
        concat_output = torch.cat(concat_output, dim=1)
        for i,view in enumerate(self.view_num):
            mask[view] = self.fc[view](concat_output)
        for view in self.view_num:
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4,None,None

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class baseline_unet(nn.Module):
    def __init__(self,view_num):
        super(baseline_unet,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.Maxpool = nn.ModuleDict()
        self.Conv1 = nn.ModuleDict()
        self.Conv2 = nn.ModuleDict()
        self.Conv3 = nn.ModuleDict()
        self.Conv4 = nn.ModuleDict()
        self.Conv5 = nn.ModuleDict()
        self.Up1 = nn.ModuleDict()
        self.Up2 = nn.ModuleDict()
        self.Up3 = nn.ModuleDict()
        self.Up4 = nn.ModuleDict()
        self.Up5 = nn.ModuleDict()
        self.Up_conv1 = nn.ModuleDict()
        self.Up_conv2 = nn.ModuleDict()
        self.Up_conv3 = nn.ModuleDict()
        self.Up_conv4 = nn.ModuleDict()
        self.Up_conv5 = nn.ModuleDict()
        self.Conv_1x1 = nn.ModuleDict()
        for view in self.view_num:
            self.Maxpool[view] = nn.MaxPool2d(kernel_size=2, stride=2)

            self.Conv1[view] = conv_block(ch_in=1, ch_out=64)
            self.Conv2[view] = conv_block(ch_in=64, ch_out=128)
            self.Conv3[view] = conv_block(ch_in=128, ch_out=256)
            self.Conv4[view] = conv_block(ch_in=256, ch_out=512)
            self.Conv5[view] = conv_block(ch_in=512, ch_out=1024)

            self.Up5[view] = up_conv(ch_in=1024, ch_out=512)
            self.Up_conv5[view] = conv_block(ch_in=1024, ch_out=512)

            self.Up4[view] = up_conv(ch_in=512, ch_out=256)
            self.Up_conv4[view] = conv_block(ch_in=512, ch_out=256)

            self.Up3[view] = up_conv(ch_in=256, ch_out=128)
            self.Up_conv3[view] = conv_block(ch_in=256, ch_out=128)

            self.Up2[view] = up_conv(ch_in=128, ch_out=64)
            self.Up_conv2[view] = conv_block(ch_in=128, ch_out=64)

            self.Conv_1x1[view] = nn.Conv2d(64, 5, kernel_size=1, stride=1, padding=0)



    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        x1 = {}
        x2 = {}
        x3 = {}
        x4 = {}
        x5 = {}
        d1 = {}
        d2 = {}
        d3 = {}
        d4 = {}
        d5 = {}
        # mask={}
        # input_shape = x[self.view_num[0]].shape[-2:]

        for view in self.view_num:
            x1[view] = self.Conv1[view](x[view])

            x2[view] = self.Maxpool[view](x1[view])
            x2[view] = self.Conv2[view](x2[view])

            x3[view] = self.Maxpool[view](x2[view])
            x3[view] = self.Conv3[view](x3[view])

            x4[view] = self.Maxpool[view](x3[view])
            x4[view] = self.Conv4[view](x4[view])

            x5[view] = self.Maxpool[view](x4[view])
            x5[view] = self.Conv5[view](x5[view])

            # decoding + concat path
            d5[view] = self.Up5[view](x5[view])
            d5[view] = torch.cat((x4[view], d5[view]), dim=1)

            d5[view] = self.Up_conv5[view](d5[view])

            d4[view] = self.Up4[view](d5[view])
            d4[view] = torch.cat((x3[view], d4[view]), dim=1)
            d4[view] = self.Up_conv4[view](d4[view])

            d3[view] = self.Up3[view](d4[view])
            d3[view] = torch.cat((x2[view], d3[view]), dim=1)
            d3[view] = self.Up_conv3[view](d3[view])

            d2[view] = self.Up2[view](d3[view])
            d2[view] = torch.cat((x1[view], d2[view]), dim=1)
            d2[view] = self.Up_conv2[view](d2[view])

            d1[view] = self.Conv_1x1[view](d2[view])

        return d1,None,None,x5

class multiview_unet(nn.Module):
    def __init__(self,view_num):
        super(multiview_unet,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.Maxpool = nn.ModuleDict()
        self.Conv1 = nn.ModuleDict()
        self.Conv2 = nn.ModuleDict()
        self.Conv3 = nn.ModuleDict()
        self.Conv4 = nn.ModuleDict()
        self.Conv5 = nn.ModuleDict()
        self.Up1 = nn.ModuleDict()
        self.Up2 = nn.ModuleDict()
        self.Up3 = nn.ModuleDict()
        self.Up4 = nn.ModuleDict()
        self.Up5 = nn.ModuleDict()
        self.Up_conv1 = nn.ModuleDict()
        self.Up_conv2 = nn.ModuleDict()
        self.Up_conv3 = nn.ModuleDict()
        self.Up_conv4 = nn.ModuleDict()
        self.Up_conv5 = nn.ModuleDict()
        self.Conv_1x1 = nn.ModuleDict()
        self.global_attn = TPAVIModule(in_channels=1024, mode='dot')
        for view in self.view_num:
            self.Maxpool[view] = nn.MaxPool2d(kernel_size=2, stride=2)

            self.Conv1[view] = conv_block(ch_in=1, ch_out=64)
            self.Conv2[view] = conv_block(ch_in=64, ch_out=128)
            self.Conv3[view] = conv_block(ch_in=128, ch_out=256)
            self.Conv4[view] = conv_block(ch_in=256, ch_out=512)
            self.Conv5[view] = conv_block(ch_in=512, ch_out=1024)

            self.Up5[view] = up_conv(ch_in=1024, ch_out=512)
            self.Up_conv5[view] = conv_block(ch_in=1024, ch_out=512)

            self.Up4[view] = up_conv(ch_in=512, ch_out=256)
            self.Up_conv4[view] = conv_block(ch_in=512, ch_out=256)

            self.Up3[view] = up_conv(ch_in=256, ch_out=128)
            self.Up_conv3[view] = conv_block(ch_in=256, ch_out=128)

            self.Up2[view] = up_conv(ch_in=128, ch_out=64)
            self.Up_conv2[view] = conv_block(ch_in=128, ch_out=64)

            self.Conv_1x1[view] = nn.Conv2d(64, 5, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        x1 = {}
        x2 = {}
        x3 = {}
        x4 = {}
        x5 = {}
        d1 = {}
        d2 = {}
        d3 = {}
        d4 = {}
        d5 = {}
        # mask={}
        # input_shape = x[self.view_num[0]].shape[-2:]

        for view in self.view_num:
            x1[view] = self.Conv1[view](x[view])

            x2[view] = self.Maxpool[view](x1[view])
            x2[view] = self.Conv2[view](x2[view])

            x3[view] = self.Maxpool[view](x2[view])
            x3[view] = self.Conv3[view](x3[view])

            x4[view] = self.Maxpool[view](x3[view])
            x4[view] = self.Conv4[view](x4[view])

            x5[view] = self.Maxpool[view](x4[view])
            x5[view] = self.Conv5[view](x5[view]) # 2, 1024, 7, 7
            # print(x5[view].shape)

        global_concat_featuer_map = [x5[view].unsqueeze(2) for view in self.view_num] #bs * 1024 * 1 * h * w
        global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)#bs * 1024 * 3 * h * w
        global_conv_feat, _ = self.global_attn(global_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            x5[view] = global_conv_feat[:, :, i, :, :]

        for view in self.view_num:
            # decoding + concat path
            d5[view] = self.Up5[view](x5[view])
            d5[view] = torch.cat((x4[view], d5[view]), dim=1)

            d5[view] = self.Up_conv5[view](d5[view])

            d4[view] = self.Up4[view](d5[view])
            d4[view] = torch.cat((x3[view], d4[view]), dim=1)
            d4[view] = self.Up_conv4[view](d4[view])

            d3[view] = self.Up3[view](d4[view])
            d3[view] = torch.cat((x2[view], d3[view]), dim=1)
            d3[view] = self.Up_conv3[view](d3[view])

            d2[view] = self.Up2[view](d3[view])
            d2[view] = torch.cat((x1[view], d2[view]), dim=1)
            d2[view] = self.Up_conv2[view](d2[view])

            d1[view] = self.Conv_1x1[view](d2[view])

        return d1,None,None,x5


class Global_and_Local_cyc_nofusion(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4'],center_aware_weight=20):
        super(Global_and_Local_cyc_nofusion,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        # self.local_attn = local_attn
        self.center_aware_weight = center_aware_weight
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.centerness = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness[view] = copy.deepcopy(self.network.classifier)
            self.centerness[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.global_attn = TPAVIModule(in_channels=2048, mode='dot')
        self.local_attn = TPAVIModule(in_channels=2048, mode='dot')

    def backbone(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
            f4_fusion[view] = copy.deepcopy(f4[view])
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_global_fusion = {}
        f4_local_fusion = {}
        f4_local = {}
        f4_fusion = {}
        mask={}
        mask_bb={}
        ctr = {}
        atten_map = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        #feature extractor
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        #M_cls
        for view in self.view_num:
            mask_bb[view] = self.classifier[view](f4[view].clone().contiguous())
            mask_bb[view] = nn.Sigmoid()(mask_bb[view])
            n, c, h, w = mask_bb[view].shape
            maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
            mask_bb[view] = maxpooling(mask_bb[view])
        #M_ctr
        for view in self.view_num:
            ctr[view] = self.centerness[view](f4[view].clone().contiguous())
            ctr[view] = nn.Sigmoid()(ctr[view])

        #get local feature
        for view in self.view_num:
            atten_map[view] = (self.center_aware_weight * mask_bb[view] * ctr[view]).sigmoid()
            f4_local[view] = f4[view].clone() * atten_map[view]

        #global fusion
        global_concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)#bs * 2048 * 3 * h * w
        global_conv_feat, _ = self.global_attn(global_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_global_fusion[view] = global_conv_feat[:, :, i, :, :]

        #local fusion
        local_concat_featuer_map = [f4_local[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        local_concat_featuer_map = torch.cat(local_concat_featuer_map, dim=2)
        local_conv_feat, _ = self.local_attn(local_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_local_fusion[view] = local_conv_feat[:, :, i, :, :]

        #fuse global feat and local feat
        for view in self.view_num:
            f4_fusion[view] = f4_global_fusion[view] + f4_local_fusion[view]

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            mask_bb[view] = self.classifier[view](f4[view].contiguous())
            mask_bb[view] = F.interpolate(mask_bb[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_bb,f4,f4_local_fusion


class Global_and_Local_conv_merge(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4'],center_aware_weight=20):
        super(Global_and_Local_conv_merge,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        # self.local_attn = local_attn
        self.center_aware_weight = center_aware_weight
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.centerness = nn.ModuleDict()
        self.merge = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.merge[view] = nn.Sequential(
                nn.Conv2d(in_channels=2048 * 2, out_channels=2048, kernel_size=1),
                nn.ReLU()
            )
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness[view] = copy.deepcopy(self.network.classifier)
            self.centerness[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.global_attn = TPAVIModule(in_channels=2048, mode='dot')
        self.local_attn = TPAVIModule(in_channels=2048, mode='dot')


    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_global_fusion = {}
        f4_local_fusion = {}
        f4_local = {}
        f4_fusion = {}
        mask={}
        mask_bb={}
        ctr = {}
        atten_map = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        #feature extractor
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        #M_cls
        for view in self.view_num:
            mask_bb[view] = self.classifier[view](f4[view].clone().contiguous())
            mask_bb[view] = nn.Sigmoid()(mask_bb[view])
            n, c, h, w = mask_bb[view].shape
            maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
            mask_bb[view] = maxpooling(mask_bb[view])
        #M_ctr
        for view in self.view_num:
            ctr[view] = self.centerness[view](f4[view].clone().contiguous())
            ctr[view] = nn.Sigmoid()(ctr[view])

        #get local feature
        for view in self.view_num:
            atten_map[view] = (self.center_aware_weight * mask_bb[view] * ctr[view]).sigmoid()
            f4_local[view] = f4[view].clone() * atten_map[view]

        #global fusion
        global_concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)#bs * 2048 * 3 * h * w
        global_conv_feat, _ = self.global_attn(global_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_global_fusion[view] = global_conv_feat[:, :, i, :, :]

        #local fusion
        local_concat_featuer_map = [f4_local[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        local_concat_featuer_map = torch.cat(local_concat_featuer_map, dim=2)
        local_conv_feat, _ = self.local_attn(local_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_local_fusion[view] = local_conv_feat[:, :, i, :, :]

        #fuse global feat and local feat
        for view in self.view_num:
            # f4_fusion[view] = f4_global_fusion[view] + f4_local_fusion[view]
            f4_fusion[view] = torch.cat([f4_global_fusion[view], f4_local_fusion[view]],dim = 1)
            f4_fusion[view] = self.merge[view](f4_fusion[view])


        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            mask_bb[view] = self.classifier[view](f4[view].contiguous())
            mask_bb[view] = F.interpolate(mask_bb[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_bb,f4_global_fusion,f4_local_fusion


class Foreground_and_Background(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4'],center_aware_weight=20):
        super(Foreground_and_Background,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        # self.local_attn = local_attn
        self.center_aware_weight = center_aware_weight
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.centerness = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness[view] = copy.deepcopy(self.network.classifier)
            self.centerness[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.global_attn = TPAVIModule(in_channels=2048, mode='dot')
        self.local_attn = TPAVIModule(in_channels=2048, mode='dot')

    def backbone(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
            f4_fusion[view] = copy.deepcopy(f4[view])
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_global_fusion = {}
        f4_local_fusion = {}
        f4_local = {}
        f4_fusion = {}
        mask={}
        mask_bb={}
        ctr = {}
        atten_map = {}
        f4_global = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        #feature extractor
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
        #M_cls
        for view in self.view_num:
            mask_bb[view] = self.classifier[view](f4[view].clone().contiguous())
            mask_bb[view] = nn.Sigmoid()(mask_bb[view])
            n, c, h, w = mask_bb[view].shape
            maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
            mask_bb[view] = maxpooling(mask_bb[view])
        #M_ctr
        for view in self.view_num:
            ctr[view] = self.centerness[view](f4[view].clone().contiguous())
            ctr[view] = nn.Sigmoid()(ctr[view])

        #get local feature
        for view in self.view_num:
            atten_map[view] = (self.center_aware_weight * mask_bb[view] * ctr[view]).sigmoid()
            f4_local[view] = f4[view].clone() * atten_map[view]
            f4_global[view] = f4[view].clone() * (torch.ones_like(atten_map[view])-atten_map[view])

        #global fusion
        global_concat_featuer_map = [f4_global[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)#bs * 2048 * 3 * h * w
        global_conv_feat, _ = self.global_attn(global_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_global_fusion[view] = global_conv_feat[:, :, i, :, :]

        #local fusion
        local_concat_featuer_map = [f4_local[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        local_concat_featuer_map = torch.cat(local_concat_featuer_map, dim=2)
        local_conv_feat, _ = self.local_attn(local_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_local_fusion[view] = local_conv_feat[:, :, i, :, :]

        #fuse global feat and local feat
        for view in self.view_num:
            f4_fusion[view] = f4_global_fusion[view] + f4_local_fusion[view]

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            mask_bb[view] = self.classifier[view](f4[view].contiguous())
            mask_bb[view] = F.interpolate(mask_bb[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_bb,f4_fusion,None

class Global_only_cyc_nofusion(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4'],center_aware_weight=20):
        super(Global_only_cyc_nofusion,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        # self.local_attn = local_attn
        self.center_aware_weight = center_aware_weight
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block=nn.ModuleDict()
        self.layer1 = nn.ModuleDict()
        self.layer2 = nn.ModuleDict()
        self.layer3 = nn.ModuleDict()
        self.layer4 = nn.ModuleDict()
        self.classifier = nn.ModuleDict()
        self.centerness = nn.ModuleDict()
        for view in self.view_num:
            self.init_block[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness[view] = copy.deepcopy(self.network.classifier)
            self.centerness[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.global_attn = TPAVIModule(in_channels=2048, mode='dot')
        self.local_attn = TPAVIModule(in_channels=2048, mode='dot')

    def backbone(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_fusion={}
        mask={}
        input_shape = x[self.view_num[0]].shape[-2:]
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])
            f4_fusion[view] = copy.deepcopy(f4[view])
        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,f4

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_global_fusion = {}
        f4_local_fusion = {}
        f4_local = {}
        f4_fusion = {}
        mask={}
        mask_bb={}
        ctr = {}
        atten_map = {}
        input_shape = x[self.view_num[0]].shape[-2:]
        #feature extractor
        for view in self.view_num:
            f0[view] = self.init_block[view](x[view])
            f1[view] = self.layer1[view](f0[view])
            f2[view] = self.layer2[view](f1[view])
            f3[view] = self.layer3[view](f2[view])
            f4[view] = self.layer4[view](f3[view])

        #global fusion
        global_concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)#bs * 2048 * 3 * h * w
        global_conv_feat, _ = self.global_attn(global_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_global_fusion[view] = global_conv_feat[:, :, i, :, :]

        #fuse global feat and local feat
        for view in self.view_num:
            f4_fusion[view] = f4_global_fusion[view]

        for view in self.view_num:
            mask[view] = self.classifier[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)

            mask_bb[view] = self.classifier[view](f4[view].contiguous())
            mask_bb[view] = F.interpolate(mask_bb[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_bb,f4,None


class Global_and_Local_CPS(nn.Module):
    def __init__(self,view_num,test_view=['1','2','3','4'],center_aware_weight=20):
        super(Global_and_Local_CPS,self).__init__()
        self.outchannel_list = {'1':2,'2':1,'3':2,'4':4}
        self.view_num = view_num
        self.test_view = test_view
        # self.local_attn = local_attn
        self.center_aware_weight = center_aware_weight
        self.network = deeplabv3_resnet50_iekd(pretrained=False, aux_loss=False)
        self.init_block_1=nn.ModuleDict()
        self.layer1_1 = nn.ModuleDict()
        self.layer2_1 = nn.ModuleDict()
        self.layer3_1 = nn.ModuleDict()
        self.layer4_1 = nn.ModuleDict()
        self.classifier_1 = nn.ModuleDict()
        self.centerness_1 = nn.ModuleDict()

        self.init_block_2=nn.ModuleDict()
        self.layer1_2 = nn.ModuleDict()
        self.layer2_2 = nn.ModuleDict()
        self.layer3_2 = nn.ModuleDict()
        self.layer4_2 = nn.ModuleDict()
        self.classifier_2 = nn.ModuleDict()
        self.centerness_2 = nn.ModuleDict()

        for view in self.view_num:
            self.init_block_1[view]=copy.deepcopy(nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            ))
            self.layer1_1[view]=copy.deepcopy(self.network.backbone['layer1'])
            self.layer2_1[view]=copy.deepcopy(self.network.backbone['layer2'])
            self.layer3_1[view]=copy.deepcopy(self.network.backbone['layer3'])
            self.layer4_1[view]=copy.deepcopy(self.network.backbone['layer4'])
            self.classifier_1[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier_1[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness_1[view] = copy.deepcopy(self.network.classifier)
            self.centerness_1[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.global_attn_1 = TPAVIModule(in_channels=2048, mode='dot')
        self.local_attn_1 = TPAVIModule(in_channels=2048, mode='dot')

        for view in self.view_num:
            self.init_block_2[view]=nn.Sequential(
                self.network.backbone['conv1'],
                self.network.backbone['bn1'],
                self.network.backbone['relu'],
                self.network.backbone['maxpool']
            )
            self.layer1_2[view]=self.network.backbone['layer1']
            self.layer2_2[view]=self.network.backbone['layer2']
            self.layer3_2[view]=self.network.backbone['layer3']
            self.layer4_2[view]=self.network.backbone['layer4']
            self.classifier_2[view]=copy.deepcopy(self.network.classifier)
            #全种类输出
            self.classifier_2[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        5,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

            self.centerness_2[view] = copy.deepcopy(self.network.classifier)
            self.centerness_2[view][-1] = torch.nn.Conv2d(self.network.classifier[-1].in_channels,
                                                        1,
                                                        kernel_size=self.network.classifier[-1].kernel_size)  # change number of outputs to 1

        self.global_attn_2 = TPAVIModule(in_channels=2048, mode='dot')
        self.local_attn_2 = TPAVIModule(in_channels=2048, mode='dot')

    def forward(self,x):
        """
        输入为字典,view -> img
        返回为字典,view -> mask
        """
        f0={}
        f1={}
        f2={}
        f3={}
        f4={}
        f4_global_fusion = {}
        f4_local_fusion = {}
        f4_local = {}
        f4_fusion = {}
        mask={}
        mask_bb={}
        ctr = {}
        atten_map = {}
        f0_2 = {}
        f1_2 = {}
        f2_2 = {}
        f3_2 = {}
        f4_2 = {}
        f4_global_fusion_2 = {}
        f4_local_fusion_2 = {}
        f4_local_2 = {}
        f4_fusion_2 = {}
        mask_2 = {}
        mask_bb_2 = {}
        ctr_2 = {}
        atten_map_2 = {}
        input_shape = x[self.view_num[0]].shape[-2:]

        """########## model 1 ###############"""

        #feature extractor
        for view in self.view_num:
            f0[view] = self.init_block_1[view](x[view])
            f1[view] = self.layer1_1[view](f0[view])
            f2[view] = self.layer2_1[view](f1[view])
            f3[view] = self.layer3_1[view](f2[view])
            f4[view] = self.layer4_1[view](f3[view])
        #M_cls
        for view in self.view_num:
            mask_bb[view] = self.classifier_1[view](f4[view].clone().contiguous())
            mask_bb[view] = nn.Sigmoid()(mask_bb[view])
            n, c, h, w = mask_bb[view].shape
            maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
            mask_bb[view] = maxpooling(mask_bb[view])

        #M_ctr
        for view in self.view_num:
            ctr[view] = self.centerness_1[view](f4[view].clone().contiguous())
            ctr[view] = nn.Sigmoid()(ctr[view])

        #get local feature
        for view in self.view_num:
            atten_map[view] = (self.center_aware_weight * mask_bb[view] * ctr[view]).sigmoid()
            f4_local[view] = f4[view].clone() * atten_map[view]

        #global fusion
        global_concat_featuer_map = [f4[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        global_concat_featuer_map = torch.cat(global_concat_featuer_map, dim=2)#bs * 2048 * 3 * h * w
        global_conv_feat, _ = self.global_attn_1(global_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_global_fusion[view] = global_conv_feat[:, :, i, :, :]

        #local fusion
        local_concat_featuer_map = [f4_local[view].unsqueeze(2) for view in self.view_num] #bs * 2048 * 1 * h * w
        local_concat_featuer_map = torch.cat(local_concat_featuer_map, dim=2)
        local_conv_feat, _ = self.local_attn_1(local_concat_featuer_map)
        for i,view in enumerate(self.view_num):
            f4_local_fusion[view] = local_conv_feat[:, :, i, :, :]

        #fuse global feat and local feat
        for view in self.view_num:
            f4_fusion[view] = f4_global_fusion[view] + f4_local_fusion[view]

        for view in self.view_num:
            mask[view] = self.classifier_1[view](f4_fusion[view].contiguous())
            mask[view] = F.interpolate(mask[view], size=input_shape, mode='bilinear', align_corners=False)



        """########## model 2 ###############"""

        # feature extractor
        for view in self.view_num:
            f0_2[view] = self.init_block_2[view](x[view])
            f1_2[view] = self.layer1_2[view](f0_2[view])
            f2_2[view] = self.layer2_2[view](f1_2[view])
            f3_2[view] = self.layer3_2[view](f2_2[view])
            f4_2[view] = self.layer4_2[view](f3_2[view])
        # M_cls
        for view in self.view_num:
            mask_bb_2[view] = self.classifier_2[view](f4_2[view].clone().contiguous())
            mask_bb_2[view] = nn.Sigmoid()(mask_bb_2[view])
            n, c, h, w = mask_bb_2[view].shape
            maxpooling_2 = nn.AdaptiveMaxPool3d((1, h, w))
            mask_bb_2[view] = maxpooling_2(mask_bb_2[view])
        # M_ctr
        for view in self.view_num:
            ctr_2[view] = self.centerness_2[view](f4_2[view].clone().contiguous())
            ctr_2[view] = nn.Sigmoid()(ctr_2[view])

        # get local feature
        for view in self.view_num:
            atten_map_2[view] = (self.center_aware_weight * mask_bb_2[view] * ctr_2[view]).sigmoid()
            f4_local_2[view] = f4_2[view].clone() * atten_map_2[view]

        # global fusion
        global_concat_featuer_map_2 = [f4_2[view].unsqueeze(2) for view in self.view_num]  # bs * 2048 * 1 * h * w
        global_concat_featuer_map_2 = torch.cat(global_concat_featuer_map_2, dim=2)  # bs * 2048 * 3 * h * w
        global_conv_feat_2, _ = self.global_attn_2(global_concat_featuer_map_2)
        for i, view in enumerate(self.view_num):
            f4_global_fusion_2[view] = global_conv_feat_2[:, :, i, :, :]

        # local fusion
        local_concat_featuer_map_2 = [f4_local_2[view].unsqueeze(2) for view in self.view_num]  # bs * 2048 * 1 * h * w
        local_concat_featuer_map_2 = torch.cat(local_concat_featuer_map_2, dim=2)
        local_conv_feat_2, _ = self.local_attn_2(local_concat_featuer_map_2)
        for i, view in enumerate(self.view_num):
            f4_local_fusion_2[view] = local_conv_feat_2[:, :, i, :, :]

        # fuse global feat and local feat
        for view in self.view_num:
            f4_fusion_2[view] = f4_global_fusion_2[view] + f4_local_fusion_2[view]

        for view in self.view_num:
            mask_2[view] = self.classifier_2[view](f4_fusion_2[view].contiguous())
            mask_2[view] = F.interpolate(mask_2[view], size=input_shape, mode='bilinear', align_corners=False)

            # mask_bb_2[view] = self.classifier_2[view](f4_2[view].contiguous())
            # mask_bb_2[view] = F.interpolate(mask_bb_2[view], size=input_shape, mode='bilinear', align_corners=False)

        return mask,mask_2,f4_global_fusion,f4_local_fusion

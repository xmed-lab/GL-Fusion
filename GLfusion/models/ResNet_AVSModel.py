import math

import torch
import torch.nn as nn
import torchvision.models as models
from models.resnet import B2_ResNet
from models.TPAVI import TPAVIModule
import pdb
import torch.nn.functional as F

class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, in_features,out_features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_features, in_features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            in_features, in_features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, in_features,out_features,interplot_size):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_features, in_features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            in_features, out_features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU()
        self.resConfUnit1 = ResidualConvUnit(in_features=in_features,out_features=in_features)
        self.resConfUnit2 = ResidualConvUnit(in_features=in_features,out_features=in_features)
        self.interplot_size = interplot_size

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        #xs[0]:1024
        #xs[1]:1024
        #output:512
        output = xs[0]


        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = self.relu(output)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        # output = nn.functional.interpolate(
        #     output, scale_factor=2, mode="bilinear", align_corners=True
        # )
        output = nn.functional.interpolate(
            output, size=self.interplot_size, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class Pred_endecoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=256, config=None, tpavi_stages=[0,1,2,3], tpavi_vv_flag=False, tpavi_va_flag=True,PRETRAINED_RESNET50_PATH='/home/listu/zyzheng/PAH/pretrain_backbone/resnet50-19c8e357.pth'):
        super(Pred_endecoder, self).__init__()
        self.PRETRAINED_RESNET50_PATH = PRETRAINED_RESNET50_PATH
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag

        self.resnet = B2_ResNet()
        self.resnet2 = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.path4 = FeatureFusionBlock(channel,interplot_size=[6,6])
        self.path3 = FeatureFusionBlock(channel,interplot_size=[11,11])
        self.path2 = FeatureFusionBlock(channel,interplot_size=[21,21])
        self.path1 = FeatureFusionBlock(channel,interplot_size=[42,42])

        for i in self.tpavi_stages:
            setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel, mode='dot'))
            print("==> Build TPAVI block...")

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 5, kernel_size=1, stride=1, padding=0),
        )

        if self.training:
            self.initialize_weights()


    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        x = x.reshape(-1, 1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_vv(self, x, stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, _ = tpavi_b(x) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x
    def tpavi_vv_multiview(self, x,o,stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        o = self.pre_reshape_for_tpavi(o)
        x, _ = tpavi_b(x,o) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x
    def tpavi_va(self, x, audio, stage):
        # x: visual, [B*T, C=256, H, W]
        # audio: [B*T, 128]
        # ra_flag: return audio feature list or not
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        audio = audio.view(-1, 5, audio.shape[-1]) # [B, T, 128]
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x, a

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, other_view=None):
        """以view1为主模态，其他view作为辅助分割模态"""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)     # B x 64 x 21 x 21
        x1 = self.resnet.layer1(x)     # BF x 256  x 21 x 21
        x2 = self.resnet.layer2(x1)    # BF x 512  x 11 x 11
        x3 = self.resnet.layer3_1(x2)  # BF x 1024 x 6 x 6
        x4 = self.resnet.layer4_1(x3)  # BF x 2048 x  3 x  3
        # print(x1.shape, x2.shape, x3.shape, x4.shape)

        conv1_feat = self.conv1(x1)  # BF x 256 x 21 x 21
        conv2_feat = self.conv2(x2)  # BF x 256 x 11 x 11
        conv3_feat = self.conv3(x3)  # BF x 256 x 6 x 6
        conv4_feat = self.conv4(x4)  # BF x 256 x 3 x 3

        o = self.resnet2.conv1(other_view)
        o = self.resnet2.bn1(o)
        o = self.resnet2.relu(o)
        o = self.resnet2.maxpool(o)  # B x 64 x 21 x 21
        o1 = self.resnet2.layer1(o)  # BF x 256  x 21 x 21
        o2 = self.resnet2.layer2(o1)  # BF x 512  x 11 x 11
        o3 = self.resnet2.layer3_1(o2)  # BF x 1024 x 6 x 6
        o4 = self.resnet2.layer4_1(o3)  # BF x 2048 x  3 x  3
        # print(conv1_feat.shape, conv2_feat.shape, conv3_feat.shape, conv4_feat.shape)
        conv1_o_feat = self.conv1(o1)  # BF x 256 x 21 x 21
        conv2_o_feat = self.conv2(o2)  # BF x 256 x 11 x 11
        conv3_o_feat = self.conv3(o3)  # BF x 256 x 6 x 6
        conv4_o_feat = self.conv4(o4)   # BF x 256 x 3 x 3

        feature_map_list = [conv1_feat,conv2_feat,conv3_feat,conv4_feat]
        other_view_list = [conv1_o_feat, conv2_o_feat, conv3_o_feat, conv4_o_feat]
        a_fea_list = [None] * 4

        # if len(self.tpavi_stages) > 0:
        #     if (not self.tpavi_vv_flag) and (not self.tpavi_va_flag):
        #         raise Exception('tpavi_vv_flag and tpavi_va_flag cannot be False at the same time if len(tpavi_stages)>0, \
        #             tpavi_vv_flag is for video self-attention while tpavi_va_flag indicates the standard version (audio-visual attention)')
        #     for i in self.tpavi_stages:
        #         tpavi_count = 0
        #         conv_feat = torch.zeros_like(feature_map_list[i]).cuda()
        #         if self.tpavi_vv_flag:
        #             conv_feat_vv = self.tpavi_vv(feature_map_list[i], stage=i)
        #             conv_feat += conv_feat_vv
        #             tpavi_count += 1
        #         if self.tpavi_va_flag:
        #             conv_feat_va, a_fea = self.tpavi_va(feature_map_list[i], other_view, stage=i)
        #             conv_feat += conv_feat_va
        #             tpavi_count += 1
        #             a_fea_list[i] = a_fea
        #         conv_feat /= tpavi_count
        #         feature_map_list[i] = conv_feat # update features of stage-i which conduct TPAVI
        if len(self.tpavi_stages) > 0:
            for i in self.tpavi_stages:
                tpavi_count = 0
                conv_feat = torch.zeros_like(feature_map_list[i]).cuda()

                conv_feat_vv = self.tpavi_vv_multiview(feature_map_list[i],other_view_list[i], stage=i)
                conv_feat += conv_feat_vv
                tpavi_count += 1

                conv_feat /= tpavi_count
                feature_map_list[i] = conv_feat # update features of stage-i which conduct TPAVI

        conv4_feat = self.path4(feature_map_list[3])            # BF x 256 x 14 x 14
        conv43 = self.path3(conv4_feat, feature_map_list[2])    # BF x 256 x 28 x 28
        conv432 = self.path2(conv43, feature_map_list[1])       # BF x 256 x 56 x 56
        conv4321 = self.path1(conv432, feature_map_list[0])     # BF x 256 x 112 x 112
        # print(conv4_feat.shape, conv43.shape, conv432.shape, conv4321.shape)

        pred = self.output_conv(conv4321)   # BF x 1 x 224 x 224
        # print(pred.shape)

        return pred


    def initialize_weights(self):
        res50 = models.resnet50(pretrained=False)
        resnet50_dict = torch.load(self.PRETRAINED_RESNET50_PATH)
        res50.load_state_dict(resnet50_dict)
        pretrained_dict = res50.state_dict()
        # print(pretrained_dict.keys())
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
        self.resnet2.load_state_dict(all_params)
        print(f'==> Load pretrained ResNet50 parameters from {self.PRETRAINED_RESNET50_PATH}')


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

    def __init__(self, n_embd, n_head,channel, view_num,attn_pdrop=0, resid_pdrop=0):
        super(transformer,self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.attn = SelfAttention(n_embd=n_embd, n_head=n_head,attn_pdrop = attn_pdrop,resid_pdrop = resid_pdrop)
        self.view_num = view_num
        self.mlp = nn.ModuleDict()
        for view in self.view_num:
            self.mlp[view] = nn.Linear(in_features=n_embd,out_features=n_embd)
        self.norm_layer = nn.LayerNorm(channel*len(self.view_num))
        self.bn = nn.BatchNorm2d(channel*len(self.view_num))
    def forward(self, x):
        #batch_size * channel * h * w * len(view_num)
        batch_size,channel,h,w = x[self.view_num[0]].shape
        x = torch.cat([x[view] for view in self.view_num],dim=1)
        x = x.reshape([batch_size,channel*len(self.view_num),-1])
        # x = x.reshape([batch_size, channel , -1])
        #TODO 记得trm模块有改动
        attn_x = self.attn(x)
        attn_x = attn_x.reshape(batch_size,channel*self.view_num,h,w)
        attn_x = self.bn(attn_x)
        x = x+attn_x
        x = x.reshape([batch_size,channel,h,w,len(self.view_num)])
        x = self.norm_layer(x)

        y = {}
        for i,view in enumerate(self.view_num):
            y[view] = x[...,i]

        return y


class AVS_Transfusion(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=256, config=None, tpavi_stages=[0,1,2,3],view_num=['1','2','3','4'], tpavi_vv_flag=False, tpavi_va_flag=True,PRETRAINED_RESNET50_PATH='/home/listu/zyzheng/PAH/pretrain_backbone/resnet50-19c8e357.pth'):
        super(AVS_Transfusion, self).__init__()
        self.PRETRAINED_RESNET50_PATH = PRETRAINED_RESNET50_PATH
        self.view_num=view_num
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag
        #share encoder
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.path4 = FeatureFusionBlock(channel,interplot_size=[6,6])
        self.path3 = FeatureFusionBlock(channel,interplot_size=[11,11])
        self.path2 = FeatureFusionBlock(channel,interplot_size=[21,21])
        self.path1 = FeatureFusionBlock(channel,interplot_size=[42,42])

        for i in self.tpavi_stages:
            setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel, mode='dot'))
            print("==> Build TPAVI block...")

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 5, kernel_size=1, stride=1, padding=0),
        )

        self.attn1 = transformer(n_embd=21 * 21, n_head=1,view_num=self.view_num)
        self.attn2 = transformer(n_embd=11 * 11, n_head=1, view_num=self.view_num)
        self.attn3 = transformer(n_embd=6 * 6, n_head=1, view_num=self.view_num)
        self.attn4 = transformer(n_embd=3 * 3, n_head=1, view_num=self.view_num)

        if self.training:
            self.initialize_weights()


    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        x = x.reshape(-1, 1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_vv(self, x, stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, _ = tpavi_b(x) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x
    def tpavi_vv_multiview(self, x,o,stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        o = self.pre_reshape_for_tpavi(o)
        x, _ = tpavi_b(x,o) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x
    def tpavi_va(self, x, audio, stage):
        # x: visual, [B*T, C=256, H, W]
        # audio: [B*T, 128]
        # ra_flag: return audio feature list or not
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        audio = audio.view(-1, 5, audio.shape[-1]) # [B, T, 128]
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x, a

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        """以view1为主模态，其他view作为辅助分割模态"""
        for view in self.view_num:
            x[view] = x[view].repeat(1,3,1,1)
        x1={}
        x2={}
        x3={}
        x4={}
        conv1_feat={}
        conv2_feat = {}
        conv3_feat = {}
        conv4_feat = {}
        for view in self.view_num:
            x[view] = self.resnet.conv1(x[view])
            x[view] = self.resnet.bn1(x[view])
            x[view] = self.resnet.relu(x[view])
            x[view] = self.resnet.maxpool(x[view])     # B x 64 x 21 x 21
            x1[view] = self.resnet.layer1(x[view])     # BF x 256  x 21 x 21
            x2[view] = self.resnet.layer2(x1[view])    # BF x 512  x 11 x 11
            x3[view] = self.resnet.layer3_1(x2[view])  # BF x 1024 x 6 x 6
            x4[view] = self.resnet.layer4_1(x3[view])  # BF x 2048 x  3 x  3
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        for view in self.view_num:
            conv1_feat[view] = self.conv1(x1[view])  # BF x 256 x 21 x 21
            conv2_feat[view] = self.conv2(x2[view])  # BF x 256 x 11 x 11
            conv3_feat[view] = self.conv3(x3[view])  # BF x 256 x 6 x 6
            conv4_feat[view] = self.conv4(x4[view])  # BF x 256 x 3 x 3

        feature_map_list={}
        for view in self.view_num:
            feature_map_list[view] = [conv1_feat[view],conv2_feat[view],conv3_feat[view],conv4_feat[view]]
        a_fea_list = [None] * 4

        # if len(self.tpavi_stages) > 0:
        #     for i in self.tpavi_stages:
        #         tpavi_count = 0
        #         conv_feat = torch.zeros_like(feature_map_list[i]).cuda()
        #
        #         conv_feat_vv = self.tpavi_vv_multiview(feature_map_list[i],other_view_list[i], stage=i)
        #         conv_feat += conv_feat_vv
        #         tpavi_count += 1
        #
        #         conv_feat /= tpavi_count
        #         feature_map_list[i] = conv_feat # update features of stage-i which conduct TPAVI
        if len(self.tpavi_stages) > 0:
            for i in self.tpavi_stages:
                tpavi_count = 0
                concat_featuer_map={}
                for view in self.view_num:
                    concat_featuer_map[view] =  feature_map_list[view][i]

                self_attn = getattr(self, f'attn{i+1}')
                conv_feat = self_attn(concat_featuer_map)
                tpavi_count = tpavi_count+1

                for view in self.view_num:
                    conv_feat[view] = conv_feat[view]/tpavi_count
                    feature_map_list[view][i] = conv_feat[view]  # update features of stage-i which conduct TPAVI
        conv4_feat={}
        conv43={}
        conv432={}
        conv4321={}
        pred={}
        for view in self.view_num:
            conv4_feat[view] = self.path4(feature_map_list[view][3])            # BF x 256 x 14 x 14
            conv43[view] = self.path3(conv4_feat[view], feature_map_list[view][2])    # BF x 256 x 28 x 28
            conv432[view] = self.path2(conv43[view], feature_map_list[view][1])       # BF x 256 x 56 x 56
            conv4321[view] = self.path1(conv432[view], feature_map_list[view][0])     # BF x 256 x 112 x 112
            # print(conv4_feat.shape, conv43.shape, conv432.shape, conv4321.shape)
            pred[view] = self.output_conv(conv4321[view])   # BF x 1 x 224 x 224
        # print(pred.shape)

        return pred,None


    def initialize_weights(self):
        res50 = models.resnet50(pretrained=False)
        resnet50_dict = torch.load(self.PRETRAINED_RESNET50_PATH)
        res50.load_state_dict(resnet50_dict)
        pretrained_dict = res50.state_dict()
        # print(pretrained_dict.keys())
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
        # self.resnet2.load_state_dict(all_params)
        print(f'==> Load pretrained ResNet50 parameters from {self.PRETRAINED_RESNET50_PATH}')

class model17(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=256, config=None, tpavi_stages=[0,1,2,3],view_num=['1','2','3','4'], tpavi_vv_flag=False, tpavi_va_flag=True,PRETRAINED_RESNET50_PATH='/home/listu/zyzheng/PAH/pretrain_backbone/resnet50-19c8e357.pth'):
        super(model17, self).__init__()
        self.PRETRAINED_RESNET50_PATH = PRETRAINED_RESNET50_PATH
        self.view_num=view_num
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag
        self.relu = nn.ReLU(inplace=True)
        self.resnet = nn.ModuleDict()
        self.conv1 = nn.ModuleDict()
        self.conv2 = nn.ModuleDict()
        self.conv3 = nn.ModuleDict()
        self.conv4 = nn.ModuleDict()
        self.path1 = nn.ModuleDict()
        self.path2 = nn.ModuleDict()
        self.path3 = nn.ModuleDict()
        self.path4 = nn.ModuleDict()
        self.output_conv = nn.ModuleDict()
        for view in self.view_num:
            self.resnet[view] = B2_ResNet()

            self.conv4[view] = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 2048, 2048)
            self.conv3[view] = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 1024, 1024)
            self.conv2[view] = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 512, 512)
            self.conv1[view] = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 256, 256)

            self.path4[view] = FeatureFusionBlock(in_features=2048,out_features=1024,interplot_size=[6,6])
            self.path3[view] = FeatureFusionBlock(in_features=1024,out_features=512,interplot_size=[11,11])
            self.path2[view] = FeatureFusionBlock(in_features=512,out_features=256,interplot_size=[21,21])
            self.path1[view] = FeatureFusionBlock(in_features=256,out_features=256,interplot_size=[42,42])

        channel_list = [256,512,1024,2048]
        for i in self.tpavi_stages:
            setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel_list[i], mode='dot'))
            print("==> Build TPAVI block...")
        for view in self.view_num:
            self.output_conv[view] = nn.Sequential(
                nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 5, kernel_size=1, stride=1, padding=0),
            )


        if self.training:
            self.initialize_weights()


    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        x = x.reshape(-1, 1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_vv(self, x, stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, _ = tpavi_b(x) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x
    def tpavi_vv_multiview(self, x,o,stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        o = self.pre_reshape_for_tpavi(o)
        x, _ = tpavi_b(x,o) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x
    def tpavi_va(self, x, audio, stage):
        # x: visual, [B*T, C=256, H, W]
        # audio: [B*T, 128]
        # ra_flag: return audio feature list or not
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        audio = audio.view(-1, 5, audio.shape[-1]) # [B, T, 128]
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x, a

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        """以view1为主模态，其他view作为辅助分割模态"""
        for view in self.view_num:
            x[view] = x[view].repeat(1,3,1,1)
        x1={}
        x2={}
        x3={}
        x4={}
        conv1_feat={}
        conv2_feat = {}
        conv3_feat = {}
        conv4_feat = {}
        for view in self.view_num:
            x[view] = self.resnet[view].conv1(x[view])
            x[view] = self.resnet[view].bn1(x[view])
            x[view] = self.resnet[view].relu(x[view])
            x[view] = self.resnet[view].maxpool(x[view])     # B x 64 x 21 x 21
            x1[view] = self.resnet[view].layer1(x[view])     # BF x 256  x 21 x 21
            x2[view] = self.resnet[view].layer2(x1[view])    # BF x 512  x 11 x 11
            x3[view] = self.resnet[view].layer3_1(x2[view])  # BF x 1024 x 6 x 6
            x4[view] = self.resnet[view].layer4_1(x3[view])  # BF x 2048 x  3 x  3
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        for view in self.view_num:
            conv1_feat[view] = self.conv1[view](x1[view])  # BF x 256 x 21 x 21
            conv2_feat[view] = self.conv2[view](x2[view])  # BF x 256 x 11 x 11
            conv3_feat[view] = self.conv3[view](x3[view])  # BF x 256 x 6 x 6
            conv4_feat[view] = self.conv4[view](x4[view])  # BF x 256 x 3 x 3

        feature_map_list={}
        for view in self.view_num:
            feature_map_list[view] = [conv1_feat[view],conv2_feat[view],conv3_feat[view],conv4_feat[view]]
        a_fea_list = [None] * 4

        if len(self.tpavi_stages) > 0:
            for i in self.tpavi_stages:
                tpavi_count = 0
                concat_featuer_map=[]
                # for view in self.view_num:
                #     concat_featuer_map[view] =  feature_map_list[view][i]
                concat_featuer_map = [feature_map_list[view][i].unsqueeze(2) for view in self.view_num]
                concat_featuer_map = torch.cat(concat_featuer_map,dim=2)
                tpavi_b = getattr(self, f'tpavi_b{i+1}')
                conv_feat,_ = tpavi_b(concat_featuer_map)
                tpavi_count = tpavi_count+1
                conv_feat = conv_feat / tpavi_count
                # for view in self.view_num:
                #     feature_map_list[view][i] = conv_feat[:,:,int(view)-1,:,:]  # update features of stage-i which conduct TPAVI
                feature_map_list['1'][i] = conv_feat[:, :, 0, :, :]
                feature_map_list['3'][i] = conv_feat[:, :, 1, :, :]
                feature_map_list['4'][i] = conv_feat[:, :, 2, :, :]
        conv4_feat={}
        conv43={}
        conv432={}
        conv4321={}
        pred={}
        for view in self.view_num:
            conv4_feat[view] = self.path4[view](feature_map_list[view][3])            # BF x 2048 x  3 x  3
            conv43[view] = self.path3[view](conv4_feat[view], feature_map_list[view][2])    # BF x 256 x 6 x 6
            conv432[view] = self.path2[view](conv43[view], feature_map_list[view][1])       # BF x 256 x 11 x 11
            conv4321[view] = self.path1[view](conv432[view], feature_map_list[view][0])     # BF x 256 x 21 x 21
            # print(conv4_feat.shape, conv43.shape, conv432.shape, conv4321.shape)
            pred[view] = self.output_conv[view](conv4321[view])   # BF x 5 x 84 x 84
        # print(pred.shape)

        return pred,None


    def initialize_weights(self):

        res50 = models.resnet50(pretrained=False)
        resnet50_dict = torch.load(self.PRETRAINED_RESNET50_PATH)
        res50.load_state_dict(resnet50_dict)
        pretrained_dict = res50.state_dict()
        # print(pretrained_dict.keys())
        all_params = {}
        for k, v in self.resnet[self.view_num[0]].state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet[self.view_num[0]].state_dict().keys())
        for view in self.view_num:
            self.resnet[view].load_state_dict(all_params)
        # self.resnet2.load_state_dict(all_params)
        print(f'==> Load pretrained ResNet50 parameters from {self.PRETRAINED_RESNET50_PATH}')

class AVS_baseline(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=256, config=None, tpavi_stages=[],view_num=['1','2','3','4'], tpavi_vv_flag=False, tpavi_va_flag=True,PRETRAINED_RESNET50_PATH='/home/listu/zyzheng/PAH/pretrain_backbone/resnet50-19c8e357.pth'):
        super(AVS_baseline, self).__init__()
        self.PRETRAINED_RESNET50_PATH = PRETRAINED_RESNET50_PATH
        self.view_num=view_num
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag
        #share encoder
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 2048, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 1024, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 512, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], 256, 256)

        self.path4 = FeatureFusionBlock(in_features=2048,out_features=1024,interplot_size=[6,6])
        self.path3 = FeatureFusionBlock(in_features=1024,out_features=512,interplot_size=[11,11])
        self.path2 = FeatureFusionBlock(in_features=512,out_features=256,interplot_size=[21,21])
        self.path1 = FeatureFusionBlock(in_features=256,out_features=256,interplot_size=[42,42])

        channel_list = [256,512,1024,2048]
        for i in self.tpavi_stages:
            setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel_list[i], mode='dot'))
            print("==> Build TPAVI block...")

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 5, kernel_size=1, stride=1, padding=0),
        )


        if self.training:
            self.initialize_weights()


    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        x = x.reshape(-1, 1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_vv(self, x, stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, _ = tpavi_b(x) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x
    def tpavi_vv_multiview(self, x,o,stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        o = self.pre_reshape_for_tpavi(o)
        x, _ = tpavi_b(x,o) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x
    def tpavi_va(self, x, audio, stage):
        # x: visual, [B*T, C=256, H, W]
        # audio: [B*T, 128]
        # ra_flag: return audio feature list or not
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        audio = audio.view(-1, 5, audio.shape[-1]) # [B, T, 128]
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x, a

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        """以view1为主模态，其他view作为辅助分割模态"""
        for view in self.view_num:
            x[view] = x[view].repeat(1,3,1,1)
        x1={}
        x2={}
        x3={}
        x4={}
        conv1_feat={}
        conv2_feat = {}
        conv3_feat = {}
        conv4_feat = {}
        for view in self.view_num:
            x[view] = self.resnet.conv1(x[view])
            x[view] = self.resnet.bn1(x[view])
            x[view] = self.resnet.relu(x[view])
            x[view] = self.resnet.maxpool(x[view])     # B x 64 x 21 x 21
            x1[view] = self.resnet.layer1(x[view])     # BF x 256  x 21 x 21
            x2[view] = self.resnet.layer2(x1[view])    # BF x 512  x 11 x 11
            x3[view] = self.resnet.layer3_1(x2[view])  # BF x 1024 x 6 x 6
            x4[view] = self.resnet.layer4_1(x3[view])  # BF x 2048 x  3 x  3
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        for view in self.view_num:
            conv1_feat[view] = self.conv1(x1[view])  # BF x 256 x 21 x 21
            conv2_feat[view] = self.conv2(x2[view])  # BF x 256 x 11 x 11
            conv3_feat[view] = self.conv3(x3[view])  # BF x 256 x 6 x 6
            conv4_feat[view] = self.conv4(x4[view])  # BF x 256 x 3 x 3

        feature_map_list={}
        for view in self.view_num:
            feature_map_list[view] = [conv1_feat[view],conv2_feat[view],conv3_feat[view],conv4_feat[view]]


        conv4_feat={}
        conv43={}
        conv432={}
        conv4321={}
        pred={}
        for view in self.view_num:
            conv4_feat[view] = self.path4(feature_map_list[view][3])            # BF x 2048 x  3 x  3
            conv43[view] = self.path3(conv4_feat[view], feature_map_list[view][2])    # BF x 256 x 6 x 6
            conv432[view] = self.path2(conv43[view], feature_map_list[view][1])       # BF x 256 x 11 x 11
            conv4321[view] = self.path1(conv432[view], feature_map_list[view][0])     # BF x 256 x 21 x 21
            # print(conv4_feat.shape, conv43.shape, conv432.shape, conv4321.shape)
            pred[view] = self.output_conv(conv4321[view])   # BF x 5 x 84 x 84
        # print(pred.shape)

        return pred,None


    def initialize_weights(self):
        res50 = models.resnet50(pretrained=False)
        resnet50_dict = torch.load(self.PRETRAINED_RESNET50_PATH)
        res50.load_state_dict(resnet50_dict)
        pretrained_dict = res50.state_dict()
        # print(pretrained_dict.keys())
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
        # self.resnet2.load_state_dict(all_params)
        print(f'==> Load pretrained ResNet50 parameters from {self.PRETRAINED_RESNET50_PATH}')



if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    model = Pred_endecoder(channel=256, tpavi_stages=[0,1,2,3], tpavi_va_flag=True)
    output = model(imgs)
    pdb.set_trace()
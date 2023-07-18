import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.view(b, c, -1).transpose(-1, -2)).transpose(-1, -2).view(y.size())

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def get_inplanes():
    return [64, 128, 256, 512, 1024]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def conv3x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(3, 1, 1),
                     stride=stride,
                     padding=(1, 0, 0),
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, k_size=3):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, k_size=3, head_conv=1):
        super().__init__()

        if head_conv == 1:
          self.conv1 = conv1x1x1(in_planes, planes)
        else:
          self.conv1 = conv3x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.eca = eca_layer(planes * self.expansion, k_size)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2,
                                       head_conv=1)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2,
                                       head_conv=1)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2,
                                       head_conv=1)
        '''
        self.downsample_block1 = self._make_layer_concat(BasicBlock,
                                                  block_inplanes[2] * 2,  
                                                  block_inplanes[2],
                                                  blocks=1,
                                                  stride=1,
                                                  downsample=nn.Sequential(
                                                                conv1x1x1(block_inplanes[2] * 2, block_inplanes[2], stride=1),
                                                                nn.BatchNorm3d(block_inplanes[2]))
                                                  )

        self.downsample_block2 = self._make_layer_concat(BasicBlock,
                                                  block_inplanes[3] * 2,   
                                                  block_inplanes[3],
                                                  blocks=1,
                                                  stride=1,
                                                  downsample=nn.Sequential(
                                                                conv1x1x1(block_inplanes[3] * 2, block_inplanes[3], stride=1),
                                                                nn.BatchNorm3d(block_inplanes[3]))
                                                  )

        self.downsample_block3 = self._make_layer_concat(BasicBlock,
                                                  block_inplanes[4] * 2,  
                                                  block_inplanes[4],
                                                  blocks=1,
                                                  stride=1,
                                                  downsample=nn.Sequential(
                                                                conv1x1x1(block_inplanes[4] * 2, block_inplanes[4], stride=1),
                                                                nn.BatchNorm3d(block_inplanes[4]))
                                                  )
        '''

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  head_conv=head_conv))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, head_conv=head_conv))

        return nn.Sequential(*layers)

    def _make_layer_concat(self, block, in_planes, out_planes, blocks, stride=1, downsample=None):
        if downsample is None:
            downsample = nn.Sequential(
                conv1x1x1(in_planes, out_planes, stride),
                nn.BatchNorm3d(out_planes))
        else:
            downsample = downsample

        layers = []
        layers.append(
            block(in_planes=in_planes,
                  planes=out_planes,
                  stride=stride,
                  downsample=downsample))
        in_planes = out_planes
        for i in range(1, blocks):
            layers.append(block(in_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        #x = self.interaction(x, self.downsample_block1)
        x = self.layer2(x)
        #x = self.interaction(x, self.downsample_block2)
        x = self.layer3(x)
        #x = self.interaction(x, self.downsample_block3)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        '''
        view_s1, view_s2 = x[::2], x[1::2]
        x = torch.cat([view_s1, view_s2], dim = -1)
        #x = x.view(x.size(0)//2, -1)
        x = self.fc(x)
        '''
        return x

    def interaction(self, x, downsample_layer):
        b,c,d,h,w = x.size()
        res = x
        
        x = rearrange(x, '(n m) c d h w -> n m c d h w', n = b//2)

        lateral = torch.cat((x[:, 1:], x[:, :1]), dim=1)
        x = torch.cat((x, lateral), dim=2)
        x = rearrange(x, 'n m c d h w -> (n m) c d h w')

        x = downsample_layer(x)

        return x + res

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

class Resnet50PAH(nn.Module):
    """Constructs a ResNet-50 model for FaceNet training using triplet loss.
    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, n_input_channels=1, n_output_channels=1, pretrained=False, checkpoint_path='/home/jwyang2/PFS-main/pretrained/r3d101_KM_200ep.pth'):
        super(Resnet50PAH, self).__init__()
        block_inplanes = get_inplanes()
        self.model = generate_model(model_depth=101, n_classes=1139)

        # layer modification
        self.conv1 = nn.Conv3d(n_input_channels,
               block_inplanes[0],
               kernel_size=(7, 7, 7),
               stride=(2, 2, 2),
               padding=(7 // 2, 3, 3),
               bias=False)
        '''
        self.projection = nn.Sequential(nn.Linear(block_inplanes[3] * 4 * 2, block_inplanes[3] * 4, bias=True),
                                        nn.Dropout(inplace=True),
                                        nn.BatchNorm1d(block_inplanes[3] * 4),
                                        #nn.ReLU(inplace=True),
                                        nn.Linear(block_inplanes[3] * 4, n_output_channels))
        '''
        self.projection = nn.Linear(block_inplanes[3] * 4, n_output_channels)
        #self.class_layers = nn.Linear(block_inplanes[3] * 4 * 2, 3)

        if pretrained:
            print("Use pretrained model, Pretrained model path is ----> {}".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            count = 0
            for k, v in checkpoint['state_dict'].items():
                if k in self.model.state_dict():
                    count += 1
                    self.model.state_dict()[k] = v
                    #print("update parameter name :{} -- size :{}".format(k, v.size()))
                else:
                    pass
            print("total updated parameters (layers) are ---> {} ".format(count))       
            #self.model.load_state_dict(checkpoint['state_dict'])
        self.model.conv1 = self.conv1
        self.model.fc = self.projection

        if pretrained is not True:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_out',
                                            nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, images):
        x = self.model(images)
        pred_x = self.projection(x)

        return pred_x

if __name__ == "__main__":

    num_classes = 2

    model = Resnet50PFS()

    input_tensor = torch.autograd.Variable(torch.rand(4, 2, 128, 128, 128))
    output = model(input_tensor)
    print(output.size())
    '''
    num_classes = 1139
    input_tensor = torch.autograd.Variable(torch.rand(8, 1, 128, 128, 128))
    model = generate_model(model_depth=50, n_classes=1139)
    output = model(input_tensor)
    print(output.size())
    '''
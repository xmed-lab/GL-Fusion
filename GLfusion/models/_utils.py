from collections import OrderedDict
from typing import Optional, Dict

from torch import nn, Tensor
from torch.nn import functional as F
import torch 


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None
    ) -> None:
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.aux_classifier = aux_classifier

        self.ctr_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ctr_fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        
        # print("self.backbone", self.backbone.conv1)
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        x_ctr = features["out"]
        x_ctr = self.ctr_avgpool(x_ctr)
        x_ctr = x_ctr.flatten(1)
        x_ctr = self.ctr_fc(x_ctr)
        # print("x_ctr.shape, in _utils segmentation", x_ctr.shape)
        result['ctr_feat'] = F.normalize(x_ctr, dim = 1)
        result['feat_mid'] = features["out"]
        return result







class _SimpleSegmentationModel_mltfrm(nn.Module):
    ##### modified for 1303_segmltfrm_prp. 
    ##### original 1303_segmltfrm was run with _SimpleSegmentationModel_mltfrm_spatatt
    __constants__ = ['aux_classifier']

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None
    ) -> None:
        super(_SimpleSegmentationModel_mltfrm, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.mlp_red = nn.Linear(4 * 2048, 2048)
        self.mlp_red =nn.Conv2d(4 * 2048, 2048, kernel_size=1, padding=0, stride=1, bias=False)
        # self.aux_classifier = aux_classifier


    def forward(self, x: Tensor, x_0: Tensor, x_1: Tensor, x_2: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        
        # https://link.springer.com/chapter/10.1007/978-3-030-87193-2_33

        x_features = self.backbone(x)["out"]

        x0_features = self.backbone(x_0)["out"]
        x1_features = self.backbone(x_1)["out"]
        x2_features = self.backbone(x_2)["out"]

        x0_dot = torch.einsum('bcl,bck->blk',x_features.view(x_features.shape[0], x_features.shape[1], -1), x0_features.view(x_features.shape[0], x_features.shape[1], -1))
        # print("x0_dot.shape", x0_dot.shape)
        x1_dot = torch.einsum('bcl,bck->blk',x_features.view(x_features.shape[0], x_features.shape[1], -1), x1_features.view(x_features.shape[0], x_features.shape[1], -1))
        x2_dot = torch.einsum('bcl,bck->blk',x_features.view(x_features.shape[0], x_features.shape[1], -1), x2_features.view(x_features.shape[0], x_features.shape[1], -1))

        x0_att = nn.functional.softmax(x0_dot.view(x_features.shape[0], -1)).view(x_features.shape[0], x0_dot.shape[1] , x0_dot.shape[2])
        x1_att = nn.functional.softmax(x1_dot.view(x_features.shape[0], -1)).view(x_features.shape[0], x1_dot.shape[1] , x1_dot.shape[2])
        x2_att = nn.functional.softmax(x2_dot.view(x_features.shape[0], -1)).view(x_features.shape[0], x2_dot.shape[1] , x2_dot.shape[2])

        # print("x0_att.shape", x0_att.shape)
        # print("x0_att.sum", x0_att.sum((-1,-2)))

        x0_att_feat = torch.einsum('bcl,blk->bck', x_features.view(x_features.shape[0], x_features.shape[1], -1), x0_att)
        x1_att_feat = torch.einsum('bcl,blk->bck', x_features.view(x_features.shape[0], x_features.shape[1], -1), x1_att)
        x2_att_feat = torch.einsum('bcl,blk->bck', x_features.view(x_features.shape[0], x_features.shape[1], -1), x2_att)
        
        

        feat_cncat = torch.cat((x_features, 
                                x0_att_feat.view(x_features.shape[0], x_features.shape[1], x_features.shape[2], x_features.shape[3]), 
                                x1_att_feat.view(x_features.shape[0], x_features.shape[1], x_features.shape[2], x_features.shape[3]), 
                                x1_att_feat.view(x_features.shape[0], x_features.shape[1], x_features.shape[2], x_features.shape[3])), dim = 1)

        feat_cmp = self.mlp_red(feat_cncat)

        result = OrderedDict()

        x = self.classifier(feat_cmp)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x
        return result




class _SimpleSegmentationModel_mltfrm_spatatt(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None
    ) -> None:
        super(_SimpleSegmentationModel_mltfrm, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.mlp_red = nn.Linear(4 * 2048, 2048)
        self.mlp_red =nn.Conv2d(4 * 2048, 2048, kernel_size=1, padding=0, stride=1, bias=False)
        # self.aux_classifier = aux_classifier


    def forward(self, x: Tensor, x_0: Tensor, x_1: Tensor, x_2: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        
        # https://link.springer.com/chapter/10.1007/978-3-030-87193-2_33

        x_features = self.backbone(x)["out"]

        x0_features = self.backbone(x_0)["out"]
        x1_features = self.backbone(x_1)["out"]
        x2_features = self.backbone(x_2)["out"]

        x0_dot = (x_features * x0_features).sum(1).unsqueeze(1)
        # print("x0_dot.shape", x0_dot.shape)
        x1_dot = (x_features * x1_features).sum(1).unsqueeze(1)
        x2_dot = (x_features * x2_features).sum(1).unsqueeze(1)

        x0_att = nn.functional.softmax(x0_dot.view(x_features.shape[0], -1)).view(x_features.shape[0], 1, x_features.shape[2], x_features.shape[3])
        x1_att = nn.functional.softmax(x1_dot.view(x_features.shape[0], -1)).view(x_features.shape[0], 1, x_features.shape[2], x_features.shape[3])
        x2_att = nn.functional.softmax(x2_dot.view(x_features.shape[0], -1)).view(x_features.shape[0], 1, x_features.shape[2], x_features.shape[3])

        # print("x0_att.shape", x0_att.shape)
        # print("x0_att.sum", x0_att.sum((-1,-2)))

        x0_att_feat = x0_att.expand(-1, 2048, -1, -1) * x_features
        x1_att_feat = x1_att.expand(-1, 2048, -1, -1) * x_features
        x2_att_feat = x2_att.expand(-1, 2048, -1, -1) * x_features

        feat_cncat = torch.cat((x_features, x0_att_feat, x1_att_feat, x2_att_feat), dim = 1)

        feat_cmp = self.mlp_red(feat_cncat)

        result = OrderedDict()

        x = self.classifier(feat_cmp)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x
        return result




class _SimpleSegmentationModel_iekd(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None
    ) -> None:
        super(_SimpleSegmentationModel_iekd, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=2)
        # self.aux_classifier = aux_classifier


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        
        # print("self.backbone", self.backbone.conv1)

        xtest = self.backbone.conv1(x)
        xtest = self.backbone.bn1(xtest)
        xtest = self.backbone.relu(xtest)
        xtest_layerbs = xtest
        xtest = self.backbone.maxpool(xtest)
        xtest_layer0 = xtest
        xtest = self.backbone.layer1(xtest)
        xtest_layer1 = xtest
        xtest = self.backbone.layer2(xtest) ### can just output here. 
        xtest_layer2 = xtest
        xtest = self.backbone.layer3(xtest)
        xtest = self.backbone.layer4(xtest)
        # print("xtest_layerbs.shape", xtest_layerbs.shape)# xtest_layerbs.shape torch.Size([2, 64, 56, 56])
        # print("xtest_layer0.shape", xtest_layer0.shape) #xtest_layer0.shape torch.Size([2, 64, 28, 28])
        # print("xtest_layer1.shape", xtest_layer1.shape) #xtest_layer1.shape torch.Size([2, 256, 28, 28])
        # print("xtest_layer2.shape", xtest_layer2.shape) #torch.Size([2, 512, 14, 14])

        result = OrderedDict()
        x = xtest

        x = self.classifier(x)
        x_maskpre = x
        x_maskpre = F.interpolate(x_maskpre, size=[56,56], mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x
        result['x_layerbs'] = xtest_layerbs
        result['x_layer1'] = xtest_layer1
        result['x_layer4'] = xtest
        result['maskfeat'] = x_maskpre
        return result





class _SimpleSegmentationModel_iekd_project(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None
    ) -> None:
        super(_SimpleSegmentationModel_iekd_project, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.aux_classifier = aux_classifier
        self.ctr_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cntr = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        
        # print("self.backbone", self.backbone.conv1)

        xtest = self.backbone.conv1(x)
        xtest = self.backbone.bn1(xtest)
        xtest = self.backbone.relu(xtest)
        xtest_layerbs = xtest
        xtest = self.backbone.maxpool(xtest)
        xtest_layer0 = xtest
        xtest = self.backbone.layer1(xtest)
        xtest_layer1 = xtest
        xtest = self.backbone.layer2(xtest) ### can just output here. 
        xtest_layer2 = xtest
        xtest = self.backbone.layer3(xtest)
        xtest = self.backbone.layer4(xtest)
        # print("xtest_layerbs.shape", xtest_layerbs.shape)# xtest_layerbs.shape torch.Size([2, 64, 56, 56])
        # print("xtest_layer0.shape", xtest_layer0.shape) #xtest_layer0.shape torch.Size([2, 64, 28, 28])
        # print("xtest_layer1.shape", xtest_layer1.shape) #xtest_layer1.shape torch.Size([2, 256, 28, 28])
        # print("xtest_layer2.shape", xtest_layer2.shape) #torch.Size([2, 512, 14, 14])

        result = OrderedDict()
        x_ctr_out = self.ctr_avgpool(xtest)
        x_ctr_out = x_ctr_out.flatten(1)
        x_ctr_out = self.cntr(x_ctr_out)
        x_ctr_out = F.normalize(x_ctr_out, dim = 1)
        x = xtest

        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x
        result['x_layerbs'] = xtest_layerbs
        result['x_layer1'] = xtest_layer1
        result['x_layer4'] = x_ctr_out.unsqueeze(-1).unsqueeze(-1)
        return result




class _SimpleSegmentationModel_iekd_maxmod(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: Optional[nn.Module] = None
    ) -> None:
        super(_SimpleSegmentationModel_iekd_maxmod, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # self.aux_classifier = aux_classifier
        self.coder = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.LeakyReLU(0.1))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        
        # print("self.backbone", self.backbone.conv1)

        xtest = self.backbone.conv1(x)
        xtest = self.backbone.bn1(xtest)
        xtest = self.backbone.relu(xtest)
        xtest_layerbs = xtest
        xtest = self.backbone.layer1(xtest)
        xtest_layer1 = xtest
        xtest = self.backbone.maxpool(xtest)
        xtest_layer1max = xtest
        xtest = self.backbone.layer2(xtest) ### can just output here. 
        xtest_layer2 = xtest
        xtest = self.backbone.layer3(xtest)
        xtest = self.backbone.layer4(xtest)
        # print("xtest_layerbs.shape", xtest_layerbs.shape)# xtest_layerbs.shape torch.Size([2, 64, 56, 56])
        # print("xtest_layer0.shape", xtest_layer0.shape) #xtest_layer0.shape torch.Size([2, 256, 56, 56])
        # print("xtest_layer1.shape", xtest_layer1.shape) #xtest_layer1.shape torch.Size([2, 256, 28, 28])
        # print("xtest_layer2.shape", xtest_layer2.shape) #torch.Size([2, 512, 14, 14])

        xtest_layer1code = self.coder(xtest_layer1)
        result = OrderedDict()
        x = xtest

        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x
        result['xtest_layer1code'] = xtest_layer1code
        return result
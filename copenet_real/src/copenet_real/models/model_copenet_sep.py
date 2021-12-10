import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import numpy as np
import math
from ..utils.geometry import rot6d_to_rotmat
import pytorch_lightning as pl

class Bottleneck(pl.LightningModule):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class copenet(pl.LightningModule):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super(copenet, self).__init__()
        npose = 21 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + 3 + 3 + 6 + npose + 10 + npose + 10, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, 3 + 6 + npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_feat_ext(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        return xf

        

class copenet_sep(pl.LightningModule):
    def __init__(self, block, layers, smpl_mean_params):
        super(copenet_sep, self).__init__()
        
        self.copenet0 = copenet(block,layers,smpl_mean_params)
        self.copenet1 = copenet(block,layers,smpl_mean_params)

    def forward(self, x0, x1,
                 bb0, bb1, 
                 init_position0, init_position1,
                 init_theta0=None, init_theta1=None, 
                 init_shape0=None, init_shape1=None, 
                 iters = 3):
        batch_size = x0.shape[0]

        
        if init_theta0 is None:
            init_orient0 = self.copenet0.init_pose[:,:6].expand(batch_size,-1)
            init_art_pose0 = self.copenet0.init_pose[:,6:22*6].expand(batch_size, -1)
        else:
            init_orient0 = init_theta0[:,:6].expand(batch_size,-1)
            init_art_pose0 = init_theta0[:,6:22*6].expand(batch_size, -1)
        if init_theta1 is None:
            init_orient1 = self.copenet1.init_pose[:,:6].expand(batch_size,-1)
            init_art_pose1 = self.copenet1.init_pose[:,6:22*6].expand(batch_size, -1)
        else:
            init_orient1 = init_theta1[:,:6].expand(batch_size,-1)
            init_art_pose1 = init_theta1[:,6:22*6].expand(batch_size, -1)
        if init_shape0 is None:
            init_shape0 = self.copenet0.init_shape.expand(batch_size, -1)
        if init_shape1 is None:
            init_shape1 = self.copenet1.init_shape.expand(batch_size, -1)

        
         # Feed images in the network to predict camera and SMPL parameters 
        xf0 = self.copenet0.forward_feat_ext(x0)
        xf1 = self.copenet1.forward_feat_ext(x1)


        pred_pose0, pred_betas0, pred_pose1, pred_betas1 = self.forward_reg(xf0, xf1,
                                                bb0, bb1,
                                                init_position0, init_position1,
                                                init_orient0, init_orient1, 
                                                init_art_pose0, init_art_pose1,
                                                init_shape0, init_shape1)

        for it in range(int(iters)-1):
            pred_pose0, pred_betas0, pred_pose1, pred_betas1 = self.forward_reg(xf0, xf1,
                                                    bb0, bb1,
                                                    pred_pose0[:,:3], pred_pose1[:,:3],
                                                    pred_pose0[:,3:9], pred_pose1[:,3:9],
                                                    pred_pose0[:,9:], pred_pose1[:,9:],
                                                    pred_betas0, pred_betas1)

        return pred_pose0, pred_betas0, pred_pose1, pred_betas1

    def forward_reg(self, xf0, xf1,
                    bb0, bb1,
                    pred_position0, pred_position1,
                    pred_orient0, pred_orient1, 
                    pred_art_pose0, pred_art_pose1,
                    pred_shape0, pred_shape1):
        
        xc0 = torch.cat([xf0, bb0, pred_position0, pred_orient0, pred_art_pose0, pred_shape0, pred_art_pose1, pred_shape1],1)
        xc0 = self.copenet0.fc1(xc0)
        xc0 = self.copenet0.drop1(xc0)
        xc0 = self.copenet0.fc2(xc0)
        xc0 = self.copenet0.drop2(xc0)
        
        pred_shape0 = pred_shape0 + self.copenet0.decshape(xc0)
        pred_pose0 = torch.cat([pred_position0, pred_orient0, pred_art_pose0],1) + self.copenet0.decpose(xc0)
        
        xc1 = torch.cat([xf1, bb1, pred_position1, pred_orient1, pred_art_pose1, pred_shape1, pred_art_pose0, pred_shape0],1)
        xc1 = self.copenet1.fc1(xc1)
        xc1 = self.copenet1.drop1(xc1)
        xc1 = self.copenet1.fc2(xc1)
        xc1 = self.copenet1.drop2(xc1)

        pred_shape1 = pred_shape1 + self.copenet1.decshape(xc1)
        pred_pose1 = torch.cat([pred_position1, pred_orient1, pred_art_pose1],1) + self.copenet1.decpose(xc1)

        return pred_pose0, pred_shape0, pred_pose1, pred_shape1
        
        # pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

    def load_from_checkpoint(self,path):
        import ipdb; ipdb.set_trace()
        try:
            super(copenet_sep,self).load_from_checkpoint(path)
        except:
            print("separated models loading...")
            super(copenet_sep,self.copenet0).load_from_checkpoint(path)
            super(copenet_sep,self.copenet1).load_from_checkpoint(path)
            

def getcopenet(smpl_mean_params, pretrained=True, **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = copenet_sep(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    if pretrained:
        resnet_imagenet = resnet.resnet50(pretrained=True)
        model.copenet0.load_state_dict(resnet_imagenet.state_dict(),strict=False)
        model.copenet1.load_state_dict(resnet_imagenet.state_dict(),strict=False)
    return model
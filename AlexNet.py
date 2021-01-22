#Author:Sherlock Novitch
# -*- codeing=utf-8 -*-
# @time:2021/1/19 18:06
# @Author:ASUS
# @File:AlexNet.py
# @Software:PyCharm

import torch.nn as nn
import torch
from collections import OrderedDict

class AlexNet(nn.Module):
    def __init__(self,num_inputs,num_classes):
        super(AlexNet,self).__init__()
        self.net1=nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(num_inputs,96,kernel_size=11,stride=4,padding=2)),   #卷积层1
            ('relu1',nn.ReLU(inplace=True)), #直接计算，不拷贝，节省内存与时间
            ('LRN1',nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2)),     #LRN层1
            ('pool1',nn.MaxPool2d(kernel_size=3,stride=2)),                        #池化层1
            ('conv2',nn.Conv2d(96,256,kernel_size=5,padding=2)),                  #卷积层2
            ('relu2',nn.ReLU(inplace=True)),
            ('LRN2',nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)),  #LRN层2
            ('pool2',nn.MaxPool2d(kernel_size=3,stride=2)),                      #池化层2
            ('conv3',nn.Conv2d(256,384,kernel_size=3,padding=1)),                  #卷积层3
            ('relu3',nn.ReLU(inplace=True)),
            ('conv4',nn.Conv2d(384,384,kernel_size=3,padding=1)),                 #卷积层4
            ('relu4',nn.ReLU(inplace=True)),
            ('conv5',nn.Conv2d(384,256,kernel_size=3,padding=1)),                  #卷积层5
            ('relu5',nn.ReLU(inplace=True)),
            ('pool3',nn.MaxPool2d(kernel_size=3,stride=2))                        #池化层3
        ]))
        self.net2=nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),                      #全连接层1 dropout
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),                      #全连接层2 dropout
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)              #全连接层3
        )
        #依照论文复现参数初始化
        for layer in self.net1:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight.data, 0, 0.01)
                torch.nn.init.constant_(layer.bias.data, 0.0)
        for layer in self.net2:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight.data, 0, 0.01)
                torch.nn.init.constant_(layer.bias.data, 1.0)
        torch.nn.init.constant_(self.net1[4].bias.data, 1.0)
        torch.nn.init.constant_(self.net1[10].bias.data, 1.0)
        torch.nn.init.constant_(self.net1[12].bias.data, 1.0)

    def forward(self,x):
        #print(x.size())
        out=self.net1(x)
        #print(out.size())
        out=out.view(-1,9216)
        out=self.net2(out)
        return out

#初始化不同
class AlexNet1(nn.Module):
    def __init__(self,num_inputs,num_classes):
        super(AlexNet1,self).__init__()
        self.net1=nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(num_inputs,96,kernel_size=11,stride=4,padding=2)),   #卷积层1
            ('relu1',nn.ReLU(inplace=True)), #直接计算，不拷贝，节省内存与时间
            ('LRN1',nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2)),     #LRN层1
            ('pool1',nn.MaxPool2d(kernel_size=3,stride=2)),                        #池化层1
            ('conv2',nn.Conv2d(96,256,kernel_size=5,padding=2)),                  #卷积层2
            ('relu2',nn.ReLU(inplace=True)),
            ('LRN2',nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)),  #LRN层2
            ('pool2',nn.MaxPool2d(kernel_size=3,stride=2)),                      #池化层2
            ('conv3',nn.Conv2d(256,384,kernel_size=3,padding=1)),                  #卷积层3
            ('relu3',nn.ReLU(inplace=True)),
            ('conv4',nn.Conv2d(384,384,kernel_size=3,padding=1)),                 #卷积层4
            ('relu4',nn.ReLU(inplace=True)),
            ('conv5',nn.Conv2d(384,256,kernel_size=3,padding=1)),                  #卷积层5
            ('relu5',nn.ReLU(inplace=True)),
            ('pool3',nn.MaxPool2d(kernel_size=3,stride=2))                        #池化层3
        ]))
        self.net2=nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),                      #全连接层1 dropout
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),                      #全连接层2 dropout
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)              #全连接层3
        )
    def forward(self,x):
        #print(x.size())
        out=self.net1(x)
        #print(out.size())
        out=out.view(-1,9216)
        out=self.net2(out)
        return out

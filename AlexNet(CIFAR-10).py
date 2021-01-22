#Author:Sherlock Novitch
# -*- codeing=utf-8 -*-
# @time:2021/1/20 9:16
# @Author:ASUS
# @File:AlexNet(CIFAR-10).py
# @Software:PyCharm


import torchvision .datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch
import torch.nn as nn
from torch import optim
import AlexNet

torch.manual_seed=1  #设置随机数种子
input_size=32*32

num_class=10
num_epoch=5
batch_size=100
lr=0.001

train_dataset=dsets.CIFAR10(root='./CIFAR-10数据',train=True,
                          transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]),download=True)
test_dataset=dsets.CIFAR10(root='./CIFAR-10数据',train=False,transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))

train_loader=Data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=Data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)



use_gpu=torch.cuda.is_available()


net=AlexNet.AlexNet1(3,num_class)
net.load_state_dict(torch.load('Alex-CIFAR-10.pkl'))
if use_gpu:
    net=net.cuda()
print(net)
for parm in net.parameters():
    print(parm.data)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)  #论文原超参数

for epoch in range(num_epoch):
    running_loss = 0.0
    running_acc = 0.0
    for i,data in enumerate(train_loader):
        img,label=data
        img.require_grad=True
        if use_gpu:
            img,label=img.cuda(),label.cuda()
        out=net(img)
        loss=criterion(out,label)
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train{} epoch,Loss:{:.6f},Acc:{:.6f}'.format
          (epoch + 1, running_loss / len(train_dataset), running_acc / len(train_dataset)))

torch.save(net.state_dict(),'Alex-CIFAR-10.pkl')



net.eval()
eval_loss=0
eval_acc=0
for data in test_loader:
    img,label=data
    if use_gpu:
        img,label=img.cuda(),label.cuda()
    out=net(img)
    loss=criterion(out,label)
    eval_loss+=loss.data.item()*label.size(0)
    _,pred=torch.max(out,1)
    num_correct=(pred==label).sum()
    eval_acc+=num_correct.data.item()
print('Test Loss:{:.6f} ,Acc:{:.6f}'.format(eval_loss/len(test_dataset),eval_acc/len(test_dataset)))
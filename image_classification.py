# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as Data
import numpy as np
import os
import argparse
from resnet import ResNet18
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")
# data processing
class train_dataset(torch.utils.data.Dataset): #自定义数据集
    def __init__(self, transform=None):
        images = ((np.load("./train.npy") / 255.0 - 0.2926) / 0.3344).astype(np.float32)
        b, _ = images.shape
        images = images.reshape(b, 28, 28, 1)
        labels = open("./train.csv").readlines()
        if labels[0] == 'image_id,label\n':
            del labels[0]
            self.label = labels
        for i in range(len(labels)):
            items = labels[i].strip('\n').split(',')
            self.label[i] = items[1]
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        # label = np.eye(10)[int(self.label[index])].astype(np.int64)
        label = np.array(self.label[index]).astype(np.int64)
        if self.transform is not None:
            img = self.transform(img)
            # label = self.transform(label)
        return (img, label)

    def __len__(self):
        return len(self.images)


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        images = ((np.load("./test.npy") / 255.0 - 0.2926) / 0.3344).astype(np.float32)
        b, _ = images.shape
        images = images.reshape(b, 28, 28, 1)
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        return (img)

    def __len__(self):
        return len(self.images)


BATCH_SIZE = 32
trainset = train_dataset(transform=transforms.Compose([   #数据增强
    transforms.ToPILImage(mode=None),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
]))

testset = test_dataset(transform=transforms.Compose([
    transforms.ToPILImage(mode=None),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
]))

trainset, valset = torch.utils.data.random_split(trainset, [25000, 5000])  #随机5：1划分训练集和验证集
print('trainsset:', len(trainset), 'valset:', len(valset))

train_loader = Data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = Data.DataLoader(dataset=valset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")
args = parser.parse_args()

#超参设置
EPOCH = 40
pre_epoch = 0
LR = 0.003

# net = ResNet18().to(device)  #是否用cude加速

#采用现成的resnet18网络进行训练
net = models.resnet18(pretrained=False)
num_fc_ftr = net.fc.in_features
net.fc = torch.nn.Linear(num_fc_ftr, 224)
net = net.to(device)
'''
checkpoint= torch.load('/home/wangcf/jiangzhuo/model/net_020.pth')
net.load_state_dict(checkpoint)
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

#开始训练
if __name__ == "__main__":
    best_acc = 87.5   
    print("Start Training, Resnet-18!")  
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                if epoch >10:
                    LR = 0.0003
                if epoch >20:
                    LR = 0.00003
                if epoch >30:
                    LR = 0.000003

                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_loader, 0):

                    length = len(train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()


                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()


                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cuda().sum()

                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.03f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),  100 * (correct. float() /  total)))


                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.03f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100 *( correct. float() /  total)))

                    f2.write('\n')
                    f2.flush()

                #验证过程
                print("Waiting val!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for i, data in enumerate(val_loader, 0):
                        net.eval()
                        images,labels = data
                        #labels=np.zeros(data.shape[0])
                        #labels=torch.from_numpy(labels)
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('Accuracy: %.3f%%' % (100 * ( correct .float() /  total  )))
                    acc = 100 * ( correct .float() /  total  )
                    print('Saving model......')
                    torch.save(net.state_dict(), './model/net_%03d.pth' %  (epoch + 1))

                    f.write("EPOCH=%03d,Accuracy= %.03f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()

                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc

            print("Training Finished, TotalEPOCH=%d" % EPOCH)

#checkpoint = torch.load('./model/net_037.pth')
#net.load_state_dict(checkpoint)

#生成预测csv
pred_all = None
label_all = None
pre_pro_all = None
net.eval()
with torch.no_grad():
    for X in test_loader:
        X = Variable(X).cuda()
        # label = Variable(label)  # .cuda()

        testout = net(X)
        _, pred = testout.max(1)
        if pred_all is None:
            pred_all = torch.cat([pred])
        else:
            pred_all = torch.cat([pred_all, pred])

    y_pred = pred_all.cpu().detach().numpy()
    y_pred = y_pred.astype(np.int32)

    pred = pd.DataFrame(y_pred)
    pred.columns = ['label']
    pred.index.name = 'image_id'
    pred.to_csv('./submit.csv', index=True)

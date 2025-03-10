from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transform
from csv_data_loader import CsvDataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
classes =['ant','bee']

train_path = '/home/yao/workspace/learn_pytorch/train.csv'
test_path = '/home/yao/workspace/learn_pytorch/val.csv'
trans = transform.Compose([transform.RandomResizedCrop(224),   # 最小边resize尺寸,非全部
                           transform.ToTensor()])
trainset = CsvDataset(train_path,transform=trans) #  datalaoder要求接受tensor

#图像不进行resize且size不同时，batchsize只能为1，否则报RuntimeError: 
# invalid argument 0: Sizes of tensors must match except in dimension 0. Got xxx and xxx in dimension 1
batch_size = 4
trainsetloader = DataLoader(trainset,batch_size=batch_size,shuffle=False,num_workers=0)

for i,data in enumerate(trainsetloader,0):   #用for循环访问所有数据，只用iter和next 最后会报错
    imgs,labels,_ = data
    batch_imgs = torchvision.utils.make_grid(imgs)
    batch_imgs = batch_imgs.numpy()
    #plt.imshow(batch_imgs.transpose((1,2,0)))
    batch_imgs =batch_imgs.transpose((1, 2, 0))[:, :, (2, 1, 0)]
    for i in range(imgs.shape[0]):
        step=imgs.shape[2]
        # add to solve:https://blog.csdn.net/u013685264/article/details/121661895
        batch_imgs = batch_imgs.copy()
        batch_imgs = cv2.putText(batch_imgs,classes[labels[i].item()],(i*step,20),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color =(0,0,255),
                    thickness=2,lineType=cv2.LINE_AA)
                    
        # print(batch_imgs)
    cv2.imshow("", batch_imgs)
    k = cv2.waitKey(0)
    if((k)==ord('q')):
        break
    else:
        continue


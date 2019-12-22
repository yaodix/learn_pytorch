from torch.utils.data import Dataset,DataLoader
import pandas as pd
from skimage import io
import os
import torch
import numpy as np
from PIL import Image
import cv2


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def cv2_loader(img_path):
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    img = img[:, :, (2, 1, 0)]
    # img = img.transpose((2,0,1))
    return  img

class CSVDataLoader(Dataset):
    def __init__(self,file_path,transform = None):
        self.file_path = file_path
        self.data_info = pd.read_csv(self.file_path,header=None)
        # 文件第一列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # 第二列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.batch_paths=[]
        # 计算 length
        self.data_len = len(self.data_info.index)
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):  #转换为numpyk可用的list
            idx = idx.tolist()
        img_path = self.image_arr[idx]
        label = self.label_arr[idx]
        try:
            img = pil_loader(img_path)   #cv_load报错
            if self.transform is not None:
                img = self.transform(img)
        except:
            print('in __getitem__ exception occured',img_path)
        return (img,label,img_path)



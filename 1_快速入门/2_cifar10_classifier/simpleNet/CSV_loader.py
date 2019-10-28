import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import  Dataset


class CSVLoader(Dataset):
    def __init__(self,csv_file,transform=None):
        self.csv_file =csv_file
        self.file_data = pd.read
        self.transform = transform
    def __len__(self):
        return len(self.file_data)

    def __getitem__(self, item):
        img_name = self.file_data[item][0]
        img_label = self.file_data[item][1]
        img = cv2.imdecode(np.fromfile(img_name,dtype=np.uint8),cv2.IMREAD_COLOR)

        if self.transform is not None:
            img = self.transform(img)

        return [img,img_label]

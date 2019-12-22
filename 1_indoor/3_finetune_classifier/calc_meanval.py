import pandas as pd
import cv2
import numpy as np
import glob
#read from csv
'''
train_path = 'C:\\pythonProjects\\learn_pytorch\\1_indoor\\3_finetune_classifier\\train.csv'
data_info = pd.read_csv(train_path)
image_paths = np.asarray(data_info.iloc[:,0])
'''
#read form image folder
root_path = "C:\\MyData\\AI_game\\VOC2012\\JPEGImages\\*.jpg"
image_paths = glob.glob(root_path)

RGB_order = True
means=[0,0,0]
stdevs=[0,0,0]

print('image size', len(image_paths))
for p in image_paths:
    img = cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR )  #BGR格式
    if RGB_order:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    for i in range(3):
        means[i] += img[:, :, i].mean()
        stdevs[i] += img[:, :, i].std()



means = np.asarray(means)  / len(image_paths)
stdevs =np.asarray(stdevs)  / len(image_paths)

print('RGB_order',RGB_order)
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
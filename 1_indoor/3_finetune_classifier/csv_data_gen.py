
#单一路径图片文件夹生成训练文件
#
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
import cv2

train_X = []
train_y = []

# set folder path
folderpath1 = 'C:\\MyData\\hymenoptera_data\\ant\\'
folderpath2= 'C:\\MyData\\hymenoptera_data\\bee\\'
classes = ['ant','bee']

#额外设置，如限制尺寸,小于一定尺寸的只放在测试集中
min_size = 224
test_X = []
test_y = []

def cv2_load(path):
    img = cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
    shape = img.shape

    return img,shape
# load image arrays
for filename in os.listdir(folderpath1):
    if filename != '.DS_Store':
        new_file_name = filename.replace(" ", "_")
        new_file_name = new_file_name.replace("（", "(")
        new_file_name = new_file_name.replace("）", "")

        imagepath = folderpath1 + filename
        new_imagepath = folderpath1 + new_file_name

        if (new_file_name != filename):
            os.rename(imagepath, new_imagepath)
        _,shape = cv2_load(new_imagepath)
        if shape[0] <min_size or shape[1]<min_size:
            test_X.append(new_imagepath)
            test_y.append(0)
        else:
            train_X.append(new_imagepath)
            train_y.append(0)
    else:
        print(filename, 'not a pic')

# load image arrays
for filename in os.listdir(folderpath2):
    if filename != '.DS_Store':  #DS_Store是Mac OS保存文件夹的自定义属性的隐藏文件
        new_file_name = filename.replace(" ", "_")
        new_file_name = new_file_name.replace("（", "(")
        new_file_name = new_file_name.replace("）", "")

        #if 'blur2ok' in
        imagepath = folderpath2 + filename
        new_imagepath = folderpath2 + new_file_name

        if (new_file_name != filename):
            os.rename(imagepath, new_imagepath)
        _,shape = cv2_load(new_imagepath)
        if shape[0] <min_size or shape[1]<min_size:
            test_X.append(new_imagepath)
            test_y.append(1)
        else:
            train_X.append(new_imagepath)
            train_y.append(1)
    else:
        print(filename, 'not a pic')

print('small size pic:', len(test_X))
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.37)

X_test.extend(test_X)
y_test.extend(test_y)
#移动
'''
cnt_label = [0,0]
for i in  range(classes.__sizeof__()):
    cnt_label[i]= np.sum(i == test_y)
for cnt ,i in enumerate(cnt_label,0):
    pass
'''

count = len(X_train)
cls_train_cnt =np.array([0,0])
cls_test_cnt =np.array([0,0])
f = open('train.csv', 'w')
for i in range(count):
    line = '%s,%d' % (X_train[i], y_train[i])  # %s %d'表示格式，%号分隔，它代表了格式化操作
    cls_train_cnt[ y_train[i]] +=1
    f.write(line)
    if (i < count - 1):
        f.write("\n")

f.close()

print('X_train: ', len(X_train))
print(' ant: ',cls_train_cnt[0])
print(' bee: ',cls_train_cnt[1])
print('Gen train.txt Done!')

count = len(X_test)
f = open('val.csv', 'w')
for i in range(count):
    line = '%s,%d' % (X_test[i], y_test[i])
    cls_test_cnt[ y_test[i]] +=1
    f.write(line)

    if (i < count - 1):
        f.write("\n")

f.close()
print('X_test: ', len(X_test))
print(' ant: ',cls_test_cnt[0])
print(' bee: ',cls_test_cnt[1])
print('Gen val.txt Done!')
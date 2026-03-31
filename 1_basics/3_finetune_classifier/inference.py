import torchvision
import torch
from torch.utils.data import DataLoader
import numpy as np
from csv_data_loader import CSVDataLoader
from torchvision.transforms import transforms
import torchvision.models  as models
import time
import matplotlib.pyplot as plt
import cv2
import shutil
import  os
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

def conf_mat(y_true,y_pred,labels):
    c = confusion_matrix(y_true,y_pred,labels)
    sns.heatmap(c,annot=True,fmt='d', cmap="YlGnBu",xticklabels=classes,yticklabels=classes)
    plt.show()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    img = inp.copy()
    img = (img*255).astype(np.uint8)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    return img.transpose((2,0,1))
def normImg2raw(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    img = inp.copy()
    img = (img*255).astype(np.uint8)
    return img[:,:,(2,1,0)]

classes =['ant','bee']
device = torch.device('cuda:0')
test_path = 'C:\\pythonProjects\\learn_pytorch\\1_indoor\\3_finetune_classifier\\val.csv'
Path = 'C:\\pythonProjects\\learn_pytorch\\1_indoor\\3_finetune_classifier\\model_weights\\epoch_20_0.9671.pt'
save_res = 'C:\\MyData\\hymenoptera_data\\wrong\\'

test_trans = transforms.Compose([transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.460],[0.229,0.224,0.225])
                           ])


testset = CSVDataLoader(test_path,test_trans)
testdataloader = DataLoader(testset,batch_size=4,shuffle=False)

model_ft  =models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs,2)

model_ft.load_state_dict(torch.load(Path))
model_ft.to(device)
model_ft.eval()          # key point

y_true =[]
y_pred = []
with torch.no_grad():   # key point
    for i,data in enumerate(testdataloader,0):
        inputs,labels,paths =data
        inputs =inputs.to(device)
        labels =labels.to(device)
        outputs = model_ft(inputs)
        probs = outputs.softmax(1)

        prob,preds = torch.max(probs,1)
        y_pred.extend(preds.cpu().tolist())
        y_true.extend(labels.cpu().tolist())

        res = (preds == labels)
        if False :
            for j,value in enumerate(res,0) :
                if value.item() is False:
                    img = inputs[j].cpu()
                    save_img = normImg2raw(img)
                    cv2.imshow('win',save_img)

                    print(paths[j],',pred:',classes[preds[j]],',prob:',prob.cpu()[j].item())
                    _,name = os.path.split(paths[j])
                    shutil.copy(paths[j],save_res+name)
                    cv2.waitKey()

trues = np.where(np.array(y_true) == 0, 'ant','bee')
preds = np.where(np.array(y_pred) == 0, 'ant','bee')
conf_mat(y_pred=preds,y_true=trues,labels=classes)
pass








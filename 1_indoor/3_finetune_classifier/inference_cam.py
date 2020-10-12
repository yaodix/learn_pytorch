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
import grad_cam

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
testdataloader = DataLoader(testset,batch_size=1,shuffle=True)

model_ft  =models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs,2)

model_ft.load_state_dict(torch.load(Path))

cam_vis = grad_cam.GradCam()
model_ft.layer4.register_forward_hook(cam_vis.farward_hook)
model_ft.layer4.register_backward_hook(cam_vis.backward_hook)

model_ft.to(device)
model_ft.eval()          # key point

for i,data in enumerate(testdataloader,0):
    inputs,labels,paths =data
    inputs =inputs.to(device)
    labels =labels.to(device)

    outputs = model_ft(inputs)

    probs = outputs.softmax(1)
    prob,preds = torch.max(probs,1)
    print('cls is',classes[preds])
    model_ft.zero_grad()
    cls_loss = cam_vis.comp_class_vec(outputs)
    cls_loss.backward()

    # 生成cam
    grads_val = cam_vis.grad_block[0].cpu().data.numpy().squeeze()
    fmap = cam_vis.fmap_block[0].cpu().data.numpy().squeeze()
    cam = cam_vis.gen_cam(fmap, grads_val)

    # 保存cam图片
    output_dir = 'C:\\files'
    img = (inputs.cpu()[0])
    img = normImg2raw(img)

    img_show = np.float32(cv2.resize(img, (224, 224))) / 255
    cam_vis.show_cam_on_image(img_show, cam, output_dir)
    break
pass








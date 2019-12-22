
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import numpy as np
from simpleNet.net import SimpleNet
from torch.utils.tensorboard import SummaryWriter

transform = torchvision.transforms.Compose([transform.ToTensor(),transform.Normalize((0.5,),(0.5,))])
trainset = torchvision.datasets.FashionMNIST('./data',train=True,download=True,transform=transform)

testset = torchvision.datasets.FashionMNIST('./data',train=False,transform=transform,download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset,batch_size =4,shuffle = True,num_workers=0)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataite= iter(trainloader)
imgs,label  = dataite.next()
img = torchvision.utils.make_grid(imgs)
net = SimpleNet()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

writer = SummaryWriter('runs/fashion_mnist_experiment_2')
running_loss = 0
for epoch in range(1):
    for i,data in enumerate(trainloader,0):
        inputs,labels = data
        optim.zero_grad()

        outputs = net(inputs)
        #_,preds = torch.max(outputs,1)
        loss = criterion(outputs,labels)  #outputs' size is 4*10 ,labels is 4
        loss.backward()
        optim.step()

        running_loss += loss.item()
        if i%1000==999:
            writer.add_scalar('training loss',running_loss/1000,epoch*len(trainloader)+i)
            running_loss=0

print('finish trainning')


class_probs = []
class_preds = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# plot all the pr curves
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)
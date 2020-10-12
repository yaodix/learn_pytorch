import torchvision
import torch
from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report
from csv_data_loader import CSVDataLoader
from torchvision.transforms import transforms
import torchvision.models  as models
import time
import torch.optim
# from torch.utils.tensorboard import  SummaryWriter

classes =['ant','bee']
train_path = 'C:\\pythonProjects\\learn_pytorch\\1_indoor\\3_finetune_classifier\\train.csv'
test_path = 'C:\\pythonProjects\\learn_pytorch\\1_indoor\\3_finetune_classifier\\val.csv'


train_trans = transforms.Compose([
                            transforms.RandomCrop(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.460],[0.229,0.224,0.225])
                           ])
test_trans = transforms.Compose([transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.460],[0.229,0.224,0.225])
                           ])
trainset = CSVDataLoader(train_path,transform=train_trans)
testset = CSVDataLoader(test_path,transform=test_trans)
device = torch.device("cuda:0")

traindataloader = DataLoader(trainset,batch_size=32,shuffle=True,num_workers=0)
testdataloader = DataLoader(testset,batch_size=4,shuffle=True,num_workers=0)

def train_model(model,criterion,optim,scheduler,num_epochs=25):
    since = time.time()
    best_val_acc= 0
    # writer = SummaryWriter('logs/')
    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch,num_epochs-1))
        #train
        model.train()   #启用 BatchNormalization 和 Dropout
        running_loss = 0
        running_corrects = 0
        for i,data in enumerate(traindataloader,0):
            imgs,labels,_= data
            imgs = imgs.to(device)
            labels = labels.to(device)
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(imgs)
                _,preds = torch.max(outputs,1)
                loss = criterion(outputs,labels)
                loss.backward()
                optim.step()
            running_loss +=loss.item()*imgs.size(0)
            running_corrects +=torch.sum(preds == labels.data)  #.data
        scheduler.step()
        epoch_loss =running_loss/trainset.__len__()
        epoch_acc = running_corrects.double()/trainset.__len__()
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

          #val
        model.eval()
        testing_loss=0
        testing_corrects = 0
        for i,data in enumerate(testdataloader,0):
            imgs,labels,_ = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(imgs)
                _,preds = torch.max(outputs,1)
                loss = criterion(outputs,labels)

            testing_loss +=loss.item()*imgs.size(0)
            testing_corrects +=torch.sum(preds == labels.data)  #.data

        val_loss =testing_loss/testset.__len__()
        val_acc = testing_corrects.double()/testset.__len__()
        # writer.add_scalars('epoch/loss',{'train':epoch_loss,'valid':val_loss},epoch)
        # writer.add_scalars('epoch/acc',{'train':epoch_acc,'valid':val_acc},epoch)

        #记录权值分布
        # for name,layer in model.named_parameters():
        #     writer.add_histogram(name+'_grad',layer.grad.cpu().data.numpy(),epoch)
        #     writer.add_histogram(name+'_data',layer.cpu().data.numpy(),epoch)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format( 'val', val_loss, val_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        Path = 'model_weights/epoch_{}_{:.4f}.pt'.format(epoch,val_acc)
        # torch.save(model.state_dict(),Path)
    print('best_val_acc:{}',best_val_acc)
    # writer.close()
model_ft  =models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs,2)
model_ft = model_ft.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.fc.parameters(),lr=0.001,momentum=0.9)  #  训练参数
exp_lr_schedules = torch.optim.lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.5)

model_ft = train_model(model_ft,criterion,optimizer_ft,exp_lr_schedules,25)





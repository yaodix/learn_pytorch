import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
plt.ion()
#prepare data
x = np.arange(0,5,0.2)
rand1 = np.random.randint(-2,2,(x.size))
rand2 = np.random.randint(-2,2,(x.size))

y1 = 2*x+2+rand1
y2 = 2*x+6 +rand2
plt.figure()

plt.scatter(x,y1,c='r')
plt.scatter(x,y2,c = 'b')
#plt.show()

x_p = torch.from_numpy(x).float()
y1_p = torch.from_numpy(y1).float()
y2_p = torch.from_numpy(y2).float()

x_t = torch.cat((x_p,x_p))
y_t = torch.cat((y1_p,y2_p))
x_t = x_t.unsqueeze(0)
y_t = y_t.unsqueeze(0)

data = torch.cat((x_t,y_t))
data = data.transpose(1,0)

label = torch.zeros(data.size()[0])
label[int(data.size()[0]/2):] =1
label = label.unsqueeze(1)


# model and loss
_w = nn.Parameter(torch.randn(2,1))
_b = nn.Parameter(torch.randn(1))
def logistic(x):
    return torch.sigmoid((torch.mm(x,_w)+_b))   #此处‘x'的表达方式

#train
step = 5e-2
optimizer = optim.SGD([_w,_b],step,momentum=0.9)
for epoch in range(0,1500):
    y_pred = logistic(data)
    loss = F.binary_cross_entropy(y_pred,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch = {},loss = {}'.format(epoch,loss.item()))
    plt.clf()
    plt.scatter(x, y1, c='r')
    plt.scatter(x, y2, c='b')
    w = _w.detach().numpy()
    b = _b.detach().numpy()
    y = -(w[0]*x+b)/w[1]
    #print('k ={} b ={} '.format(-w[0] / w[1], -b / w[1]))
    plt.plot(x,y,'-g')
    plt.pause(0.01)
    plt.show()

plt.ioff()
plt.scatter(x, y1, c='r')
plt.scatter(x, y2, c='b')
w = _w.detach().numpy()
b = _b.detach().numpy()
y = -(w[0] * x + b) / w[1]
print('k ={} b ={} '.format(-w[0]/w[1],-b/w[1]))
plt.plot(x, y, '-g')
plt.show()






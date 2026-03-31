
#功能： 圆拟合
import torch
import  numpy as np
import matplotlib.pyplot as plt

plt.ion()   #打开交互模式

x = (np.arange(1,5,0.1))
y1 = np.sqrt(4-(x-3)**2)+4
y2 = -(np.sqrt(4-(x-3)**2)-4)
plt.figure()
x_ = np.append(x,x)
y_ = np.append(y1,y2)
#plt.scatter(x_train,y_train)

x_train = torch.from_numpy(x_).float()
y_target = torch.from_numpy(y_).float()
#print('x_train type ={} '.format(x_train.dtype))
_w = torch.randint(1,4,(2,)).float() #x0 y0
_r2 = torch.randint(1,4,(1,)).float()
_w.requires_grad = True
_r2.requires_grad = True
#print('_w type ={} '.format(_w.dtype))
step = 1e-2
def circle_func(x_):
    x=x_[:40]
    val =_r2-(x-_w[0])**2
    com = torch.full_like(val,0.001)
    val_smooth = torch.max(val,com)
    _y1 =   torch.sqrt(val_smooth)+_w[1]   #max for smooth !!
    _y2 = -(torch.sqrt(val_smooth)-_w[1])
    return torch.cat((_y1,_y2),0)

def loss_func(y_train,_y):
    return torch.mean((y_train-_y)**2)

for epoch in range(0,1000):
    _y = circle_func(x_train)
    loss = loss_func(y_target,_y)
    if epoch >600:
        step = 1e-3
    loss.backward()
    with torch.no_grad():
        _w -= step*_w.grad
        _r2 -= step*_r2.grad
        _w.grad.zero_()
        _r2.grad.zero_()

    print('epoch {} loss = {}'.format(epoch,loss.item() ))
    y_res = circle_func(x_train)
    y_result = y_res.detach().numpy()
    plt.scatter(x_, y_,c='r')
    plt.scatter(x_, y_result,c='g')

    plt.show()
    plt.pause(0.01)
    plt.clf()

y_res = circle_func(x_train)
y_result = y_res.detach().numpy()
print('x0 = {}，y0 = {} ,r = {}'.format(_w.detach().numpy()[0],_w.detach().numpy()[1],np.sqrt(_r2.item())))
plt.scatter(x_, y_, c='r')
plt.scatter(x_,y_result,c = 'g')
plt.show()


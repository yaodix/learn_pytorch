#功能：直线拟合

import torch
import  numpy as np
import matplotlib.pyplot as plt

def linear_func(x):
    return _w*x+_b

def loss_func(y_train,_y):
    return torch.mean((y_train-_y)**2)
plt.ion()   #打开交互模式
#y = 2.5x+4.2
w = 2.5
b = 4.2
x = (np.arange(0,8,0.2))
y = w*x+b
randnum = np.random.randint(-2,2,x.size)
y = y+randnum
print(randnum.size)

plt.figure()
plt.plot(x,y,'*r')
#plt.show()

x_train = torch.from_numpy(x).float()
y_train = torch.from_numpy(y).float()
#print('x_train type ={} '.format(x_train.dtype))

_w = torch.randn(1,requires_grad=True) #params to learn
_b = torch.randn(1,requires_grad= True)
#print('_w type ={} '.format(_w.dtype))

step = 1e-2
for epoch in range(0,600):
    _y = linear_func(x_train)
    loss = loss_func(y_train,_y)
    if epoch >300:
        step =5e-3
    loss.backward()
    with torch.no_grad(): #make all the operations in the block have no gradients, just to update  values.
        _w -= step*_w.grad
        _b -= step*_b.grad
        _w.grad.zero_()
        _b.grad.zero_()

    print('epoch {} loss = {}'.format(epoch,loss.item() ))
    y_res = _w * x_train + _b
    y_result = y_res.detach().numpy()

    plt.plot(x, y, '*r')
    plt.plot(x, y_result, "-g")
    plt.show()
    plt.pause(0.1)
    plt.clf()

#输出最终结果
y_res = _w*x_train+_b
print('_w = {}, _b={}'.format(_w.item(),_b.item()))
y_result = y_res.detach().numpy()
plt.plot(x,y_result,"-g")
plt.show()




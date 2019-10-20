#函数功能：多元函数拟合

import torch
import  numpy as np
import matplotlib.pyplot as plt


plt.ion()   #打开交互模式


x = (np.arange(-3,3,0.2))
w_target =np.array([0.5,3,2.4])
b_target = np.array([0.9])
y = b_target[0]+w_target[0]*x +w_target[1]*x**2+w_target[2]*x**3


plt.figure()
plt.plot(x,y,'*r')


x_train = torch.from_numpy(x).float()
y_train = torch.from_numpy(y).float()
#print('x_train type ={} '.format(x_train.dtype))
_w = torch.randn(2,requires_grad=True)
_b = torch.randn(1,requires_grad= True)
#print('_w type ={} '.format(_w.dtype))


step = 1e-2
def linear_func(x):
    return _w[0]*x+_w[1]*x**2+_b

def loss_func(y_train,_y):
    return torch.mean((y_train-_y)**2)

for epoch in range(0,400):
    _y = linear_func(x_train)
    loss = loss_func(y_train,_y)
    if epoch >200:
        step = 1e-3
    loss.backward()
    with torch.no_grad():
        _w -= step*_w.grad
        _b -= step*_b.grad
        _w.grad.zero_()
        _b.grad.zero_()

    print('epoch {} loss = {}'.format(epoch,loss.item() ))
    y_res = linear_func(x_train)
    y_result = y_res.detach().numpy()
    plt.plot(x, y, '*r')
    plt.plot(x, y_result, "-g")
    plt.show()
    plt.pause(0.03)
    plt.clf()

plt.ioff()
y_res =linear_func(x_train)
#print('_w = {}, _b={}'.format(_w.item(),_b.item()))
plt.plot(x, y, '*r')
y_result = y_res.detach().numpy()
plt.plot(x,y_result,"-g")
plt.show()


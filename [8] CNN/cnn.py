import torch
from torch import nn
# from d2l import torch as d2l
import time
import platform
import warnings


def generate():
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    Y = torch.tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
    X = X.reshape(1, 1, 6, 8)
    Y = Y.reshape(1, 1, 6, 7)
    return X, Y

def corr2d(X, K):
    """计算二维互相关性"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1), (X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w]*K).sum()
    return Y

def get_net():
    net = nn.Conv2d(1, 1, (1, 2))
    return net

def trian(num_epoch):
    X, Y = generate()
    net = get_net()
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr = 0.1)

    for i in range(num_epoch):
        l = loss(net(X), Y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        
        print(f'epoch{i+1}, loss{l}')

if __name__=="__main__":
    warnings.filterwarnings('ignore')
    time_start = time.time()
    trian(50)
    time_end = time.time()
    print(f'time cost{time_end - time_start}')
    # print( platform.platform())
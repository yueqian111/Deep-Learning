import random
import torch
from d2l import torch as d2l
 
#数据集
def synthetic_data(w,b,num):
    X=torch.normal(0,1,(num,len(w)))#均值 方差 大小(行，列)
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))#列向量返回 -1表示自动计算，1表示固定
 
ture_w=torch.tensor([2,-3.4])
ture_b=4.2
features,labels=synthetic_data(ture_w,ture_b,1000)
 
 
 
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)#打乱
#起始 结束+1 步幅
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]
 
batch_size=10
 
#for X,y in  data_iter(batch_size,features,labels):
 #   print(X,'\n',y)
 #   break
 
 
#定义初始化模型参数
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
#定义模型
def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X,w)+b
#定义损失函数
def squared_loss(y_hat,y):
    """均方误差"""
    return(y_hat-y.reshape(y_hat.shape))**2/2
#定义优化算法
def sgd(params,lr,batch_size):
#参数 学习率 batch_size
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -=lr*param.grad/batch_size
            param.grad.zero_()
 
#训练过程
lr=0.03
num_epochs=3#数据扫3遍
net=linreg
loss=squared_loss
 
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')
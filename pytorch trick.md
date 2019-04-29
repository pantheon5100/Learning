<h1>pytorch trick</h1>

[TOC]

# 1.什么情况下应该设置 cudnn.benchmark = True？

```python
torch.backends.cudnn.benchmark = True
```
一般来讲，应该遵循以下准则：

+ 如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
+ 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率.

# 2.PyTorch中在反向传播前为什么要手动将梯度清零？
作者：Pascal
[链接](https://www.zhihu.com/question/303070254/answer/573037166)

这种模式可以让梯度玩出更多花样，比如说梯度累加（gradient accumulation）传统的训练函数，一个batch是这么训练的：
``` python
for i,(images,target) in enumerate(train_loader):

# 1. input output
images = images.cuda(non_blocking=True)
target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
outputs = model(images)
loss = criterion(outputs,target)

# 2. backward
optimizer.zero_grad()   # reset gradient
loss.backward()
optimizer.step()
```
获取loss：输入图像和标签，通过infer计算得到预测值，计算损失函数；optimizer.zero_grad() 清空过往梯度；loss.backward() 反向传播，计算当前梯度；optimizer.step() 根据梯度更新网络参数简单的说就是进来一个batch的数据，计算一次梯度，更新一次网络使用梯度累加是这么写的：
``` python
for i,(images,target) in enumerate(train_loader):
# 1. input output
images = images.cuda(non_blocking=True)
target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
outputs = model(images)
loss = criterion(outputs,target)

# 2.1 loss regularization
loss = loss/accumulation_steps   
# 2.2 back propagation
loss.backward()
# 3. update parameters of net
if((i+1)%accumulation_steps)==0:
    # optimizer the net
    optimizer.step()        # update parameters of net
    optimizer.zero_grad()   # reset
```
gradient获取loss：输入图像和标签，通过infer计算得到预测值，计算损失函数；loss.backward()

反向传播，计算当前梯度；多次循环步骤1-2，不清空梯度，使梯度累加在已有梯度上；梯度累加了一定次数后，先optimizer.step() 根据累计的梯度更新网络参数，然后optimizer.zero_grad() 清空过往梯度，为下一波梯度累加做准备；

总结来说：梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空，不断累加，累加一定次数后，根据累加的梯度更新网络参数，然后清空梯度，进行下一次循环。一定条件下，batchsize越大训练效果越好，梯度累加则实现了batchsize的变相扩大，如果accumulation_steps为8，则batchsize '变相' 扩大了8倍，是我们这种乞丐实验室解决显存受限的一个不错的trick，使用时需要注意，学习率也要适当放大。

更新1：关于BN是否有影响，之前有人是这么说的：As far as I know, batch norm statistics get updated on each forward pass, so no problem if you don't do .backward() every time.BN的估算是在forward阶段就已经完成的，并不冲突，只是accumulation_steps=8和真实的batchsize放大八倍相比，效果自然是差一些，毕竟八倍Batchsize的BN估算出来的均值和方差肯定更精准一些。

更新2：根据 @李韶华 的分享，可以适当调低BN自己的momentum参数bn自己有个momentum参数：x_new_running = (1 - momentum) * x_running + momentum * x_new_observed. momentum越接近0，老的running stats记得越久，所以可以得到更长序列的统计信息我简单看了下[PyTorch 1.0的源码](https://github.com/pytorch/pytorch/blob/162ad945902e8fc9420cbd0ed432252bd7de673a/torch/nn/modules/batchnorm.py#L24)，BN类里面momentum这个属性默认为0.1，可以尝试调节下。



# 3. Pytorch loss function 总结

参考 ：[pytorch loss function conclusion](https://blog.csdn.net/zhangxb35/article/details/72464152)

很多loss函数都有 `size_average` 和 `reduce` 两个布尔类型的参数，需要解释一下。因为一般损失函数都是直接计算batch的数据， 因此返回的loss结果都是维度为(batch_size, ...) 的向量。

+ 如果 `reduce = false` ,那么`size_average`参数无效， 直接返回向量形式的loss；
+ 如果 `reduce = True` ， 那么loss返回的是标量
  + 如果`size_average = True` , 返回 `loss.mean()`;
  + 如果`size_average = False` ， 返回 `loss.sum()`

## 3.1 nn.L1Loss

$$
\operatorname{loss}\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right)=\left|\mathbf{x}_{i}-\mathbf{y}_{i}\right|
$$

要求 $x$ 和 $y$ 的维度要一样（可以是向量或者矩阵），得到的 loss 维度也是对应一样的。这里用下标 i 表示第 i 个元素。一阶范数。
```python
loss_fn = torch.nn.L1Loss(reduce=False, size_average=False)
input = torch.autograd.Variable(torch.randn(3,4))
target = torch.autograd.Variable(torch.randn(3,4))
loss = loss_fn(input, target)
print(input)
print(target)
print(loss)
print(input.size(), target.size(), loss.size())
```


> tensor([[ 0.6793,  1.8591, -1.1041, -0.1995],
>         [ 0.2771, -0.8312,  0.1774, -0.7146],
>         [ 0.5816, -0.1780, -1.5800, -0.9133]])
> tensor([[ 1.2876,  0.2281,  1.0222, -0.4885],
>         [-0.1980, -0.7899, -1.8150,  0.1471],
>         [-1.5262, -0.5305, -0.4179, -0.2347]])
> tensor([[0.6083, 1.6311, 2.1263, 0.2890],
>         [0.4751, 0.0413, 1.9924, 0.8617],
>         [2.1078, 0.3525, 1.1622, 0.6786]])
> torch.Size([3, 4]) torch.Size([3, 4]) torch.Size([3, 4])



## 3.2 nn.SmoothL1Loss

也叫作 HUber Loss，误差在（-1,1）上是平方损失，其他情况是L1损失。
$$
\operatorname{loss}\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right)=\left\{\begin{array}{ll}{\frac{1}{2}\left(\mathbf{x}_{i}-\mathbf{y}_{i}\right)^{2}} & {\text { if }\left|\mathbf{x}_{i}-\mathbf{y}_{i}\right|<1} \\ {\left|\mathbf{x}_{i}-\mathbf{y}_{i}\right|-\frac{1}{2},} & {\text { otherwise }}\end{array}\right.
$$
![huber loss](C:\Users\saber\Documents\LearningMD\torch trick pic\虎贝尔.png)


这里很上面的 L1Loss 类似，都是 element-wise（逐点，点乘） 的操作，下标 i 是 x 的第 i 个元素。

a loss function used in robust regression, that is ==less sensitive to outliers== in data than the squared error loss.

```python
loss_fn = torch.nn.SmoothL1Loss(reduce=False, size_average=False)
input = torch.autograd.Variable(torch.randn(3,4))
target = torch.autograd.Variable(torch.randn(3,4))
loss = loss_fn(input, target)
print(input); print(target); print(loss)
print(input.size(), target.size(), loss.size())
```

> tensor([[-0.2866, -0.6683,  0.2946, -0.8280],
>         [ 0.0132, -0.0168, -0.1908,  0.9864],
>         [-0.6769, -0.1438, -1.4344,  1.3525]])
> tensor([[ 0.1833,  0.2032,  0.3971,  0.5473],
>         [ 0.8244,  1.0036,  0.7856, -2.5749],
>         [-0.2325, -0.1510, -0.1874, -1.5165]])
> tensor([[1.1043e-01, 3.7979e-01, 5.2451e-03, 8.7531e-01],
>         [3.2901e-01, 5.2042e-01, 4.7668e-01, 3.0613e+00],
>         [9.8726e-02, 2.5969e-05, 7.4704e-01, 2.3690e+00]])
> torch.Size([3, 4]) torch.Size([3, 4]) torch.Size([3, 4])



## 3.3 nn.MSELoss

均方损失函数，用法和上面类似，这里 loss, x, y 的维度是一样的，可以是向量或者矩阵，i 是下标。
$$
\operatorname{loss}\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right)=\left(\mathbf{x}_{i}-\mathbf{y}_{i}\right)^{2}
$$


## 3.4 nn.BCELoss

二分类用的交叉熵，用的时候需要在该层前面加上 `Sigmoid` 函数.

离散版的交叉熵定义是 :
$$
H(\boldsymbol{p}, \boldsymbol{q})=-\sum_{i} \boldsymbol{p}_{i} \log \boldsymbol{q}_{i}
$$
其中 p,q 都是向量，且都是概率分布。如果是二分类的话，因为只有正例和反例，且两者的概率和为 1，那么只需要预测一个概率就好了，因此可以简化成
$$
\operatorname{loss}\left(\mathbf{x}_{i}, \mathbf{y}_{i}\right)=-\boldsymbol{w}_{i}\left[\mathbf{y}_{i} \log \mathbf{x}_{i}+\left(1-\mathbf{y}_{i}\right) \log \left(1-\mathbf{x}_{i}\right)\right]
$$

注意这里 x,y 可以是向量或者矩阵，i 只是下标；$x_i$ 表示第 i 个样本预测为 正例 的概率，$y_i $表示第 i 个样本的标签，wi 表示该项的权重大小。可以看出，loss, x, y, w 的维度都是一样的。 

```python
import torch.nn.functional as F
loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)
input = Variable(torch.randn(3, 4))
target = Variable(torch.FloatTensor(3, 4).random_(2))
loss = loss_fn(F.sigmoid(input), target)
print(input); print(target); print(loss)
```

这里比较奇怪的是，权重的维度不是 2，而是和 x, y 一样，有时候遇到正负例样本不均衡的时候，可能要多写一句话

```python
class_weight = Variable(torch.FloatTensor([1, 10])) # 这里正例比较少，因此权重要大一些
target = Variable(torch.FloatTensor(3, 4).random_(2))
weight = class_weight[target.long()] # (3, 4)
loss_fn = torch.nn.BCELoss(weight=weight, reduce=False, size_average=False)
# balabala...
```



## 3.5 nn.BCEWithLogitsLoss
上面的 nn.BCELoss 需要手动加上一个 Sigmoid 层，这里是结合了两者，这样做能够利用 log_sum_exp trick，使得数值结果更加稳定（numerical stability）。建议使用这个损失函数。

值得注意的是，文档里的参数只有 weight, size_average 两个，但是实际测试 reduce 参数也是可以用的。此外两个损失函数的 target 要求是 FloatTensor，而且不一样是只能取 0, 1 两种值，任意值应该都是可以的。

## 3.6 nn.CrossEntropyLoss
多分类用的交叉熵损失函数，用这个 loss 前面不需要加 Softmax 层。

这里损失函数的计算，按理说应该也是原始交叉熵公式的形式，但是这里限制了 target 类型为 torch.LongTensr，而且不是多标签意味着标签是 one-hot 编码的形式，即只有一个位置是 1，其他位置都是 0，那么带入交叉熵公式中化简后就成了下面的简化形式。参考 cs231n 作业里对 Softmax Loss 的推导。 

$$
\operatorname{loss}(\mathbf{x}, \text { label })=-\boldsymbol{w}_{\text { label }} log \frac{e^{\mathrm{x}_{\text { label }}}}{\sum_{j=1}^{N} e^{\mathrm{x}_{j}}}
$$

$$
=\boldsymbol{w}_{\text { label }}\left[-\mathbf{x}_{\text { label }}+\log \sum_{j=1}^{N} e^{\mathbf{x}_{j}}\right]
$$

这里的 x∈RN，是没有经过 Softmax 的激活值，N 是 x 的维度大小（或者叫特征维度）；label∈[0,C−1] 是标量，是对应的标签，可以看到两者维度是不一样的。C 是要分类的个数。w∈RC 是维度为 C 的向量，表示标签的权重，样本少的类别，可以考虑把权重设置大一点。

```python
weight = torch.Tensor([1,2,1,1,10])
loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False, weight=weight)
input = Variable(torch.randn(3, 5)) # (batch_size, C)
target = Variable(torch.FloatTensor(3).random_(5))
loss = loss_fn(input, target)
print(input); print(target); print(loss)
```

## 3.7 nn.NLLLoss

用于多分类的负对数似然损失函数（Negative Log Likelihood） 

$$
\operatorname{loss}(\mathbf{x}, \text { label })=-\mathbf{x}_{\text { label }}
$$

在前面接上一个 nn.LogSoftMax 层就等价于交叉熵损失了。事实上，nn.CrossEntropyLoss 也是调用这个函数。注意这里的 xlabelxlabel 和上个交叉熵损失里的不一样（虽然符号我给写一样了），这里是经过 logSoftMaxlogSoftMax 运算后的数值，

## 3.8 nn.NLLLoss2d
和上面类似，但是多了几个维度，一般用在图片上。现在的 pytorch 版本已经和上面的函数合并了。

input, (N, C, H, W)
target, (N, H, W)

比如用全卷积网络做 Semantic Segmentation 时，最后图片的每个点都会预测一个类别标签。

## 3.9 nn.KLDivLoss

KL 散度，又叫做相对熵，算的是两个分布之间的距离，越相似则越接近零。 
loss(x,y)=1N∑i=1N[yi∗(logyi−xi)]
loss(x,y)=1N∑i=1N[yi∗(log⁡yi−xi)]

注意这里的 xi 是 log 概率，刚开始还以为 API 弄错了。



# 4. view and view_as

view()函数是在`torch.Tensor.view()` 下的一个函数，可以用tensor调用，也可以用variable调用。

其作用在于返回和原tensor数据个数相同，但size不同的tensor。

+ reshape函数调用不依赖于tensor在内存中是不是连续的。即 reshape≈tensor.contiguous().view

```python
import numpy as np
import torch
from torch.autograd import Variable
 
x = torch.Tensor(2,2,2)
print(x)
 
y = x.view(1,8)
print(y)
 
z = x.view(-1,4)  # the size -1 is inferred from other dimensions
print(z)
 
t = x.view(8)
print(t)
```

**view_as** 返回被视作与给定的tensor相同大小的原tensor。等效于：

`self.view(tensor.size())`

```python
a = torch.Tensor(2, 4)
b = a.view_as(torch.Tensor(4, 2))
print (b)
```

# 5. pytorch  dataloader 深入剖析

参考链接： https://www.cnblogs.com/ranjiewen/p/10128046.html


输入数据PipeLine
pytorch 的数据加载到模型的操作顺序是这样的：

① 创建一个 Dataset 对象
② 创建一个 DataLoader 对象
③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练

```python
dataset = MyDataset()
dataloader = DataLoader(dataset)
num_epoches = 100
for epoch in range(num_epoches):
    for img, label in dataloader:
        ....
```

所以，作为直接对数据进入模型中的关键一步， DataLoader非常重要。

首先简单介绍一下DataLoader，它是PyTorch中数据读取的一个重要接口，该接口定义在dataloader.py中，只要是用PyTorch来训练模型基本都会用到该接口（除非用户重写…），该接口的目的：将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练。

官方对DataLoader的说明是：“数据加载由数据集和采样器组成，基于python的单、多进程的iterators来处理数据。”关于iterator和iterable的区别和概念请自行查阅，在实现中的差别就是iterators有`__iter__`和`__next__`方法，而iterable只有`__iter__`方法  



# 6. 获取网络的任意一层的输出

![check f](C:\Users\saber\Documents\agit\Learning\torch trick pic\check feature.jpg)

```python
    # load model and data
    model = P3D199()
    model = model.cuda()
    model.eval()
    data=torch.autograd.Variable(torch.rand(16,3,16,160,160)).cuda() 
    # verify
    out=model(data)
    feature=model.feature
    out2=model.fc(feature)
    print(out==out2) 
```



# 7.设置随机种子

```python
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
```



# 8. 如何在训练时固定一些层？Pytorch迁移学习小技巧 以及 Pytorch小技巧的一些总结

[参考1](https://blog.csdn.net/VictoriaW/article/details/72779407)

[参考2](https://blog.csdn.net/u011268787/article/details/80170482)


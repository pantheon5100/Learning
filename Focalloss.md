<h1>Focalloss</h1>

[TOC]



# 1.WHY ?

Focal loss 主要是为了解决 one-stage 目标检测中正负样本比例严重失衡问题。该损失函数降低了大量简单负样本在训练中所占的权重，也可以理解为一种困难原本挖掘。

# 2. HOW？

基于交叉熵损失函数：
$$
\mathrm{L}=-\mathrm{ylogy}^{\prime}-(1-y) \log \left(1-y^{\prime}\right)=\left\{\begin{array}{ll}{-\log y^{\prime}} & {y=1} \\ {-\log \left(1-y^{\prime}\right),} & {y=0}\end{array}\right.
$$
$y~'$  是经过激活函数的输出，在0-1之间。可见普通的交叉熵对于正样本而言，输出概率越大损失越小。遂于负样本而言，输出概率越小则损失越小。此时的损失函数在大量简单样本的迭代过程中比较缓慢且可能无法优化至最优。

Focal loss：
$$
\mathrm{L}_{f l}=\left\{\begin{array}{ll}{-\left(1-y^{\prime}\right)^{\gamma} \log y^{\prime}} & {y=1} \\ {-y^{\prime \gamma} \log \left(1-y^{\prime}\right),} & {y=0}\end{array}\right.
$$
![pogrc](Focalloss.assets\pogtc.png)

首先在原有的基础上加了一个因子，其中$\gamma > 0$  减少易分类样本的损失，更关注困难的、错分的样本。

例如gamma为2，对于正类样本而言，预测结果为0.95肯定是简单样本，所以（1-0.95）的gamma次方就会很小，这时损失函数值就变得更小。而预测概率为0.3的样本其损失相对很大。对于负类样本而言同样，预测0.1的结果应当远比预测0.7的样本损失值要小得多。对于预测概率为0.5时，损失只减少了0.25倍，所以更加关注于这种难以区分的样本。这样减少了简单样本的影响，大量预测概率很小的样本叠加起来后的效应才可能比较有效。

此外，加入平衡因子alpha，用来平衡正负样本本身的比例不均：

$$
\mathrm{L}_{f l}=\left\{\begin{array}{ll}{-\alpha\left(1-y^{\prime}\right)^{\gamma} \log y^{\prime}} & {y=1} \\ {-(1-\alpha) y^{\prime \gamma} \log \left(1-y^{\prime}\right),} & {y=0}\end{array}\right.
$$
只添加alpha虽然可以平衡正负样本的重要性，多少无法解决简单与困难样本的问题。

lambda调节简单样本权重降低的速率，当lambda为0时即为交叉熵损失函数，当lambda增加是，调整因子的影响也在增加。实验发现lambda为2是最优的。

# 3.Conclusion

作者认为one-stage和two-stage的表现差异主要原因是大量前景背景类别不平衡导致。作者设计了一个简单密集型网络RetinaNet来训练在保证速度的同时达到了精度最优。在双阶段算法中，在候选框阶段，通过得分和nms筛选过滤掉了大量的负样本，然后在分类回归阶段又固定了正负样本比例，或者通过OHEM在线困难挖掘使得前景和背景相对平衡。而one-stage阶段需要产生约100k的候选位置，虽然有类似的采样，但是训练仍然被大量负样本所主导。

# 4. 附录

## 4.1 交叉熵

### 信息论

交叉熵是信息论中的一个概念，要想了解交叉熵的本质，需要从最基本的概念讲起。

### 4.1.1 信息量

假设$X$ 是一个离散型随机变量，取其值集合为$\mathcal{X}$ ，概率分布函数 
$$
p(x)=\operatorname{Pr}(X=x), x \in \chi
$$
定义事件 $X=x_0$ 的信息量为：
$$
I\left(x_{0}\right)=-\log \left(p\left(x_{0}\right)\right)
$$

### 4.1.2 熵

表示所有信息量的期望，即：
$$
H(X)=-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(p\left(x_{i}\right)\right)
$$
对于0-1分布问题，有：
$$
\begin{aligned} H(X) &=-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(p\left(x_{i}\right)\right) \\ &=-p(x) \log (p(x))-(1-p(x)) \log (1-p(x)) \end{aligned}
$$

### 4.1.3 相对熵（KL散度）

参考代码：https://github.com/thushv89/exercises_thushv_dot_com/blob/master/kl_divergence.ipynb

博客：https://www.jianshu.com/p/7b7c0777f74d

相对熵又称KL散度,如果我们对于同一个随机变量 x 有两个单独的概率分布 P(x) 和 Q(x)，我们可以使用 KL 散度（Kullback-Leibler (KL) divergence）来衡量这两个分布的差异，两个分布的差异越大，KL散度越大

维基百科对相对熵的定义：

> In the context of machine learning, DKL(P‖Q) is often called the information gain achieved if P is used instead of Q.

即如果用P来描述目标问题，而不是用Q来描述目标问题，得到的==信息增量==。

在机器学习中，P往往用来表示样本的真实分布，比如[1,0,0]表示当前样本属于第一类。Q用来表示模型所预测的分布，比如[0.7,0.2,0.1] 
直观的理解就是如果用P来描述样本，那么就非常完美。而用Q来描述样本，虽然可以大致描述，但是不是那么的完美，信息量不足，需要额外的一些“信息增量”才能达到和P一样完美的描述。如果我们的Q通过反复训练，也能完美的描述样本，那么就不再需要额外的“信息增量”，Q等价于P。

KL散度的计算公式：
$$
D_{K L}(p \| q)=\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(\frac{p\left(x_{i}\right)}{q\left(x_{i}\right)}\right)
$$
$n$ 为事件的所有可能性，其中p(x)是目标分布，q(x)是去匹配的分布，如果两个分布完全匹配，那么
$$
D_{K L}(p \| q)=0
$$


$D_{KL}$的值越小，表示q分布和p分布越接近,KL 散度是一种衡量两个分布（比如两条线）之间的匹配程度的方法。



### 4.1.4 交叉熵

$$
\begin{aligned} D_{K L}(p \| q) &=\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(p\left(x_{i}\right)\right)-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(q\left(x_{i}\right)\right) \\ &=-H(p(x))+\left[-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(q\left(x_{i}\right)\right)\right] \end{aligned}
$$

等式的前一部分恰巧就是p的熵，等式的后一部分，就是交叉熵： 
$$
H(p, q)=-\sum_{i=1}^{n} p\left(x_{i}\right) \log \left(q\left(x_{i}\right)\right)
$$

在机器学习中，我们需要==评估label和predicts之间的差距==，使用KL散度刚刚好，即DKL(y||y^)，由于KL散度中的前一部分−H(y)不变，故在优化过程中，只需要关注交叉熵就可以了。所以一般在机器学习中直接用用交叉熵做loss，评估模型。

机器学习中常常使用你MSE作为loss函数:
$$
\operatorname{loss}=\frac{1}{2 m} \sum_{i=1}^{m}\left(y_{i}-\hat{y}_{i}\right)^{2}
$$
使用交叉熵则变为：
$$
loss=-\frac{1}{m} \sum_{j=1}^{m} \sum_{i=1}^{n} y_{j i} \log \left(\hat{y_{j i}}\right)
$$
m为当前batch的样本数



## 4.2 OHEM

[参考链接](https://blog.csdn.net/wfei101/article/details/78067257)
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



## 4.2 one-stage 和 two-stage



## 4.3 OHEM
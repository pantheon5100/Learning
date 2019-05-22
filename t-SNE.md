<h1>t-SNE 原理与推导</h1>

> t-SNE(t-distributed stochastic neighbor embedding)是用于**降维**的一种机器学习算法，是由 Laurens van der Maaten 和 Geoffrey Hinton在08年提出来。此外，t-SNE 是一种非线性降维算法，非常适用于高维数据降维到2维或者3维，进行可视化。
>
> t-SNE是由SNE(Stochastic Neighbor Embedding, SNE; Hinton and Roweis, 2002)发展而来。我们先介绍SNE的基本原理，之后再扩展到t-SNE。最后再看一下t-SNE的实现以及一些优化。

[TOC]

# 1. SNE

## 1.1 基本原理

SNE是通过仿射(affinitie) 变换将数据点映射到概率分布上，主要包括两个步骤：

+ SNE 构建一个高维对象之间的概率分布，是的相似的对象有更高的概率被选择，而不相似的对象有较低的概率被选择。
+ SNE 在低维空间里构建这些点的概率分布，是的这两个概率分布之间尽可能的相似。

我们看到t-SNE模型是非监督的降维，他跟kmeans等不同，他不能通过训练得到一些东西之后再用于其它数据（比如kmeans可以通过训练得到k个点，再用于其它数据集，而t-SNE只能单独的对数据做操作，也就是说他只有fit_transform，而没有fit操作）

# 2. t-SNE

## 2.1 算法

**步骤1：**

​	随机邻接嵌入（SNE）通过将数据点之间的高维欧几里得距离转换为表示相似性的条件概率而开始，数据点$x_i$、$x_j$之间的条件概率$p_{j|i}$由下式给出：
$$
p_{j | i}=\frac{\exp \left(-\left\|x_{i}-x_{j}\right\|^{2} / 2 \sigma_{i}^{2}\right)}{\sum_{k \neq i} \exp \left(-\left\|x_{i}-x_{k}\right\|^{2} / 2 \sigma_{i}^{2}\right)}
$$
​	其中$\delta_i$ 是以数据点$x_i$ 为中心的高斯方差。

**步骤2：**

​	对于高维数据点$x_i$ 和$x_j$ 的低维对应点$y_i$ 和$y_j$ 而言，可以计算类似的条件概率$q_{j|i}$ 
$$
q_{j | i}=\frac{\exp \left(-\left\|y_{i}-y_{j}\right\|^{2}\right)}{\sum_{k \neq i} \exp \left(-\left\|y_{i}-y_{k}\right\|^{2}\right)}
$$
SNE 试图最小化条件概率的差异。

**步骤3：**

​    为了测量条件概率差的和最小值，SNE使用梯度下降法最小化KL距离。而SNE的代价函数关注于映射中数据的局部结构，优化该函数是非常困难的，而t-SNE采用重尾分布，以减轻拥挤问题和SNE的优化问题。

定义困惑度（KL散度）：
$$
\operatorname{Perp}\left(P_{i}\right)=2^{H\left(P_{i}\right)}
$$
其中$H(P_i)$ 是香农熵：
$$
H\left(P_{i}\right)=-\sum_{j} p_{j|i} \log _{2} p_{j | i}
$$

## 2.2. 时间和空间复杂度

   算法计算对应的是条件概率，并试图最小化较高和较低维度的概率差之和，这涉及大量的计算，对系统资源要求高。t-SNE的复杂度随着数据点数量有着时间和空间二次方。

# 3.本质

​	t-SNE 非线性降维算法通过基于具有多个特征的数据点的相似性识别观察到的簇来在数据中找到模式。本质上是一种降维和可视化技术。另外t-SNE的输出可以作为其他分类算法的输入特征
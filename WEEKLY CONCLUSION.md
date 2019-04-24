<h1>Weekly Conclusion</h1>

[TOC]

# WEEK 1  (4.15-4.21)

- [ ] Neaten the below

我这周比较了三个版本（两个pytorch版本（torch1：ooooverflow，torch2：官方face++）），一个TensorFlow版本（tensor））的bisenet实现代码 最终选择了官方版本（torch2）与一个其他人实现的版本（torch1）两个相结合：模型结构使用torch1，训练代码使用torch2最终能达到0.56，其中torch1达到0.45
torch2实现在多gpu服务器上，本来打算就只修改它的，最终没有改成功有很多bug。
我发现这几个版本的模型结构有很多不一样的地方，torch1和tensor模型结构比较相似，但loss完全不同，torch2和torch1的loss一直，所以最总决定结合torch1和torch2.

 github如下
tensor版本：
https://github.com/GeorgeSeif/Semantic-Segmentation-Suite

torch1版本：
https://github.com/ooooverflow/BiSeNet

torch2版本：https://github.com/ycszen/TorchSeg
 

# WEEK 2 (4.22-4.28)

- [ ] 设计Attention实验
- [ ] 使用 facol loss


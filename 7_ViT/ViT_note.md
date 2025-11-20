# vit summary

To summarize, in Vision transformer, images are reorganized as 2D grids of patches(treating them as tokens). The models are trained on those patches.

### CNN和ViT区别

CNN的设计有两个重要假设（归纳偏置)

* 特征平移不变性
* 局部性：图像中的像素主要与其周围的像素相互作用以形成特征。

CNN models are very good at these two biases. ViT do not have this assumption

### ViT

#### 模型理论

常见模型

[google](https://huggingface.co/google)/[vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) : 86M，

[google]()/[vit-large-patch16-224]() : 0.3B

[google]()/[vit-huge-patch14-224]() : 0.6B

代码细节

cls_token如何实现分类计算？

One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

前向传播过程清晰地展示了 ViT 的工作流程：

 **图像分块 → 添加[CLS]令牌 → 加入位置信息 → Transformer编码（信息融合）→ 提取[CLS]特征作为图像全局表示 → 分类输出** 。

其中，`x[:, 0]`是提取图像全局表示的核心操作。

vit中好像并没有看到位置信息

### SwinTransformer


### Vit应用

https://github.com/Gabriel-ds1/ViT-Playground

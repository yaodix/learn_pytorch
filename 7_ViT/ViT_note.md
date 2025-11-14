
cls_token如何实现分类计算？

One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.

前向传播过程清晰地展示了 ViT 的工作流程：

 **图像分块 → 添加[CLS]令牌 → 加入位置信息 → Transformer编码（信息融合）→ 提取[CLS]特征作为图像全局表示 → 分类输出** 。

其中，`x[:, 0]`是提取图像全局表示的核心操作。

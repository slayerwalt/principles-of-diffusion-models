# 1. 深度生成建模 {#ch-overview-dgm}

> *凡是我无法创造的，我就无法理解。*
>
> — Richard P. Feynman

深度生成模型（deep generative models，DGMs）是一类神经网络，它们学习图像、文本、音频等高维数据上的概率分布，从而生成与数据集相似的新样本。我们用 $p_{{\boldsymbol{\phi}}}$ 表示模型分布，用 $p_{\mathrm{data}}$ 表示数据分布。给定有限的数据集，我们通过最小化一个损失来拟合 ${\boldsymbol{\phi}}$，该损失衡量 $p_{{\boldsymbol{\phi}}}$ 与 $p_{\mathrm{data}}$ 之间的差异。训练完成后，生成就是运行模型的采样过程，抽取 ${\mathbf{x}} \sim p_{{\boldsymbol{\phi}}}$（密度 $p_{{\boldsymbol{\phi}}}({\mathbf{x}})$ 能否直接计算，取决于模型类别）。评价模型质量时，需要考察生成样本及其汇总统计量与来自 $p_{\mathrm{data}}$ 的样本及其统计量的匹配程度，并结合特定任务的指标或感知指标。

本章将为这些思想建立数学和概念基础。我们在[第 1.1 节](01-what-is-deep-generative-modeling.md#sec-formulation-dgm)中形式化描述这一问题，在[第 1.2 节](02-examples.md#sec-examples-dgm)中介绍具有代表性的模型类别，并在[第 1.3 节](03-taxonomy.md#sec-taxonomy)中总结一套实用的分类方法。

**读完本章后，你将能够……**

-   用通俗的语言解释生成模型试图从数据集中学到什么，以及对数据分布建模意味着什么；

-   理解自回归模型、VAE、基于能量的模型、归一化流、GAN 和扩散模型的基本思想，了解它们在学习和生成新样本的方式上有何不同；

**阅读路线。** 如果你已经熟悉现代生成模型的基本思想，可以略读本章，之后将其作为快速查阅的参考。

# 1.3 建模方法的分类 {#sec-taxonomy}

如前所述，深度生成模型涵盖了多种建模策略。一个根本区别在于，这些模型如何对底层数据分布进行*参数化*，也就是它们是*显式地*指定 $p_{\boldsymbol{\phi}}(\mathbf{x})$，还是仅*隐式地*指定它；这一区分与训练目标无关。

-   **显式模型：** 这类模型通过可计算或近似可计算的密度函数或概率质量函数，直接对概率分布 $p_{\boldsymbol{\phi}}(\mathbf{x})$ 进行参数化。例如，AR、NF、VAE 和 DM 都精确定义了 $p_{\boldsymbol{\phi}}(\mathbf{x})$，或通过可计算的界来定义它。

-   **隐式模型：** 这类模型仅通过采样过程指定分布，通常采用 $\mathbf{x} = {\mathbf{G}}_{\boldsymbol{\phi}}(\mathbf{z})$ 的形式，其中噪声变量 $\mathbf{z} \sim p_{\mathrm{prior}}$。此时，$p_{\boldsymbol{\phi}}(\mathbf{x})$ 没有可用的闭式表达式，甚至可能根本没有定义。

[表 1.1](03-taxonomy.md#tb-comparison-explicit-implicit) 简要总结了这些不同方法。

<span id="tb-comparison-explicit-implicit"></span>

**表 1.1：显式与隐式生成模型的比较**

| | 显式：精确似然 | 显式：近似似然 | 隐式 |
| --- | --- | --- | --- |
| **似然** | 可计算 | 界／近似 | 不直接建模／难以计算 |
| **目标** | MLE | ELBO | 对抗目标 |
| **示例** | NFs, ARs | VAEs, DMs | GANs |

**与扩散模型的联系。** 综合来看，这些经典的深度生成模型家族展示了对复杂分布建模的互补策略。除了各自本身的重要性，它们也为理解扩散模型提供了指导原则。扩散方法继承了其中多个视角的思想：通过变分训练目标与 VAE 相联系；通过学习对数密度梯度的 score 匹配方法与 EBM 相联系，这些梯度与能量函数密切相关；通过连续时间变换与 NF 相联系。

为了给后续章节讨论的扩散方法奠定基础，我们将重点介绍三个主要范式：VAE（[第 2.1 节](../02-variational-perspective/01-variational-autoencoder.md#sec-vae)）、EBM（[第 3.1 节](../03-score-based-perspective/01-energy-based-models.md#sec-ebm)）和 NF（[第 5.1 节](../05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#sec-flow-based-method)）。这些内容将为现代基于扩散的生成建模的核心原理奠定基础，后续章节将进一步展开这些原理。

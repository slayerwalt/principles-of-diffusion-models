# 6. 扩散模型的统一系统视角 {#ch-unified-perspective}

> *数学是给不同事物赋予相同名称的艺术。*
>
> — Henri Poincaré

本章提出一种系统视角，将变分、基于 score 与基于流的视角联系到同一幅完整图景中。虽然出发直觉不同，这些方法都汇聚到现代扩散模型的同一核心机制：定义前向扰动过程，描绘一条边缘分布路径，再学习构造生成动力学所需的局部量，使其沿这条路径反向演化。在[第 2 章](../02-variational-perspective/index.md#ch-variational)、[第 3 章](../03-score-based-perspective/index.md#ch-score-based)、[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)与[第 5 章](../05-flow-based-perspective/index.md#ch-flow-based)的基础上，可以看到共同的做法：先定义沿边缘分布路径演化的前向扰动过程，再学习随时间变化的向量场，沿该路径将简单先验输运为数据分布。

贯穿所有视角的关键要素是[第 6.1 节](01-conditioning-tricks.md#sec-conditional-trick)介绍的条件化技巧，它将不可计算的边缘目标转化为可计算的条件目标，从而实现稳定高效的训练。

[第 6.2 节](02-training-objectives.md#sec-elucidating)系统分析训练目标，辨明其基本组成部分，并阐明变分、基于 score 与基于流的视角如何构造损失函数。

[第 6.3 节](03-equivalent-parameterizations.md#sec-equivalent-parametrizations)说明，任意形如 ${\mathbf{x}}_t = \alpha_t {\mathbf{x}}_0 + \sigma_t {\boldsymbol{\epsilon}}$ 的仿射前向噪声注入，都可以等价地转换为标准线性调度 ${\mathbf{x}}_t = (1 - t){\mathbf{x}}_0 + t{\boldsymbol{\epsilon}}$。此外，噪声预测、干净数据预测、score 预测与速度预测等常用参数化，在梯度层面可以相互转换。因此，噪声调度与参数化的选择都遵循同一建模原理。

最后，[第 6.4 节](04-fokker-planck-connections.md#sec-comparison-connection)汇总讨论，指出支配这些方法的规则：Fokker–Planck 方程。无论采用变分方案（离散时间去噪）、基于 score 的方法（SDE 形式），还是基于流的方法（ODE 形式），都在构造一个边缘分布遵循相同密度演化的生成器。因此，Fokker–Planck 方程是三个视角共同遵守的普遍约束，差异仅来自参数化与训练目标。

**读完本章后，你将能够……**

-   认识变分、基于 score 与基于流训练背后的共同条件化原理：不可计算的边缘目标可以通过可计算的条件目标来学习；

-   理解干净数据、噪声、score 与速度这四种常见预测类型，并在前向调度非退化时转换它们的 oracle 预测；

-   区分扩散训练中的主要设计选择：前向噪声调度、预测目标、时间采样分布与时间权重，并理解哪些变化属于数学重参数化，哪些仍会影响实际优化；

-   区分“等价”的不同含义：表示同样的 oracle 信息、总体目标在权重或加性常数意义下等价、遵循相同边缘密度路径，以及作为实际训练或采样算法具有相同行为；

-   理解不同仿射概率路径如何通过时间重参数化与空间缩放联系起来，同时明确这些变换保留和不保留哪些性质；

-   理解连续性方程与 Fokker–Planck 方程如何在密度层面连接扩散模型的变分、确定性 ODE 与随机 SDE 描述。

**阅读路线。** 本章是全书概念上的关键。首次阅读时，应重点关注条件化原理、四种预测类型之间的关系，以及连接三个视角的连续性／Fokker–Planck 视角。阅读重点在于辨明所用的是哪一种等价关系，以及这一等价关系究竟保留了什么，无须记住每个转换公式。之后再回顾权重、调度与参数化选择的详细分析，理解为何数学上等价的形式在实践中仍可能表现不同。

继续阅读以下各节及[参考文献](../../references.md#references)。

[6.1 条件化技巧：扩散模型的秘诀](01-conditioning-tricks.md#sec-conditional-trick)

[6.2 理解扩散模型训练损失的路线图](02-training-objectives.md#sec-elucidating)

[6.3 扩散模型中的等价性](03-equivalent-parameterizations.md#sec-equivalent-parametrizations)

[6.4 底层原理：Fokker–Planck 方程](04-fokker-planck-connections.md#sec-comparison-connection)

[6.5 结语](05-closing-remarks.md#subsec-takeaway-dm-gaussianfm)

<span id="equi-para"></span>

[命题 6.3.1](03-equivalent-parameterizations.md#equi-para)
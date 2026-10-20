# 3.5 小结：NCSN 与 DDPM 的比较 {#sec-comparison-ncsn-ddpm}

我们先比较[图 3.7](04-noise-conditional-score-networks.md#fig-smld-ddpm-graphical-model)中的 NCSN 与 DDPM 图模型，其主要异同总结于[表 3.1](05-comparison.md#tb-comparison-smld-ddpm)。

<span id="tb-comparison-smld-ddpm"></span>

**表 3.1：NCSN 与 DDPM 的比较**

|  | **NCSN** | **DDPM** |
| --- | --- | --- |
| ${\mathbf{x}}_{i+1}\vert {\mathbf{x}}_{i}$ | 推导得到 ${\mathbf{x}}_{i+1} = {\mathbf{x}}_i + \sqrt{\sigma^2_{i+1}-\sigma^2_{i}}\boldsymbol{\epsilon}$ | 定义为 ${\mathbf{x}}_{i+1} = \sqrt{1-\beta_{i}} {\mathbf{x}}_{i} + \sqrt{\beta_{i}}\boldsymbol{\epsilon}$ |
| ${\mathbf{x}}_{i}\vert {\mathbf{x}}$ | 定义为 ${\mathbf{x}}_{i} = {\mathbf{x}} + \sigma_{i}\boldsymbol{\epsilon}$ | 推导得到 ${\mathbf{x}}_i = \bar{\alpha}_i{\mathbf{x}} + \sqrt{1 - \bar{\alpha}_i^2}\boldsymbol{\epsilon}$ |
| $p_{\text{prior}}$ | $\mathcal{N}(\boldsymbol{0}, \sigma_L^2{\mathbf{I}})$ | $\mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$ |
| 损失 | $\mathbb{E}_{i} \mathbb{E}_{p_{\text{data}}({\mathbf{x}})} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}\left(\mathbf{0}, {\mathbf{I}}\right)} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_i, \sigma_i) + \frac{\boldsymbol{\epsilon}}{\sigma_i}\right\rVert_2^2 \right]$ | $\mathbb{E}_{i}\mathbb{E}_{p_{\text{data}}({\mathbf{x}})} \mathbb{E}_{\boldsymbol{\epsilon}\sim \mathcal{N}(\boldsymbol{0},{\mathbf{I}})}\left[ \left\lVert\boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) - \boldsymbol{\epsilon}\right\rVert_2^2 \right]$ |
| 采样 | 每层应用 Langevin 动力学；<br> 用输出初始化下一层 | 使用 <br> $p_{\boldsymbol{\phi}^\times}({\mathbf{x}}_{i-1} \vert  {\mathbf{x}}_i)$ 遍历 Markov 链 |

**共同的瓶颈。** 尽管形式不同，NCSN 与 DDPM 都依赖密集的时间离散化。这带来一个关键限制：采样通常需要数百乃至数千次迭代，使生成缓慢且计算成本高昂。

<span id="ques-dm-sampling-slow"></span>

**问题 3.5.1**

如何加速扩散模型的采样？

我们将在以下章节再次讨论这一挑战：[第 9 章](../09-numerical-solvers/index.md#ch-numerical-solvers)介绍更高效的采样器；[第 10 章](../10-distillation/index.md#ch-distillation)将原本需要数百步采样的预训练扩散模型压缩为少步学生生成器；[第 11 章](../11-fast-generators/index.md#ch-fast-scratch)研究如何在扩散思想启发的原则下，直接从零学习快速的少步生成器。
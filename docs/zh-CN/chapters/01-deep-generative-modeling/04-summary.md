# 1.4 结语 {#sec-ch1-cr}

本章介绍了深度生成建模的基础概念。我们首先定义了主要目标：学习一个可计算的模型分布 $p_{\text{model}}$（由 $\boldsymbol{\phi}$ 参数化），以近似未知且复杂的数据分布 $p_{\text{data}}$。其中一个主要挑战是，定义有效概率密度所需的归一化常数，也称配分函数 $Z(\boldsymbol{\phi})$，在计算上难以求得。

为绕开这一问题，人们提出了多类深度生成模型，每一类都采用不同的策略。我们概述了几种主要方法，包括基于能量的模型（EBM）、自回归模型（AR）、变分自编码器（VAE）、归一化流（NF）和生成对抗网络（GAN）。这些模型大致可分为显式模型和隐式模型：显式模型定义可计算的密度，隐式模型则仅通过采样过程定义分布。

这些经典框架各有重要意义，其中三个构成了本书重点讨论的扩散模型的概念起源：VAE、EBM 和 NF。在接下来的章节中，我们将从这三个基础范式出发，追溯扩散模型的发展：

1.  B 部分将从变分视角（[第 2 章](../02-variational-perspective/index.md#ch-variational)）开始，说明 VAE（的层次潜变量结构）如何自然地引出 Denoising Diffusion Probabilistic Models（DDPM）的表述。

2.  随后，我们将考察基于 score 的视角（[第 3 章](../03-score-based-perspective/index.md#ch-score-based)）。这一视角源自 EBM 和 score 匹配，进而发展出 Noise Conditional Score Networks（NCSN）和更一般的 Score SDE 框架（[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)）。

3.  最后，我们将考察基于流的视角（[第 5 章](../05-flow-based-perspective/index.md#ch-flow-based)），它基于归一化流的原理，将生成描述为连续变换，并通过流匹配的概念加以推广。

理解这些起源后，我们将建立一个连贯的框架，用于解释扩散模型的多种表述，并揭示将它们统一起来的深层原理。

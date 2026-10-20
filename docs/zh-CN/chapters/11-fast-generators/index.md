# 11. 从头学习快速生成器 {#ch-fast-scratch}

> *真理总能在简洁中找到，而非在事物的繁多与混乱之中。*
>
> — Isaac Newton

在[第 10 章](../10-distillation/index.md#ch-distillation)中，我们看到，扩散模型中缓慢的迭代采样器可以通过蒸馏压缩为少步生成器。从工程角度看，两阶段流程很实用，因为它将复杂的生成模型训练任务分解为清晰、独立的目标。第一阶段学习数据分布，第二阶段加速采样或提升质量。这种分离使每个阶段都能独立优化，从而令整个系统更易管理、更稳定，也更可靠。

本章将重点转向一个推动深度生成建模发展的核心问题：

<span id="question-11-0-1"></span>

**问题 11.0.1**

能否设计一个独立的生成原则，使训练稳定、高效，采样迅速，并且让用户轻松引导或控制生成内容？

本章沿着这一方向，讨论另一种方法：在*不*依赖预训练模型的情况下训练基于扩散的少步生成器。我们关注的是*流映射模型*。这里，*流映射*是经典的 ODE 解映射，其概念源于微分方程和动力系统，并且在连续介质力学和流体力学中，与跟踪物质粒子的运动密切相关。具体而言，${\boldsymbol{\Psi}}_{s\to t}$ 将时刻 $s$ 的状态，映射为在给定动力学下到达时刻 $t$ 时的状态。流映射模型直接学习近似这一映射，从而只用一次或少量网络计算便可跨越很长的轨迹[^1]。

这一表述为将概率质量从先验分布 $p_{\mathrm{prior}}$ 输运至数据分布 $p_{\mathrm{data}}$ 提供了具有理论依据的方法，同时在各中间时刻保持前向扩散过程指定的边缘分布 $p_t$。

**学完本章后，你将能够……**

-   区分瞬时速度场与其 ODE 解映射（流映射）；

-   理解精确 ODE 流的半群性质，以及它如何启发基于一致性的学习目标；

-   理解 Consistency Model、Consistency Trajectory Model 和 Mean Flow 如何作为代表性方法学习特殊或一般流映射；

-   区分 oracle 流映射恒等式与实际近似，并理解合适的参数化如何在实现少步生成的同时保留对局部速度信息的访问。

**阅读路线。** 本章的核心对象是 ODE 流映射。应将各个方法理解为学习同一底层对象的不同参数化与训练策略。

[^1]: 流映射的数学及历史背景见[公式 (4.2.2)](../04-score-sde-framework/02-matching-marginals.md#eq-flow-map-def)及其附近的讨论。

请继续阅读以下各节及[参考文献](../../references.md#references)。

[11.1 引言](01-prologue.md#sec-fast-scratch-prologue)

[11.2 特殊流映射：离散时间 Consistency Model](02-discrete-consistency-models.md#sec-discrete-cm)

[11.3 特殊流映射：连续时间 Consistency Model](03-continuous-consistency-models.md#sec-conti-cm)

[11.4 一般流映射：Consistency Trajectory Model](04-consistency-trajectory-models.md#sec-ctm)

[11.5 一般流映射：Mean Flow](05-mean-flow.md#sec-mf)

[11.6 结语](06-closing-remarks.md#sec-ch11-cr)

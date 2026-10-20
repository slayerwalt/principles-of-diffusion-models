# 5. 基于流的视角：从归一化流到流匹配 {#ch-flow-based}

> *万物流动。*
>
> — Heraclitus

作为概率论基石的*变量替换公式*（[Tabak and Vanden-Eijnden, 2010](../../references.md#ref-tabak2010density)；[Turner, 2013](../../references.md#ref-tabak2013family)），在现代生成建模中焕发了新的生命力。在[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)，我们看到 Fokker–Planck 方程描述概率密度如何在扩散过程中演化，而同样的边缘密度演化也可以通过对应的 PF-ODE 以确定性方式实现。这自然启发我们直接从确定性概率输运的视角研究生成建模。

**密度的变量替换公式。** 给定可逆变换 ${\mathbf{f}}$，${\mathbf{x}}={\mathbf{f}}({\mathbf{z}})$ 的密度为下式，其中 ${\mathbf{z}}\sim p_{\mathrm{prior}}$：

<span id="eq-nf-change-of-var"></span>

$$
\begin{aligned}
p({\mathbf{x}})
=
p_{\mathrm{prior}}({\mathbf{z}})
\left|
\det
\frac{\partial {\mathbf{f}}^{-1}({\mathbf{x}})}{\partial {\mathbf{x}}}
\right|,
\qquad
\text{where }
{\mathbf{z}}={\mathbf{f}}^{-1}({\mathbf{x}}).
\end{aligned}
\tag{5.0.1}
$$

当 ${\mathbf{f}}$ 可计算时，这个看似简单的公式允许双向输运样本与密度，构成了归一化流的基础；我们将在[第 5.1 节](01-normalizing-flows-and-neural-odes.md#sec-flow-based-method)介绍归一化流。不过，当从连续时间角度观察这一变换时，会发生什么？

本章沿这一思路，从归一化流走向神经 ODE，再到流匹配。与[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)的联系尤其关键。在 Score SDE 中，学得的 score 用于构造 PF-ODE 的速度场。流匹配则直接学习速度场。对于同一条扩散概率路径，它们提供了同一个确定性概率流的两种视角。

因此，可以用下图理解本章，其中两者依赖相同的条件化原理：

<span id="fig-score-fm-mental-picture"></span>

![Score SDE 与流匹配共享条件化原理和概率流](../../../en/assets/chapter-05/score-fm-mental-picture.png)

**图 5.1：** **Score SDE 与流匹配。** 两个框架都使用上排具有闭式表达式的*条件*量进行训练。对条件变量取平均（竖直箭头），得到下排不可计算的*边缘*量，这正是训练后模型恢复的量。两列的区别只在于所用的量不同。在扩散路径上，score 通过中间的转换确定速度，因此两条路线最终得到同一个 ODE；流匹配直接学习这一速度，并且不要求任一端点为 Gaussian 分布。

来源：作者绘制。

流匹配将这一原理推广到通常的 Gaussian 扩散设定之外。只要所选条件概率路径能够高效采样、相应条件速度可以计算，它就能在更一般的、可获得样本的源分布与目标分布之间构造连续输运。

为帮助扎实理解本章，[附录 B](../../appendices/b-density-evolution/index.md#app-continuity)提供了直观且自成一体的概述，介绍变量替换原理的不同形式：从通常的变量替换公式出发，到连续性方程，再到随机动力学下的 Fokker–Planck 方程。

**读完本章后，你将能够……**

-   通过输运概率分布这一共同思想，将归一化流、神经 ODE 与流匹配联系起来，从离散的可逆变换走向连续时间速度场；

-   认识到与[第 4 章](../04-score-sde-framework/index.md#ch-score-sde) Score SDE 框架之间的关键类比：对于同一扩散概率路径，Score SDE 学习决定 PF-ODE 速度的 score，而流匹配可以直接学习该速度场；两种情况下，都能通过可计算的条件目标学习不可计算的边缘目标；

-   理解连续性方程是连接 ODE 速度场与其概率分布演化的密度层面规则；

-   理解流匹配如何将这一条件化策略推广到 Gaussian 扩散路径之外，在可获得样本的一般源分布与目标分布之间进行输运，同时通过可计算的条件路径和速度保持无须模拟的训练；

-   区分指定的概率路径 $\{p_t\}_{t\in[0,1]}$ 与实现它的具体速度场、耦合及粒子轨迹，并理解为何不同的流可以共享相同的边缘分布；

-   理解构造可计算条件流的两种互补方式：从条件概率路径出发推导其速度，或从条件流映射出发推导沿轨迹的速度。

**阅读路线。** 已熟悉归一化流与神经 ODE 的读者，可以略读[第 5.1 节](01-normalizing-flows-and-neural-odes.md#sec-flow-based-method)，从[第 5.2.1 节](02-flow-matching-framework.md#subsec-lesson-score)开始。首次阅读时，重点关注 Score SDE 与流匹配视角之间的联系、连续性方程，以及从条件速度到边缘速度的构造。之后再回顾详细的路径构造与选读的理论性质。

继续阅读以下各节及[参考文献](../../references.md#references)。

[5.1 基于流的模型：归一化流与神经 ODE](01-normalizing-flows-and-neural-odes.md#sec-flow-based-method)

[5.2 流匹配框架](02-flow-matching-framework.md#sec-flow-matching-framework)

[5.3 构造分布之间的概率路径与速度](03-probability-paths-and-velocities.md#sec-instance-fm)

[5.4 （选读）标准仿射流的性质](04-canonical-affine-flow.md#sec-optimal-canonical)

[5.5 结语](05-closing-remarks.md#sec-ch5-cr)

<span id="sec-flow-based-method"></span>

[第 5.1 节](01-normalizing-flows-and-neural-odes.md#sec-flow-based-method)

<span id="sec-flow-matching-framework"></span>

[第 5.2 节](02-flow-matching-framework.md#sec-flow-matching-framework)

<span id="subsec-fm-framework"></span>

[第 5.2.2 节](02-flow-matching-framework.md#subsec-fm-framework)

<span id="eq-instant-change-of-var"></span>

[式 (5.1.9)](01-normalizing-flows-and-neural-odes.md#eq-instant-change-of-var)
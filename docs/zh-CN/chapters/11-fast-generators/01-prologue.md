# 11.1 引言 {#sec-fast-scratch-prologue}

<span id="fig-timeline"></span>

![学习流映射的扩散相关方法的时间线](../../../en/assets/chapter-11/timeline.png)

**图 11.1：** **学习流映射的扩散相关方法的时间线。** 底层的流映射是经典 ODE 解映射；此处所列日期仅指用于蒸馏、近似或学习该映射的现代机器学习方法。我们用<span style="color: #1F77B4">蓝色</span>表示特殊情形 ${\color{#1F77B4}{\boldsymbol{\Psi}}_{s\to 0}}$，用<span style="color: #FF7F0E">橙色</span>表示一般映射 ${\color{#FF7F0E}{\boldsymbol{\Psi}}_{s\to t}}$。

来源：作者绘制。

图中链接：[第 10.1 节](../10-distillation/01-prologue.md#sec-distillation-prologue) · [第 10.3 节](../10-distillation/03-progressive-distillation.md#sec-pd) · [第 11.2 节](02-discrete-consistency-models.md#sec-discrete-cm) · [第 11.4 节](04-consistency-trajectory-models.md#sec-ctm) · [第 11.3 节](03-continuous-consistency-models.md#sec-conti-cm) · [第 11.5 节](05-mean-flow.md#sec-mf)。

**流映射模型的动机。** 在[第 10 章](../10-distillation/index.md#ch-distillation)中，我们说明了如何通过蒸馏预训练扩散模型的知识，估计 oracle 流映射损失 $\mathcal{L}_{\mathrm{oracle}}({\boldsymbol{\theta}})$（见[公式 (10.1.4)](01-prologue.md#eq-general-cm-loss-rep)）中无法直接获得的回归目标，从而得到少步生成器。这条路线有效且实用：两阶段流程可通过工程设计获得稳健性，并且在数据效率和计算效率上通常仍具有竞争力。

本章将重点转向深度生成建模的一个更广泛的核心挑战：*能否建立一个独立的生成原则，在不依赖预训练模型的情况下，实现稳定、可扩展、高效的训练，快速采样，以及能够轻松按照用户意图引导的生成？* 设计这类独立原则是生成建模的核心任务。

扩散模型提供了一条有用的设计原则：以一个将数据逐渐变换为简单先验（噪声）的连续时间前向过程作为参照，将建模任务表述为学习恢复这一过程的逆时间输运，从而匹配所需的边缘分布。与一次性生成映射相比，这种依赖时间的表述也更便于在中间步骤引导生成过程。将其具体化为受扩散启发的方法，就得到如下问题：

<span id="question-11-1-1"></span>

**问题 11.1.1**

能否学习流映射 ${\boldsymbol{\Psi}}_{s\to t}(\cdot)$，用网络 ${\mathbf{G}}_{\boldsymbol{\theta}}(\cdot,s,t)$（*流映射模型*）实现这一映射，在不使用预训练模型的同时保持高保真生成？

<span id="fig-flow-map-illustration"></span>

![流映射沿 PF-ODE 的 oracle 轨迹输运状态](../../../en/assets/chapter-11/flow-map.png)

**图 11.2：** **流映射示意图。** 从任意状态 ${\mathbf{x}}_s$（位于时刻 $s$）出发，流映射 ${\boldsymbol{\Psi}}_{s \to t}$ 将其输运到同一条 PF-ODE oracle 轨迹上时刻 $t$ 的对应点。

来源：作者绘制。

本章围绕这一目标发展相关方法，并以一个共同目标组织论述。该目标同样是蒸馏的基础，为各种流映射表述提供了统一视角（[Boffi et al., 2025](../../references.md#ref-boffi2024flow)；[Hu et al., 2025](../../references.md#ref-hu2025cmt)）：

<span id="eq-general-cm-loss-rep"></span>

$$
\mathcal{L}_{\mathrm{oracle}}({\boldsymbol{\theta}})
  := \mathbb{E}_{s,t}\,\mathbb{E}_{{\mathbf{x}}_s \sim p_s}\Big[
     w(s,t)  d\big({\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t),  {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)\big)
  \Big].
\tag{10.1.4}
$$

这里，$s,t$ 从某个时间分布（例如均匀分布）采样，$w(s,t) \ge 0$ 为时间对 $(s,t)$ 分配权重，$d(\cdot,\cdot)$ 是差异度量，例如 $\ell_2$ 范数的平方。oracle 流映射 ${\boldsymbol{\Psi}}_{s\to t}$ 表示理想变换：它将状态 ${\mathbf{x}}_s$ 从时刻 $s$ 直接输运到时刻 $t$：

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s) = {\mathbf{x}}_s + \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathrm{d}u,
$$

其中 oracle 漂移为

$$
{\mathbf{v}}^*({\mathbf{x}}_u,u) = {\mathbb{E}} \left[\alpha_u'{\mathbf{x}}_0+ \sigma_u'{\boldsymbol{\epsilon}} | {\mathbf{x}}_u\right],
$$

也可以使用等价的参数化（见[第 6 章](../06-unified-perspective/index.md#ch-unified-perspective)），常见选择包括 ${\mathbf{x}}$ 预测和 ${\mathbf{v}}$ 预测形式。

在 oracle 损失的最优处，学到的模型精确恢复真实流映射：

$$
{\mathbf{G}}^*({\mathbf{x}}_s, s, t) = {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s), \quad \text{for all }\, s,t, \text{ and }\, {\mathbf{x}}_s \sim p_s.
$$

由于流映射 ${\boldsymbol{\Psi}}_{s\to t}$ 无法以闭式表示，必须进行近似。[第 10 章](../10-distillation/index.md#ch-distillation)讨论的一种选择是依赖预训练扩散模型。另一种选择是引入新的、更易计算的替代目标，本章将对此展开介绍。为清晰起见，可以根据训练循环是否查询教师，将现有方法大致分为两类：*蒸馏*显式调用教师模型；*从头训练*则通过构造自足的替代目标，避免调用教师。

在这一具有理论依据的目标之上，我们将讨论学习流映射模型的系统性方法，旨在开发兼具实用性和计算效率、且生成结果能更准确反映真实数据分布的方法。首先对这一范式作总体介绍。

**特殊流映射：一致性函数。** *Consistency Models*（[Song et al., 2023](../../references.md#ref-song2023consistency)）是流映射学习最早的开创性方法之一。它们学习少步去噪器 ${\mathbf{f}}_{\boldsymbol{\theta}}(\cdot,s)$，近似流映射到时间原点这一特殊情形：

$$
{\boldsymbol{\Psi}}_{s\to 0}(\cdot), \qquad s \in (0,T].
$$

其关键思想是，将每个状态 ${\mathbf{x}}_s$ 映射到穿过该状态的 PF-ODE 轨迹在时刻 $0$ 的终点 ${\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)$。形式上，CM 方法族（[Song et al., 2023](../../references.md#ref-song2023consistency)；[Song and Dhariwal, 2024](../../references.md#ref-songimproved)；[Geng et al., 2024](../../references.md#ref-geng2024consistency)；[Lu and Song, 2024](../../references.md#ref-lu2024simplifying)）的 oracle 训练目标为

<span id="eq-oracle-cm"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{oracle}\text{-}\mathrm{CM}}({\boldsymbol{\theta}})
    := {\mathbb{E}}_{s} {\mathbb{E}}_{{\mathbf{x}}_s\sim p_s} \left[ w(s)  d \left({\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s), {\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)\right)\right].
\end{aligned}
\tag{11.1.0}
$$

但在实践中，oracle ${\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)$ 无法获得。因此，用*停止梯度*的目标 ${\mathbf{f}}_{{\boldsymbol{\theta}}^-}$ 替代它，该目标取自同一轨迹上略早的步骤 ${\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)$：

$$
{\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s) \approx {\mathbf{f}}_{{\boldsymbol{\theta}}^-} \left({\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s),\,s-\Delta s\right), \quad \Delta s > 0,
$$

其中 ${\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)$ 本身也必须近似。有两种实用策略：（i）*蒸馏*，依赖预训练扩散模型；（ii）*从头训练*，在没有教师引导的情况下使用单点估计。

**一般流映射。** 两种代表性方法是 *Consistency Trajectory Model*（CTM）和 *Mean Flow*（MF）。

**Consistency Trajectory Models。** *Consistency Trajectory Model*（CTM）<br>
（[Kim et al., 2024](../../references.md#ref-kim2023consistency)）是首个学习任意起止时刻之间一般流映射 ${\boldsymbol{\Psi}}_{s\to t}$ 的工作，可以看作[公式 (10.1.4)](01-prologue.md#eq-general-cm-loss-rep)统一目标下的一个具体实例。CTM 采用受 Euler 方法启发的参数化，将 oracle 流映射写为

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
:= {\mathbf{x}}_s + \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\mathop{}\!\mathrm{d} u
= \frac{t}{s} {\mathbf{x}}_s
+ \frac{s-t}{s}
\underbrace{\Big[{\mathbf{x}}_s + \frac{s}{s-t} \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\Big]}_{\approx~{\mathbf{g}}_{{\boldsymbol{\theta}}}},
\end{aligned}
$$

由此得到神经网络参数化

$$
{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t)
:= \frac{t}{s}\,{\mathbf{x}}_s + \frac{s-t}{s}\,{\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t),
$$

其中 ${\mathbf{g}}_{\boldsymbol{\theta}}$ 是通过训练使 ${\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)\approx {\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t)$ 成立的神经网络。

由于 oracle ${\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$ 无法获得，CTM 使用在中间时刻 $u$ 计算的*停止梯度*目标进行训练：

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
 \approx
{\mathbf{G}}_{{\boldsymbol{\theta}}^-} \big({\boldsymbol{\Psi}}_{s\to u}({\mathbf{x}}_s),\,u,\,t\big),
\qquad u\in[t,s],
$$

其中中间状态 ${\boldsymbol{\Psi}}_{s\to u}({\mathbf{x}}_s)$ 用以下两种方式之一近似：（i）*蒸馏*，对预训练扩散教师应用少步求解器；（ii）*从头训练*，直接通过 ${\mathbf{G}}_{\boldsymbol{\theta}}$ 参数化构造由自身诱导的教师。

**Mean Flow。** Mean Flow（MF）（[Geng et al., 2025](../../references.md#ref-geng2025mean)）以流匹配为基础，对区间 $[t,s]$（其中 $t\le s$）上的*平均漂移*进行建模：

$$
{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t)
 \approx
{\mathbf{h}}^*({\mathbf{x}}_s,s,t)
:= \frac{1}{\,t-s\,}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u,
$$

这同样符合[公式 (10.1.4)](01-prologue.md#eq-general-cm-loss-rep)。对恒等式

$$
(t-s)\,{\mathbf{h}}^*({\mathbf{x}}_s,s,t)  =  \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u
$$

关于 $s$ 求导，可以得到一个自指关系，由此引出 MF 目标

$$
\mathcal{L}_{\mathrm{MF}}({\boldsymbol{\theta}})
:= {\mathbb{E}}_s\,{\mathbb{E}}_{{\mathbf{x}}_s\sim p_s} \Big[w(s)\,\big\|
{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t)-{\mathbf{h}}_{{\boldsymbol{\theta}}^-}^{\mathrm{tgt}}({\mathbf{x}}_s,s,t)
\big\|_2^2\Big],
$$

其停止梯度目标为

$$
{\mathbf{h}}_{{\boldsymbol{\theta}}^-}^{\mathrm{tgt}}({\mathbf{x}}_s,s,t)
:= {\mathbf{v}}^*({\mathbf{x}}_s,s)  -  (s-t) \left((\partial_{\mathbf{x}} {\mathbf{h}}_{{\boldsymbol{\theta}}^-})\,{\mathbf{v}}^*({\mathbf{x}}_s,s) + \partial_s {\mathbf{h}}_{{\boldsymbol{\theta}}^-}\right).
$$

实践中，oracle 速度 ${\mathbf{v}}^*({\mathbf{x}}_s,s)$ 同样需要近似。两种常见策略是：（i）*蒸馏*，使用通过流匹配训练的预训练扩散模型；（ii）*从头训练*，使用由前向扰动过程导出的单点条件速度 $\alpha'_s\,{\mathbf{x}}_0+\sigma'_s\,{\boldsymbol{\epsilon}}$，该过程为 ${\mathbf{x}}_s = \alpha_s {\mathbf{x}}_0+\sigma_s {\boldsymbol{\epsilon}}$。

**CTM 与 MF 的关系。** CTM 和 MF 近似同一个路径积分，但对其采用不同的替代参数化：

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
&:= {\mathbf{x}}_s  +  \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u
\\[2pt]
&= \frac{t}{s}\,{\mathbf{x}}_s
 +  \frac{s-t}{s}\,
\underbrace{\Big[{\mathbf{x}}_s + \frac{s}{\,s-t\,} \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u\Big]}_{\approx~{\mathbf{g}}_{{\boldsymbol{\theta}}} }
\\[4pt]
&= {\mathbf{x}}_s  +  (t-s)\,
\underbrace{\Big[\frac{1}{\,t-s\,} \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u\Big]}_{\approx~{\mathbf{h}}_{{\boldsymbol{\theta}}} }.
\end{aligned}
$$

换言之，CTM 通过 ${\mathbf{g}}_{\boldsymbol{\theta}}$ 学习一个*斜率位移*，MF 则学习*平均漂移* ${\mathbf{h}}_{\boldsymbol{\theta}}$；两者都是近似同一个积分的一致方式，该积分定义了 ${\boldsymbol{\Psi}}_{s\to t}$。

**接下来的内容。** 我们首先介绍 CM 方法族，它关注特定流映射 ${\boldsymbol{\Psi}}_{s\to 0}$。这部分涵盖其在[第 11.2 节](02-discrete-consistency-models.md#sec-discrete-cm)中的离散时间起点，以及在[第 11.3 节](03-continuous-consistency-models.md#sec-conti-cm)中的连续时间扩展。随后转向一般流映射，详细讨论两个关键代表 CTM 和 MF。[第 11.4 节](04-consistency-trajectory-models.md#sec-ctm)和[第 11.5 节](05-mean-flow.md#sec-mf)分别介绍它们的参数化、训练策略与实际近似。

需要说明的是，[第 D.6 节](../../appendices/d-supplementary-proofs/06-edm.md#sec-edm)介绍的 *Elucidating Diffusion Model*（EDM）为 ${\mathbf{x}}$ 预测模型的网络参数化设计提供了系统指导，并已展现出良好的实证表现。虽然这一节可以作为选读内容，但 EDM 表述为 CM 风格的模型提供了有价值的基础。

为使后续论述清晰，我们并不严格按照这些方法出现的时间顺序展开，而是按概念关系组织内容。不过，为承认原创贡献并尊重时间先后，我们在[图 11.1](01-prologue.md#fig-timeline)中给出了历史时间线。

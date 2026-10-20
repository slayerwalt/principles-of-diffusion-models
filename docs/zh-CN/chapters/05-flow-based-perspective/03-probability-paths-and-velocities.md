# 5.3 构造分布之间的概率路径与速度 {#sec-instance-fm}

流匹配的本质是将源分布逐渐转化为目标分布。为了引导这一转化，需要两个关键对象：*概率路径* $p_{t}$，给出每个时刻 $t$ 演化分布的快照；*速度场* ${\mathbf{v}}_{t}$，描述各个粒子如何沿路径运动。二者并不独立，而是通过连续性方程联系起来，该方程保证粒子动力学与分布演化一致。因此，学习任务归结为寻找忠实驱动这一过程的速度场 ${\mathbf{v}}_t$。困难在于，对于一般的复杂分布，真实边缘速度 ${\mathbf{v}}_t$ 未知，因此目标不可计算，无法直接获得。

条件流匹配的核心思想，是构造人为设计但可计算的过程，解决真实边缘速度不可计算的问题。为此，引入条件变量 ${\mathbf{z}}$，设计条件速度 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})$ 和／或条件路径 $p_t({\mathbf{x}}_t|{\mathbf{z}})$，并有意使它们保持简单。

由于这些条件对象具有已知闭式形式，可以用作模型回归的替代目标。只要满足两个实际要求，就能得到有效训练损失 $\mathcal{L}_{\mathrm{CFM}}$：(i) 能够从 $p_t(\cdot|{\mathbf{z}})$ 高效采样；(ii) 相应速度 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 具有闭式表达式。

应如何设计性质良好的条件过程？我们从一个完全清楚的情形——Gaussian 到 Gaussian 的桥接（[第 5.3.1 节](03-probability-paths-and-velocities.md#subsec-special-fm-instantiations)）获得启发。该示例展示两种自然设计策略：在每个时刻 $t$ 采用 Gaussian 概率路径，或指定仿射速度场，两者都可解析处理。

在这一认识的指导下，我们采用两种互补构造视角，将其扩展到一般端点分布（另见[第 B.1.2 节](../../appendices/b-density-evolution/01-change-of-variable.md#subsec-cov-continuity-eq)）。二者对应输运的经典 Euler 描述与 Lagrange 描述：

-   **先构造条件概率路径（Euler 视角）。** 从条件密度路径 $p_t(\cdot|{\mathbf{z}})$ 出发，寻找满足连续性方程的条件速度场 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$。由于单独的密度路径通常不能唯一确定速度，这一构造还需要选择一个具体的相容速度场。

-   **先构造条件流（Lagrange 视角）。** 直接通过条件流映射 ${\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}})$ 指定各个粒子如何运动。诱导密度由推前得到；当流映射可逆时，对应的 Euler 速度场为

    $$
    {\mathbf{v}}_t
        =
        \partial_t{\boldsymbol{\Psi}}_{0\to t}
        \circ
        {\boldsymbol{\Psi}}_{0\to t}^{-1}.
    $$

[第 5.3.2 节](03-probability-paths-and-velocities.md#subsec-fm-instantiations)详细介绍第一种方法，展示它与[第 5.2.1 节](02-flow-matching-framework.md#subsec-lesson-score)讨论的扩散模型构造之间的紧密类比；[第 5.3.3 节](03-probability-paths-and-velocities.md#subsec-prob-path-flow)介绍第二种方法。两种视角共同提供定义 $p_t({\mathbf{x}}_t| {\mathbf{z}})$ 与 ${\mathbf{v}}_t({\mathbf{x}}_t| {\mathbf{z}})$ 的实用框架，支持无须模拟的训练，以及任意源分布与目标分布之间的流构造。

## 5.3.1 关键特例：Gaussian 到 Gaussian 桥接中的边缘分布 $p_t({\mathbf{x}}_t)$ 与速度 ${\mathbf{v}}_t({\mathbf{x}}_t)$ {#subsec-special-fm-instantiations}

从 Gaussian 端点情形开始，此时能够解析计算边缘密度 $p_t({\mathbf{x}}_t)$ 与速度场 ${\mathbf{v}}_t({\mathbf{x}}_t)$。它为一般条件密度 $p_t({\mathbf{x}}_t |{\mathbf{z}}_t)$ 与速度场 ${\mathbf{v}}_t({\mathbf{x}}_t |{\mathbf{z}}_t)$ 的构造提供模板。

当源分布 $p_{\text{src}}$ 与目标分布 $p_{\text{tgt}}$ 都是 Gaussian 分布时，速度场 ${\mathbf{v}}_t(\cdot)$ 具有闭式表达式。考虑插值边缘密度路径：

<span id="eq-gaussian-density-path"></span>

$$
\begin{aligned}
p_t({\mathbf{x}}_t) = \mathcal{N}\left({\mathbf{x}}_t; \boldsymbol{\mu}(t), \sigma^2(t) {\mathbf{I}}\right),
\end{aligned}
\tag{5.3.1}
$$

其中均值 $\boldsymbol{\mu}(t)$ 与方差 $\sigma^2(t)>0$ 随时间变化。两个端点为

$$
p_{\text{src}} = p_0 =  \mathcal{N}\left({\mathbf{x}}; \boldsymbol{\mu}(0), \sigma^2(0) {\mathbf{I}}\right), \quad
p_{\text{tgt}} = p_1 =  \mathcal{N}\left({\mathbf{x}}; \boldsymbol{\mu}(1), \sigma^2(1) {\mathbf{I}}\right),
$$

因此，路径 $\{p_t\}_{t \in [0,1]}$ 连接这些分布。

对于给定路径 $\{p_t\}_{t \in [0,1]}$，确实存在许多速度场，可以诱导 ODE 流 $\boldsymbol{\Psi}_{0\to t}({\mathbf{x}})$，使 ${\mathbf{x}} \sim p_0$ 蕴含 $\boldsymbol{\Psi}_{0\to t}({\mathbf{x}}) \sim p_t$。对于这一 Gaussian 路径，一个特别简单的实现为[^1]：

<span id="eq-gaussian-sol-map"></span>

$$
\begin{aligned}
\boldsymbol{\Psi}_{0\to t}({\mathbf{x}}) := \boldsymbol{\mu}(t) + \sigma(t) \left( \frac{{\mathbf{x}} - \boldsymbol{\mu}(0)}{\sigma(0)} \right).
\end{aligned}
\tag{5.3.2}
$$

对于定义好的 Gaussian 路径 $p_t$（对所有 $t$ 都是 Gaussian），诱导[式 (5.3.2)](03-probability-paths-and-velocities.md#eq-gaussian-sol-map)中 ODE 流的速度场 ${\mathbf{v}}_t(\cdot)$，可唯一地解析刻画如下（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)）：

<span id="closed-form-vf"></span>

**命题 5.3.1：Gaussian 密度路径的闭式速度场**

令 $p_t$ 为[式 (5.3.1)](03-probability-paths-and-velocities.md#eq-gaussian-density-path)中的 Gaussian 路径。则生成[式 (5.3.2)](03-probability-paths-and-velocities.md#eq-gaussian-sol-map)中 ODE 流的速度场 ${\mathbf{v}}_t(\cdot)$，对于定义好的 ${\boldsymbol{\Psi}}_{0\to t}$ 是唯一的，且具有闭式表达式：

$$
{\mathbf{v}}_t({\mathbf{x}}) = \frac{\sigma'(t)}{\sigma(t)} \left( {\mathbf{x}} - \boldsymbol{\mu}(t) \right) + \boldsymbol{\mu}'(t).
$$

**命题证明。**

考虑初始条件为 ${\mathbf{y}}$ 的 ODE：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \boldsymbol{\Psi}_{0\to t}({\mathbf{y}}) = {\mathbf{v}}_t(\boldsymbol{\Psi}_{0\to t}({\mathbf{y}})).
$$

由于 $\boldsymbol{\Psi}_{0\to t}$ 可逆（因为 $\sigma(t)>0$），可以令 ${\mathbf{x}} = \boldsymbol{\Psi}_{0\to t}({\mathbf{y}})$ 与 ${\mathbf{y}} = \boldsymbol{\Psi}_{0\to t}^{-1}({\mathbf{x}}) = \boldsymbol{\Psi}_{t\to 0}({\mathbf{x}})$，得到

$$
\boldsymbol{\Psi}_{0\to t}'\big(\boldsymbol{\Psi}_{0\to t}^{-1}({\mathbf{x}})\big) = {\mathbf{v}}_t({\mathbf{x}}).
$$

对[式 (5.3.2)](03-probability-paths-and-velocities.md#eq-gaussian-sol-map)关于 $t$ 求导，得到

$$
\boldsymbol{\Psi}_{0\to t}'({\mathbf{x}}) = \boldsymbol{\mu}'(t) + \sigma'(t) \left( \frac{{\mathbf{x}} - \boldsymbol{\mu}(0)}{\sigma(0)} \right).
$$

解出 ${\mathbf{y}} = \boldsymbol{\Psi}_{0\to t}^{-1}({\mathbf{x}})$，得到

$$
{\mathbf{y}} = \boldsymbol{\mu}(0) + \sigma(0) \left( \frac{{\mathbf{x}} - \boldsymbol{\mu}(t)}{\sigma(t)} \right).
$$

将其代入 $\boldsymbol{\Psi}_{0\to t}'({\mathbf{x}})$，得到

$$
{\mathbf{v}}_t({\mathbf{x}}) = \frac{\sigma'(t)}{\sigma(t)} \left( {\mathbf{x}} - \boldsymbol{\mu}(t) \right) + \boldsymbol{\mu}'(t),
$$

即所需结论。

对于固定流映射 ${\boldsymbol{\Psi}}_{0\to t}$（先构造流的视角），速度由下式唯一确定：

$$
{\mathbf{v}}_t  =  \partial_t {\boldsymbol{\Psi}}_{0\to t} \circ {\boldsymbol{\Psi}}_{0\to t}^{-1}.
$$

在这一构造下，二元组 $(p_t,{\mathbf{v}}_t)$ 自动满足连续性方程。相比之下，对于给定密度路径 $t\mapsto p_t$，若不固定 ${\boldsymbol{\Psi}}_{0\to t}$（先构造概率路径的视角），速度场并不唯一。

这一区别精确刻画了先构造流与先构造概率路径两种视角的不同。

以潜变量 ${\mathbf{z}}$ 为条件时，这一闭式刻画仍然成立。下文将扩展这一认识，为一般边缘分布情形构造条件 Gaussian 路径 $p_t(\cdot|{\mathbf{z}})$，并推导相应的条件速度场 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$。

## 5.3.2 先构造条件概率路径来得到 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 与 $p_t(\cdot|{\mathbf{z}})$ {#subsec-fm-instantiations}

<span id="fig-conditional-path"></span>

![两种常见的条件概率路径](../../../en/assets/chapter-05/conditional-path.png)

**图 5.9：** **两种常见条件概率路径的示意图。** 包括：(i) *双侧*，以 ${\mathbf{x}}_0 \sim p_{\text{tgt}}$ 与 ${\mathbf{x}}_1 \sim p_{\text{src}}$ 为条件，端点分布可以是一般分布；(ii) *单侧*，以 ${\mathbf{x}}_0 \sim p_{\text{tgt}}$ 或 ${\mathbf{x}}_1 \sim p_{\text{src}}$ 为条件。

来源：作者绘制。

我们希望先构造条件密度路径 $p_t(\cdot|{\mathbf{z}})$，再推导相应的条件速度场 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$（通过[命题 5.3.1](03-probability-paths-and-velocities.md#closed-form-vf)），其中以 $\pi({\mathbf{z}})$ 为条件。根据 ${\mathbf{z}}$ 的选择，有两种自然情形：(i) 以 ${\mathbf{z}}=({\mathbf{x}}_0,{\mathbf{x}}_1)$ 进行双侧条件化；(ii) 以 ${\mathbf{z}}={\mathbf{x}}_0$ 或 ${\mathbf{x}}_1$ 进行单侧条件化。无论哪种情形，构造都必须匹配边界分布：

$$
\begin{aligned}
p_{\mathrm{src}}({\mathbf{x}}_0) = \int p_0({\mathbf{x}}_0|{\mathbf{z}}) \pi({\mathbf{z}})\mathop{}\!\mathrm{d}{\mathbf{z}}, \quad
   p_{\mathrm{tgt}}({\mathbf{x}}_1) = \int p_1({\mathbf{x}}_1|{\mathbf{z}}) \pi({\mathbf{z}}) \mathop{}\!\mathrm{d}{\mathbf{z}}.
\end{aligned}
$$

一旦指定具体构造，验证这些约束就很直接，因此这里不重点讨论验证步骤。

**I. 双侧 ${\mathbf{z}} = ({\mathbf{x}}_0, {\mathbf{x}}_1)$：“光束状”路径。**

**$\pi({\mathbf{z}})$ 的选择。** 考虑一般分布 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$，定义在 $\mathbb{R}^D$ 上。令 ${\mathbf{z}} = ({\mathbf{x}}_0, {\mathbf{x}}_1)$，其中 ${\mathbf{x}}_0 \sim p_{\mathrm{src}}$ 与 ${\mathbf{x}}_1 \sim p_{\mathrm{tgt}}$ 独立，即

$$
\pi({\mathbf{z}}) = p_{\mathrm{src}}({\mathbf{x}}_0)  p_{\mathrm{tgt}}({\mathbf{x}}_1).
$$

**条件路径 $p_t(\cdot|{\mathbf{z}})$ 的选择。** 通过具有固定小方差 $\sigma > 0$ 的线性插值定义条件路径[^2]：

$$
p_t({\mathbf{x}}_t | {\mathbf{z}} = ({\mathbf{x}}_0, {\mathbf{x}}_1)) = \mathcal{N}({\mathbf{x}}_t; a_t{\mathbf{x}}_0 + b_t{\mathbf{x}}_1,  \sigma^2 {\mathbf{I}}),
$$

其中 $a_t$ 与 $b_t$ 是随时间变化的函数，满足 $a_0 = 1, b_0 = 0$ 与 $a_1 = 0, b_1 = 1$。（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)；[Liu, 2022](../../references.md#ref-liu2022rectified)）建议选择 $a_t = 1-t$、$b_t = t$。在确定性情形 $\sigma = 0$ 下，得到

$$
p_t({\mathbf{x}}_t |{\mathbf{z}}) = \delta \big({\mathbf{x}}_t - [a_t{\mathbf{x}}_0 + b_t{\mathbf{x}}_1]\big),
$$

它描述从 ${\mathbf{x}}_0$ 到 ${\mathbf{x}}_1$ 的确定性插值路径。

**导出的条件速度 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$。** 由[命题 5.3.1](03-probability-paths-and-velocities.md#closed-form-vf)，条件速度为

$$
{\mathbf{v}}_t({\mathbf{x}} | {\mathbf{z}}) = a'_t{\mathbf{x}}_0 + b'_t{\mathbf{x}}_1.
$$

**CFM 损失。** 当 $\sigma=0$，从而 ${\mathbf{x}}_t = a_t{\mathbf{x}}_0 + b_t{\mathbf{x}}_1$ 时，CFM 损失化为

$$
\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{t, {\mathbf{x}}_0\sim p_{\text{src}}, {\mathbf{x}}_1\sim p_{\text{tgt}}} \left\| {\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \left(a'_t{\mathbf{x}}_0 + b'_t{\mathbf{x}}_1\right) \right\|^2.
$$

由[式 (5.2.9)](02-flow-matching-framework.md#eq-marginal-oracle-velocity)与[式 (5.2.11)](02-flow-matching-framework.md#eq-minimum-velocity)，最优速度场为

$$
{\mathbf{v}}^*({\mathbf{x}}_t, t) = \mathbb{E} \left[ {\mathbf{x}}_t'|{\mathbf{x}}_t \right] = \mathbb{E} \left[ a'_t{\mathbf{x}}_0 + b'_t{\mathbf{x}}_1|{\mathbf{x}}_t \right].
$$

这里，期望对 $p(\mathbf{x}_0, \mathbf{x}_1|\mathbf{x}_t)$ 求取，它是源与目标样本对 $(\mathbf{x}_0, \mathbf{x}_1)$ 的条件分布，这些样本对可能产生观测插值 $\mathbf{x}_t = a_t\mathbf{x}_0 + b_t\mathbf{x}_1$，对应时刻 $t$。

**II. 单侧 ${\mathbf{z}} = {\mathbf{x}}_0$ 或 ${\mathbf{x}}_1$：“聚光灯状”路径。** 我们在单侧设定下展示先构造条件概率路径的方法，考虑满足 $p_{\mathrm{src}}=\mathcal N(\boldsymbol 0,{\mathbf{I}})$ 与 $p_{\mathrm{tgt}}=p_{\mathrm{data}}$ 的标准生成设定。关键在于，这一 Gaussian 源并非额外假设，而是下文定义的条件路径的直接结果。[第 5.3.3 节](03-probability-paths-and-velocities.md#subsec-prob-path-flow)将对任意端点作更一般的处理。

**$\pi({\mathbf{z}})$ 的选择。** 取 ${\mathbf{z}}={\mathbf{x}}_1$，其中 $\pi({\mathbf{z}})=p_{\mathrm{data}}({\mathbf{x}}_1)$（${\mathbf{z}}={\mathbf{x}}_0\sim p_{\mathrm{prior}}$ 的情形类似）。

**条件路径 $p_t(\cdot|{\mathbf{z}})$ 的选择。** 对固定 ${\mathbf{x}}_1\sim p_{\mathrm{data}}$，定义

$$
p_t({\mathbf{x}}_t|{\mathbf{z}}={\mathbf{x}}_1)=\mathcal N \big({\mathbf{x}}_t; b_t {\mathbf{x}}_1, a_t^2 {\mathbf{I}}\big),
$$

其中 $a_0=1, b_0=0, a_1=0, b_1=1$（通常理解为极限）。在边界处，

$$
p_0(\cdot|{\mathbf{z}}={\mathbf{x}}_1)=\mathcal N(\cdot; \boldsymbol 0,{\mathbf{I}}),\qquad
p_1(\cdot|{\mathbf{z}}={\mathbf{x}}_1)=\delta(\cdot - {\mathbf{x}}_1).
$$

对 ${\mathbf{x}}_1$ 边缘化，得到 $\{p_t\}_{t\in[0,1]}$，满足 $p_0=\mathcal N(\boldsymbol 0,{\mathbf{I}})$（与 $p_{\mathrm{data}}$ 无关）以及 $p_1=p_{\mathrm{data}}$。

**导出的条件速度 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$。** 对于 $t\in(0,1)$，其中 $a_t>0$，将[命题 5.3.1](03-probability-paths-and-velocities.md#closed-form-vf)应用于条件 Gaussian 路径，得到

$$
\mathbf v_t({\mathbf{x}}|{\mathbf{x}}_1)
= b_t' {\mathbf{x}}_1 + \frac{a_t'}{a_t} \big({\mathbf{x}} - b_t {\mathbf{x}}_1\big).
$$

**单侧 CFM 目标。** 使用 $t\sim\mathcal U(0,1)$（或任意固定采样分布）与 ${\mathbf{x}}_1\sim p_{\mathrm{data}}$，CFM 损失成为

<span id="eq-one-sided-cfm"></span>

$$
\begin{aligned}
\mathcal L_{\mathrm{CFM}}
={\mathbb{E}}_{t,{\mathbf{x}}_1} {\mathbb{E}}_{{\mathbf{x}}_t\sim p_t(\cdot|{\mathbf{x}}_1)}
\Big\|\mathbf v_\phi({\mathbf{x}}_t,t)-
\Big[b_t'{\mathbf{x}}_1+\tfrac{a_t'}{a_t}\big({\mathbf{x}}_t-b_t{\mathbf{x}}_1\big)\Big]\Big\|_2^2.
\end{aligned}
\tag{5.3.3}
$$

由 MSE 最优性，唯一最优解是边缘速度场

$$
\mathbf v^*({\mathbf{x}},t)
= {\mathbb{E}} \left[\mathbf v_t({\mathbf{x}}|{\mathbf{x}}_1) |dle|  {\mathbf{x}}_t={\mathbf{x}}\right]
= {\mathbb{E}} \left[a_t'{\mathbf{x}}_0+b_t'{\mathbf{x}}_1 |dle|  {\mathbf{x}}_t={\mathbf{x}}\right].
$$

**与双侧目标的等价性。** 对于成对样本 $({\mathbf{x}}_0,{\mathbf{x}}_1)$，其中 ${\mathbf{x}}_t=a_t{\mathbf{x}}_0+b_t{\mathbf{x}}_1$，

$$
\mathbf v_t({\mathbf{x}}_t|{\mathbf{x}}_1)
= b_t'{\mathbf{x}}_1 + \tfrac{a_t'}{a_t}\big({\mathbf{x}}_t-b_t{\mathbf{x}}_1\big)
= a_t'{\mathbf{x}}_0 + b_t'{\mathbf{x}}_1.
$$

因此，单侧损失回归的是给定 ${\mathbf{x}}_t$ 时双侧目标的*条件期望*：

$$
\mathbf v^*({\mathbf{x}},t)={\mathbb{E}} \big[a_t'{\mathbf{x}}_0 + b_t'{\mathbf{x}}_1 | {\mathbf{x}}_t={\mathbf{x}}\big],
$$

所以单侧与双侧 CFM 目标具有相同的最优解。

**Gaussian FM = 扩散模型。** 我们采用 FM 约定，$t=0$ 表示源／先验，$t=1$ 表示目标／数据：

$$
p_{\mathrm{src}}=p_{\mathrm{prior}},\qquad p_{\mathrm{tgt}}=p_{\mathrm{data}}.
$$

相比之下，扩散模型通常按从数据到噪声的方向索引时间（即 $t=0$ 为数据，$t=1$ 为先验）。本节主体讨论采用 FM 时间约定。不过，为便于比较，[表 5.2](03-probability-paths-and-velocities.md#tb-interpolant-instances)中的 VE 与 VP 项保留通常的扩散时间索引，其中 $t=0$ 对应数据，$t=1$ 对应先验／噪声。若进一步有 $p_{\mathrm{src}}=\mathcal N(\boldsymbol 0,{\mathbf{I}})$，则对于固定条件 ${\mathbf{x}}_1\sim p_{\mathrm{data}}$，自然选择 Gaussian 条件路径 $p_t(\cdot |{\mathbf{x}}_1)$，而目标分布 $p_{\mathrm{tgt}}$ 本身不必是 Gaussian。我们称这一 Gaussian 条件路径设定为 *Gaussian FM*。扩散概率路径是此类 Gaussian 路径的重要子类。不过，一般 Gaussian 调度未必对应有效的前向扩散 SDE；调度还必须满足[引理 4.5.1](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#forward-sde)的 SDE 相容条件。

选择 $a_t = 1-t$ 与 $b_t = t$（等价地，为 $\alpha_t = t$ 与 $\sigma_t = 1-t$，这里按扩散模型中的 $a_t:=\sigma_t$、$b_t:=\alpha_t$ 重新标记），即可恢复熟悉的 FM/RF 调度（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)；[Liu, 2022](../../references.md#ref-liu2022rectified)）。

<span id="tb-interpolant-instances"></span>

**表 5.2：** 不同插值形式的总结，写作 ${\mathbf{x}}_t=a_t{\mathbf{x}}_0+b_t{\mathbf{x}}_1$，其中 ${\mathbf{x}}_0\sim p_{\mathrm{src}}=p_{\mathrm{prior}}$ 与 ${\mathbf{x}}_1\sim p_{\mathrm{tgt}}=p_{\mathrm{data}}$。FM/RF 与三角函数两列采用 FM 约定（$t=0$ 为源，$t=1$ 为目标），VE/VP 两列则保留通常的扩散时间约定（$t=0$ 为数据，$t=1$ 为先验／噪声），其中 $a_t:=\sigma_t$ 与 $b_t:=\alpha_t$。对于具有有限终端噪声水平的 VE，精确的终端边缘分布为 $p_{\mathrm{data}}*\mathcal{N}(\boldsymbol{0},a_1^2{\mathbf{I}})$；表中的 Gaussian 先验是终端噪声相对数据尺度占主导时通常采用的近似。

<div style="overflow-x: auto;" markdown="1">
<table markdown="1">
<tr markdown="1">
<th markdown="block">



</th>
<th markdown="block">

**VE**

</th>
<th markdown="block">

**VP**

</th>
<th markdown="block">

**FM/RF**

</th>
<th markdown="block">

**三角函数**（[Albergo et al., 2023](../../references.md#ref-albergo2023stochastic)）

</th>
</tr>
<tr markdown="1">
<td markdown="block">

$a_t$（先验系数）

</td>
<td markdown="block">

$a_t$

</td>
<td markdown="block">

$\sqrt{1-b_t^2}$

</td>
<td markdown="block">

$1-t$

</td>
<td markdown="block">

$\cos \bigl(\tfrac{\pi}{2}t\bigr)$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

$b_t$（数据系数）

</td>
<td markdown="block">

$1$

</td>
<td markdown="block">

$b_t$

</td>
<td markdown="block">

$t$

</td>
<td markdown="block">

$\sin \bigl(\tfrac{\pi}{2}t\bigr)$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

$a_0$

</td>
<td markdown="block">

$0$

</td>
<td markdown="block">

$0$

</td>
<td markdown="block">

$1$

</td>
<td markdown="block">

$1$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

$b_0$

</td>
<td markdown="block">

$1$

</td>
<td markdown="block">

$1$

</td>
<td markdown="block">

$0$

</td>
<td markdown="block">

$0$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

$a_1$

</td>
<td markdown="block">

$a_1$

</td>
<td markdown="block">

$1$

</td>
<td markdown="block">

$0$

</td>
<td markdown="block">

$0$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

$b_1$

</td>
<td markdown="block">

$1$

</td>
<td markdown="block">

$0$

</td>
<td markdown="block">

$1$

</td>
<td markdown="block">

$1$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

$p_{\text{prior}}$

</td>
<td markdown="block">

$p_{\mathrm{data}} * \mathcal{N}(\boldsymbol{0},a_1^2{\mathbf{I}}) \approx \mathcal{N}(\boldsymbol{0},a_1^2{\mathbf{I}})$

</td>
<td markdown="block">

$\mathcal{N}(\boldsymbol{0},{\mathbf{I}})$

</td>
<td markdown="block">

$\mathcal{N}(\boldsymbol{0},{\mathbf{I}})$

</td>
<td markdown="block">

$\mathcal{N}(\boldsymbol{0},{\mathbf{I}})$

</td>
</tr>
</table>
</div>

在 Gaussian FM 设定中，光束状与聚光灯状条件路径都导出类似标准扩散损失的训练目标。如[第 6 章](../06-unified-perspective/index.md#ch-unified-perspective)将详细说明的，在采用线性调度 $a_t = 1-t$ 与 $b_t = t$ 时，Gaussian FM 实际上可等价解释为训练模型预测*速度*的扩散模型。这一视角表明，流匹配与扩散是可以相互转换的两种等价形式。Gaussian FM 目标在实践中尤其有吸引力：其损失函数 ${\mathbb{E}}_{t,{\mathbf{x}}_t} \left[\left\lVert{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - ({\mathbf{x}}_1 - {\mathbf{x}}_0)\right\rVert_2^2\right]$ 简单，并且已有研究表明，它在大规模场景中取得了有竞争力的性能（[Esser et al., 2024](../../references.md#ref-esser2024scaling)）。

## 5.3.3 先构造条件流来得到 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 与 $p_t(\cdot|{\mathbf{z}})$ {#subsec-prob-path-flow}

考虑一般情形，端点 $p_{\mathrm{src}}$（位于 $t=0$）与 $p_{\mathrm{tgt}}$（位于 $t=1$）均为任意分布。目标是直接在轨迹空间中设计条件流，将样本从 $p_{\mathrm{src}}$ 输运到 $p_{\mathrm{tgt}}$，并给出可用作回归目标的闭式 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})$。

**动机。** 不必先设计条件密度路径，也可以直接指定条件流映射 ${\boldsymbol{\Psi}}_{0\to t}(\cdot;{\mathbf{z}})$，沿轨迹移动样本。这有两个实际优势：(i) 对沿轨迹的运动求时间导数，就能立即得到训练的回归目标；(ii) 在具有几何结构的空间中（Riemann 流形、Lie 群或受约束子流形），直接由几何构造条件流映射 ${\boldsymbol{\Psi}}_{0\to t}$ 往往很自然，例如使用测地线、指数映射或预度量（[Lipman et al., 2024](../../references.md#ref-lipman2024flow)），从而得到解析的、无须模拟的训练目标速度。

**条件仿射流（与[命题 5.3.1](03-probability-paths-and-velocities.md#closed-form-vf)的联系）。** 固定条件变量 ${\mathbf{z}}\sim\pi$（例如单侧“聚光灯”训练中的 ${\mathbf{z}}={\mathbf{x}}_1\sim p_{\mathrm{tgt}}$），并通过随时间变化的*条件仿射流*推前 ${\mathbf{x}}_0\sim p_{\mathrm{src}}$：

$$
{\boldsymbol{\Psi}}_{0\to t}({\mathbf{x}}_0;{\mathbf{z}}) := \boldsymbol\mu_t({\mathbf{z}}) + {\mathbf{A}}_t({\mathbf{z}}) {\mathbf{x}}_0,\qquad t\in[0,1],
$$

其中 $\boldsymbol\mu_t({\mathbf{z}})\in{\mathbb{R}}^D$，且 ${\mathbf{A}}_t({\mathbf{z}})\in{\mathbb{R}}^{D\times D}$ 对于 $t\in(0,1)$ 可逆。边界 ${\mathbf{A}}_0({\mathbf{z}})={\mathbf{I}},\ \boldsymbol\mu_0({\mathbf{z}})=\mathbf 0$ 恢复 $p_{\mathrm{src}}$，对应时刻 $t=0$。通常将 $t\to1$ 时的边界理解为极限（终端映射可以将质量集中到低维集合或一个点）[^3]。

**诱导的条件路径 $p_t(\cdot|{\mathbf{z}})$。** 这一构造定义

$$
p_t(\cdot|{\mathbf{z}}) = \bigl({\boldsymbol{\Psi}}_{0\to t}( \cdot ;{\mathbf{z}})\bigr)_{\#} p_{\mathrm{src}},
\qquad
p_t(\cdot)  =  \int p_t(\cdot|{\mathbf{z}})\pi({\mathbf{z}})\mathop{}\!\mathrm{d}{\mathbf{z}}.
$$

对于 $\mathcal{L}_{\mathrm{CFM}}$，最终关键在于如何从中采样：先抽取 ${\mathbf{z}} \sim \pi$，再抽取 ${\mathbf{x}}_0 \sim p_{\mathrm{src}}$，最后令

$$
{\mathbf{x}}_t = \boldsymbol\mu_t({\mathbf{z}}) + {\mathbf{A}}_t({\mathbf{z}}){\mathbf{x}}_0.
$$

当 ${\boldsymbol{\Psi}}_{0\to t}$ 关于 ${\mathbf{x}}_0$ 为仿射时，$p_t( \cdot | {\mathbf{z}})$ 为 Gaussian 分布，当且仅当 $p_{\mathrm{src}}$ 为 Gaussian 分布。特别地，对于任意非 Gaussian 的 $p_{\mathrm{src}}$，仿射流通常得到非 Gaussian 的 $p_t(\cdot|{\mathbf{z}})$。

**导出的条件速度 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$。** 条件速度 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 通过关于 $t$ 对条件流映射 ${\boldsymbol{\Psi}}_{0\to t}$ 求导得到。遵循[命题 5.3.1](03-probability-paths-and-velocities.md#closed-form-vf)的推导，考虑由流映射 ${\boldsymbol{\Psi}}_{0\to t}({\mathbf{y}};{\mathbf{z}})$ 定义、初始条件为 ${\mathbf{y}}$ 的*条件* ODE，目标是确定对应的条件速度场 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} {\boldsymbol{\Psi}}_{0\to t}({\mathbf{y}};{\mathbf{z}})
= {\mathbf{v}}_t \Big(\underbrace{{\boldsymbol{\Psi}}_{0\to t}({\mathbf{y}};{\mathbf{z}})}_{{\mathbf{x}}} \Big| {\mathbf{z}}\Big).
$$

由于 ${\boldsymbol{\Psi}}_{0\to t}(\cdot;{\mathbf{z}})$ 对于 $t\in(0,1)$ 可逆，可以将初始点 ${\mathbf{y}}$（对应当前状态 ${\mathbf{x}}={\boldsymbol{\Psi}}_{0\to t}({\mathbf{y}};{\mathbf{z}})$）表示为

$$
{\mathbf{y}}
=
{\boldsymbol{\Psi}}_{0\to t}^{-1}({\mathbf{x}};{\mathbf{z}})
=
{\boldsymbol{\Psi}}_{t\to0}({\mathbf{x}};{\mathbf{z}}).
$$

相应的 Euler 速度场因此为

$$
{\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}})
:=
\left.
\frac{\partial}{\partial t}
{\boldsymbol{\Psi}}_{0\to t}({\mathbf{y}};{\mathbf{z}})
\right|_{{\mathbf{y}}={\boldsymbol{\Psi}}_{0\to t}^{-1}({\mathbf{x}};{\mathbf{z}})}.
$$

也就是说，先对前向流映射关于 $t$ 求导，同时保持初始点 ${\mathbf{y}}$ 固定；再代入那个沿轨迹到达 ${\mathbf{x}}$ 的初始点，到达时刻为 $t$。

由于 ${\mathbf{x}}_t=\boldsymbol\mu_t({\mathbf{z}})+{\mathbf{A}}_t({\mathbf{z}}){\mathbf{x}}_0$ 且 ${\mathbf{A}}_t({\mathbf{z}})$ 在 $(0,1)$ 上可逆，有 ${\mathbf{x}}_0={\mathbf{A}}_t({\mathbf{z}})^{-1}\bigl({\mathbf{x}}-\boldsymbol\mu_t({\mathbf{z}})\bigr)$，得到

$$
{\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}})
 =
\boldsymbol\mu_t'({\mathbf{z}}) + {\mathbf{A}}_t'({\mathbf{z}}) {\mathbf{A}}_t({\mathbf{z}})^{-1} \bigl({\mathbf{x}}-\boldsymbol\mu_t({\mathbf{z}})\bigr).
$$

**单侧条件化（${\mathbf{z}}={\mathbf{x}}_1$）。** 选择 $\boldsymbol\mu_t({\mathbf{z}})=b_t {\mathbf{z}}$ 与 ${\mathbf{A}}_t({\mathbf{z}})=a_t {\mathbf{I}}$，其中 $a_0=1,a_1=0$ 与 $b_0=0,b_1=1$（$a_t>0$ 对于 $t\in(0,1)$ 成立），得到

$$
{\mathbf{x}}_t = a_t {\mathbf{x}}_0 + b_t {\mathbf{x}}_1,
\qquad
{\mathbf{v}}_t({\mathbf{x}}|{\mathbf{x}}_1) = b_t' {\mathbf{x}}_1 + \frac{a_t'}{a_t} \bigl({\mathbf{x}}-b_t{\mathbf{x}}_1\bigr).
$$

对于成对样本 $({\mathbf{x}}_0,{\mathbf{x}}_1)$（其中 ${\mathbf{x}}_t=a_t{\mathbf{x}}_0+b_t{\mathbf{x}}_1$），这简化为通常的 CFM 目标：

$$
{\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{x}}_1)=a_t' {\mathbf{x}}_0 + b_t' {\mathbf{x}}_1.
$$

**双侧条件化（${\mathbf{z}}=({\mathbf{x}}_0,{\mathbf{x}}_1)$）。** 相同模板在 $\boldsymbol\mu_t({\mathbf{x}}_0,{\mathbf{x}}_1)=b_t {\mathbf{x}}_1$ 与 ${\mathbf{A}}_t({\mathbf{x}}_0,{\mathbf{x}}_1)=a_t {\mathbf{I}}$ 下，使条件路径成为*确定性*的：

$$
{\mathbf{x}}_t=a_t {\mathbf{x}}_0+b_t {\mathbf{x}}_1,
\qquad
p_t(\cdot|{\mathbf{x}}_0,{\mathbf{x}}_1)=\delta\left(\cdot - (a_t{\mathbf{x}}_0+b_t{\mathbf{x}}_1)\right),
$$

而条件速度为

$$
{\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{x}}_0,{\mathbf{x}}_1)=a_t' {\mathbf{x}}_0+b_t' {\mathbf{x}}_1,
$$

即标准的双侧 CFM 目标。

**无条件 Gaussian 路径作为特例。** 若 $\boldsymbol{\mu}_t$ 与 ${\mathbf{z}}$ 无关（记为 $\boldsymbol{\mu}(t)$），且 ${\mathbf{A}}_t = \tfrac{\sigma(t)}{\sigma(0)} {\mathbf{I}}$，则

$$
{\boldsymbol{\Psi}}_{0\to t}({\mathbf{x}}_0)=\boldsymbol\mu(t)+\sigma(t) \frac{{\mathbf{x}}_0-\boldsymbol\mu(0)}{\sigma(0)},
\quad
{\mathbf{v}}_t({\mathbf{x}})=\boldsymbol\mu'(t)+\frac{\sigma'(t)}{\sigma(t)}\bigl({\mathbf{x}}-\boldsymbol\mu(t)\bigr),
$$

这就恢复了[命题 5.3.1](03-probability-paths-and-velocities.md#closed-form-vf)中的 Gaussian 密度路径与闭式速度。

## 5.3.4 先构造概率路径与先构造流的比较 {#probability-path-first-vs-flow-first}

这两种构造对应输运的互补 Euler 描述与 Lagrange 描述。*先构造概率路径*（Euler）的视角，首先指定条件密度路径 $p_t(\cdot|{\mathbf{z}})$，再通过连续性方程选择相容的速度场。由于单独的密度路径通常不能唯一确定速度，可能需要额外的输运结构来作出选择。

*先构造流*（Lagrange）的视角则指定条件流映射 ${\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}})$，直接描述粒子轨迹。相应的密度路径由推前得到；当映射可逆时，其 Euler 速度可唯一恢复为

$$
{\mathbf{v}}_t
=
\partial_t{\boldsymbol{\Psi}}_{0\to t}
\circ
{\boldsymbol{\Psi}}_{0\to t}^{-1}.
$$

因此，两种构造从不同起点描述同类输运对象，但指定的密度路径不一定会选择与另行指定的流映射相同的速度。

下表总结这些区别。当条件路径具有闭式速度时，先构造路径很自然；当对轨迹具有强结构先验时，先构造流很自然。

<span id="tb-conditional-path-vs-flow"></span>

**表 5.3：** 先构造条件概率路径与先构造条件流的比较。

<div style="overflow-x: auto;" markdown="1">
<table markdown="1">
<tr markdown="1">
<th markdown="block">

**比较项**

</th>
<th markdown="block">

**先构造条件概率路径**

</th>
<th markdown="block">

**先构造条件流**

</th>
</tr>
<tr markdown="1">
<td markdown="block">

**给定对象**

</td>
<td markdown="block">

条件密度路径 $p_t(\cdot|{\mathbf{z}})$。

</td>
<td markdown="block">

条件流映射 ${\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}})$（对每个固定 ${\mathbf{z}}$ 给出轨迹）。

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**获得速度**

</td>
<td markdown="block">

对每个 ${\mathbf{z}}$，寻找 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$，使

$$
\partial_t p_t(\cdot|{\mathbf{z}})
+
\nabla \cdot
\left(
p_t(\cdot|{\mathbf{z}}){\mathbf{v}}_t(\cdot|{\mathbf{z}})
\right)
=
0;
$$

非唯一：若 $\nabla\!\cdot~\left(p_t \mathbf w_t\right)=0$，则 ${\mathbf{v}}_t+\mathbf w_t$ 产生相同的 $p_t$。

</td>
<td markdown="block">

沿路径（对每个 ${\mathbf{z}}$）：

$$
{\mathbf{v}}_t
\left(
{\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}})
\mid
{\mathbf{z}}
\right)
=
\tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}
{\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}}).
$$

当 ${\boldsymbol{\Psi}}_{0\to t}$ 可逆时，可以求解

$$
{\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}})
=
\left.
\partial_t
{\boldsymbol{\Psi}}_{0\to t}({\mathbf{y}}|{\mathbf{z}})
\right|_{{\mathbf{y}}={\boldsymbol{\Psi}}_{0\to t}^{-1}({\mathbf{x}}|{\mathbf{z}})}.
$$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 的闭式形式**

</td>
<td markdown="block">

当 $p_t(\cdot|{\mathbf{z}})$ 为 Gaussian／指数族分布时较方便；否则获得 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 并不容易。

</td>
<td markdown="block">

当 ${\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}})$ 具有结构（仿射／低秩）时较方便；无须计算密度。

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 的唯一性**

</td>
<td markdown="block">

对于每个 ${\mathbf{z}}$，除非施加选择规则（例如势流／最小动能），否则 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 不完全确定。

</td>
<td markdown="block">

给定 ${\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}})$ 后，$p_t(\cdot|{\mathbf{z}})=({\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}}))_\#p_0$ 与粒子轨迹都被确定。若流映射可逆，对应的 Euler 速度场唯一确定；否则，未必存在唯一的全局 Euler 场。

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**可实现性**

</td>
<td markdown="block">

必须验证构造的 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 在预期支撑集上满足连续性方程。

</td>
<td markdown="block">

由构造自动成立：

$$
p_t(\cdot|{\mathbf{z}})
=
\left(
{\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}})
\right)_\#
p_0(\cdot|{\mathbf{z}}).
$$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**匹配 $(p_{\mathrm{src}},p_{\mathrm{tgt}})$**

</td>
<td markdown="block">

混合条件分布：

$p_{\mathrm{src}} = \int p_0(\cdot|{\mathbf{z}})\pi({\mathbf{z}})\mathop{}\!\mathrm{d}{\mathbf{z}},$

$p_{\mathrm{tgt}} = \int p_1(\cdot|{\mathbf{z}})\pi({\mathbf{z}})\mathop{}\!\mathrm{d}{\mathbf{z}}.$

对于 Gaussian 仿射条件路径，若系数与 ${\mathbf{z}}$ 无关，则 $p_{\mathrm{src}}$ 可被强制为 Gaussian。对于一般固定端点 $p_{\mathrm{src}}$（可能非 Gaussian），$p_t(\cdot|{\mathbf{z}})$ 的选择通常不能固定 $p_{\mathrm{src}}$。

</td>
<td markdown="block">

设 ${\boldsymbol{\Psi}}_{0\to 0}=\mathrm{Id}$，并选择边界条件以到达任意 $p_{\mathrm{tgt}}$。

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**适用场景**

</td>
<td markdown="block">

类似扩散的构造；通过条件 Gaussian 分布 $p_t(\cdot|{\mathbf{z}})$ 得到解析目标。

</td>
<td markdown="block">

通过映射 ${\boldsymbol{\Psi}}_{0\to t}(\cdot|{\mathbf{z}})$ 施加强结构先验；易于控制边界；容纳奇异／低维端点；适合基于映射的正则化／输运代价。

</td>
</tr>
</table>
</div>

## 5.3.5 条件路径为直线不意味着边缘流轨迹为直线 {#conditional-vs-marginal-straightness}

标准仿射插值

$$
{\mathbf{x}}_t
=
(1-t){\mathbf{x}}_0+t{\mathbf{x}}_1
$$

具有一种很有吸引力的几何性质：给定特定端点对 $({\mathbf{x}}_0,{\mathbf{x}}_1)$ 后，插值是一条具有恒定速度 ${\mathbf{x}}_1-{\mathbf{x}}_0$ 的直线。这种条件层面的直线性广泛用于流匹配与 Rectified Flow。

不过，必须区分这些*条件插值路径*的直线性，与生成时所用*边缘 ODE 轨迹*的直线性。在耦合 $\pi({\mathbf{x}}_0,{\mathbf{x}}_1)$ 下，总体最优边缘速度为

$$
{\mathbf{v}}^\star({\mathbf{x}},t)
=
\mathbb{E}
\left[
{\mathbf{x}}_1-{\mathbf{x}}_0
\,\middle|\,
{\mathbf{x}}_t={\mathbf{x}}
\right].
$$

端点对的后验分布通常随 $({\mathbf{x}},t)$ 变化。因此，即使训练使用的每条条件插值都是直线，沿边缘 ODE 轨迹遇到的速度也不必保持恒定。

所以，仅选择标准仿射调度，并不意味着学得的边缘 ODE 严格为直线，也不意味着一步 Euler 更新就能精确求解。Rectified Flow 正是针对这一区别：其矫正或 `Reflow` 过程改变端点耦合，使诱导的因果流逐渐接近直线流（[Liu et al., 2022](../../references.md#ref-liu2022flow)）。

以下示例明确展示这一区别。

**示例：一个简单的一维反例**

考虑以下一维设定：

$$
x_0 \sim \mathcal{N}(0, \sigma_0^2),
\qquad
x_1 \sim \mathcal{N}(0, \sigma_1^2),
$$

采用独立耦合，以及标准仿射插值 $x_t = (1-t)x_0 + tx_1$。每条条件路径 $(1-t)x_0 + tx_1$ 都是关于时间的直线。下面计算诱导的边缘速度场。

由于 $(x_1 - x_0,\, x_t)$ 联合服从 Gaussian 分布，条件期望关于 $x$ 是线性的，因此

$$
v^*(x,t)
= {\mathbb{E}}[x_1 - x_0| x_t = x]
= \frac{\operatorname{Cov}(x_1 - x_0,\, x_t)}
       {\operatorname{Var}(x_t)}\, x.
$$

直接计算可得

$$
\operatorname{Var}(x_t)
= (1-t)^2\sigma_0^2 + t^2\sigma_1^2
=: V(t),
$$

以及

$$
\operatorname{Cov}(x_1 - x_0,\, x_t)
= t\sigma_1^2 - (1-t)\sigma_0^2
= \tfrac{1}{2} V'(t).
$$

从而

$$
v^*(x,t) = \frac{V'(t)}{2V(t)}\, x.
$$

因此，边缘 ODE

$$
\frac{\mathop{}\!\mathrm{d} z_t}{\mathop{}\!\mathrm{d} t} = v^*(z_t, t)
$$

具有精确解

$$
z_t
= z_0 \sqrt{\frac{V(t)}{V(0)}}
= z_0 \sqrt{(1-t)^2 + \frac{\sigma_1^2}{\sigma_0^2}\, t^2}.
$$

现在，关于时间的直线插值要求 $z_t$ 关于 $t$ 为仿射，即 $z_t = (1-t)z_0 + t\,z_1$，其中 $z_1$ 为某个端点。然而，

$$
\sqrt{(1-t)^2 + \frac{\sigma_1^2}{\sigma_0^2}\, t^2}
$$

不是 $t$ 的仿射函数，对任意正的 $\sigma_0, \sigma_1$ 都如此。因此，ODE 轨迹通常并非关于时间的直线插值。[图 5.10](03-probability-paths-and-velocities.md#fig-straightness-counterexample)使用 $\sigma_0=2$ 与 $\sigma_1=3$ 的示例展示这一现象。

例如，当 $\sigma_0 = \sigma_1$ 时，有 $z_1 = z_0$，因此直线插值只能是常值路径 $z_t = z_0$。而实际 ODE 轨迹满足

$$
z_{1/2} = \frac{z_0}{\sqrt{2}} \neq z_0.
$$

因此，尽管每条条件插值路径都是直线，诱导的边缘 ODE 轨迹却未必如此。

<span id="fig-straightness-counterexample"></span>

![仿射插值下的边缘 ODE 轨迹未必是直线](../../../en/assets/chapter-05/straightness-counterexample.png)

**图 5.10：** **标准仿射插值下，边缘 ODE 轨迹并非直线。** 取 $x_0 \sim \mathcal{N}(0, \sigma_0^2)$ 与 $x_1 \sim \mathcal{N}(0, \sigma_1^2)$，其中 $\sigma_0=2$、$\sigma_1=3$，并采用独立耦合。灰线展示采样得到的条件路径 $(1-t)x_0 + tx_1$，它们按构造就是直线。实线曲线展示精确的边缘 ODE 轨迹 $z_t = z_0\sqrt{(1-t)^2 + (\sigma_1/\sigma_0)^2 t^2}$，虚线展示从 $z_0$ 到 $z_1$ 的对应直线插值。实线与虚线之间可见的差距证实，边缘 ODE 轨迹关于 $t$ 并非仿射，尽管每条条件路径都是直线。边缘密度 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 分别展示在左右边缘。

来源：作者借助 AI 辅助编程绘制。

[^1]: 在（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)）中，作者考虑 $\boldsymbol{\Psi}_{0\to t}({\mathbf{x}}) = \boldsymbol{\mu}(t) + \sigma(t){\mathbf{x}}$，这需要对 $\boldsymbol{\mu}(t)$ 与 $\sigma(t)$ 施加约束来保证边界条件。我们采用等价的归一化形式，避免这些约束。

[^2]: 也可以添加固定的条件方差 $\sigma^2{\mathbf{I}}$。由于方差不随时间变化，条件速度仍为 $a_t'{\mathbf{x}}_0+b_t'{\mathbf{x}}_1$，但端点边缘分布会成为 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 的 Gaussian 平滑版本，在 $\sigma\to0$ 时趋近所需端点。

[^3]: 允许 ${\mathbf{A}}_1({\mathbf{z}})$ 为奇异（例如 $\mathbf 0$），与 $(0,1)$ 上的可逆性相容，并使路径在 $t=1$ 收缩到指定端点。
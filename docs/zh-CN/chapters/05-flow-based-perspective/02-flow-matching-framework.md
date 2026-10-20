# 5.2 流匹配框架 {#sec-flow-matching-framework}

Score SDE（[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)）与 NODE（[第 5.1 节](01-normalizing-flows-and-neural-odes.md#sec-flow-based-method)）都将生成建模描述为连续输运过程。Score SDE 通过随机动力学及相应的 PF-ODE 表述输运，NODE 则直接参数化确定性 ODE。在标准生成设定下，二者都旨在将简单 Gaussian 先验 $p_{\mathrm{prior}}$ 输运为数据分布 $p_{\mathrm{data}}$。

NODE 的实际限制是，基于似然的训练需要反复求解 ODE。*流匹配*（Flow Matching，FM）框架（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)；[Lipman et al., 2024](../../references.md#ref-lipman2024flow)；[Tong et al., 2024](../../references.md#ref-tongimproving)）通过回归目标训练神经网络直接预测速度场，避免了这一成本。因此，生成时需要 ODE 积分，训练循环内部则不需要，使 FM 训练无须模拟。

更一般地，FM 并不限于从 Gaussian 分布到数据的输运。它考虑源分布 $p_{\mathrm{src}}$ 与目标分布 $p_{\mathrm{tgt}}$，假设二者都易于采样。标准生成建模是其特殊情形：

$$
p_{\mathrm{src}}=p_{\mathrm{prior}},
\qquad
p_{\mathrm{tgt}}=p_{\mathrm{data}},
$$

其中 $p_{\mathrm{prior}}$ 通常为 Gaussian 分布。

本节采用 FM 视角[^1]。目标是学习随时间变化的速度场 $\mathbf{v}_t({\mathbf{x}}_t)$，使对应 ODE 沿指定概率路径 $\{p_t\}_{t\in[0,1]}$ 输运样本，满足

$$
p_0=p_{\mathrm{src}},
\qquad
p_1=p_{\mathrm{tgt}}.
$$

当 $p_{\mathrm{src}}$ 为 Gaussian 分布时，我们称这一设定为 *Gaussian 流匹配*。因此，FM 既为生成建模提供无须模拟的训练方法，也为学习任意端点分布之间的连续输运提供一般框架。

## 5.2.1 从基于 score 的方法得到的启示 {#subsec-lesson-score}

我们用略有不同但等价的形式重新考察 Score SDE 框架（[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)），提炼出启发 FM 方法的关键认识。这一分析揭示扩散模型如何隐式学习概率流，并引出更直接的形式。

**步骤 1：定义条件路径及其边缘密度。**

<span id="fig-data-prior-conditioning"></span>

![条件转移分布连接数据与 Gaussian 先验](../../../en/assets/chapter-05/data-prior-conditioning.png)

**图 5.6：** **条件转移分布示意图。** $p_t({\mathbf{x}}_t | {\mathbf{x}}_0) = \mathcal{N}({\mathbf{x}}_t;\alpha_t {\mathbf{x}}_0,\sigma_t^2 \mathbf{I})$ 定义一条 Gaussian 条件概率路径，从数据样本 ${\mathbf{x}}_0 \sim p_{\text{data}}$（左）走向 Gaussian 先验 $p_{\text{prior}}$（右）。

来源：作者绘制。

扩散模型指定连续时间密度族 $\{p_t\}_{t \in [0,1]}$，将简单先验 $p_{\text{prior}}$（例如 Gaussian）作为源分布，从 $t=1$ 输运到目标数据分布 $p_{\text{data}}$，对应时刻 $t=0$：

$$
p_1({\mathbf{x}}_1) = p_{\text{prior}}({\mathbf{x}}_1), \quad p_0({\mathbf{x}}_0) = p_{\text{data}}({\mathbf{x}}_0).
$$

这条路径通过前向条件分布隐式定义：

<span id="eq-lesson-dm-conditional-path"></span>

$$
\begin{aligned}
p_t({\mathbf{x}}_t|{\mathbf{x}}_0) = \mathcal{N}({\mathbf{x}}_t; \alpha_t {\mathbf{x}}_0, \sigma_t^2 \mathbf{I}), \quad {\mathbf{x}}_0\sim p_{\text{data}}
\end{aligned}
\tag{5.2.1}
$$

它诱导出边缘密度

$$
p_t({\mathbf{x}}_t) := \int p_t({\mathbf{x}}_t |{\mathbf{x}}_0)  p_{\text{data}}({\mathbf{x}}_0) \mathrm{d}{\mathbf{x}}_0.
$$

条件 Gaussian 分布中递增的方差 $\sigma_t^2$，推动 $p_t$ 向 Gaussian 先验演化。

**步骤 2：速度场。** 边缘密度 $p_t$ 随时间的演化由速度场 ${\mathbf{v}}_t : \mathbb{R}^D \to \mathbb{R}^D$ 支配，该速度场从 Fokker–Planck 方程导出：

<span id="eq-dm-velocity"></span>

$$
\begin{aligned}
{\mathbf{v}}_t({\mathbf{x}}) := f(t){\mathbf{x}} - \frac{1}{2}g^2(t)\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}),
\end{aligned}
\tag{5.2.2}
$$

它通过 PF-ODE 定义确定性的粒子流：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = \underbrace{f(t){\mathbf{x}}(t) - \frac{1}{2}g^2(t)\nabla_{{\mathbf{x}}} \log p_t\big({\mathbf{x}}(t)\big)}_{{\mathbf{v}}_t({\mathbf{x}}(t))}.
$$

这个 ODE 沿时间正向输运初始随机变量 ${\mathbf{x}}(0) \sim p_{\text{data}}$，或沿时间反向输运 ${\mathbf{x}}(1) \sim p_{\text{prior}}$，使 ${\mathbf{x}}(t)$ 的演化边缘密度匹配 $p_t$，对每个 $t \in [0,1]$ 都成立（见下文“底层规则”）。

标量函数 $f(t)$ 与 $g(t)$ 由相应前向 SDE 的系数确定，或等价地，由条件路径中定义的 Gaussian 核参数 $\alpha_t$ 与 $\sigma_t$ 确定（见[引理 4.5.1](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#forward-sde)）。

**步骤 3：通过条件化策略学习。** [式 (5.2.2)](02-flow-matching-framework.md#eq-dm-velocity)中的 PF-ODE 速度依赖通常未知的边缘 score $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)$。因此，基于 score 的方法使用神经网络 ${\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)$，通过理想的 score 匹配目标学习这个 score：

$$
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi})
=
\mathbb{E}_{t\sim\mathcal{U}[0,1],\,{\mathbf{x}}_t\sim p_t}
\left[
\left\|
{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
-
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)
\right\|^2
\right].
$$

学得 score 后，由它得到 PF-ODE 速度：

$$
{\mathbf{v}}_t({\mathbf{x}}_t)
=
f(t){\mathbf{x}}_t
-
\frac{1}{2}g^2(t)
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t),
$$

其中 ${\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t) \approx \nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}})$。

困难在于，计算回归目标时无法获得边缘 score 本身。与[第 3 章](../03-score-based-perspective/index.md#ch-score-based)一样，我们用条件化技巧解决这一问题。条件分布 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 可计算，且其条件 score 满足

<span id="eq-marginal-oracle-score"></span>

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)
=
\mathbb{E}_{{\mathbf{x}}_0\sim p(\cdot|{\mathbf{x}}_t)}
\left[
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\right].
\end{aligned}
\tag{5.2.3}
$$

由此得到去噪 score 匹配目标

$$
\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi})
:=
\mathbb{E}_{t,\,{\mathbf{x}}_0\sim p_{\mathrm{data}},\,
{\mathbf{x}}_t\sim p_t(\cdot|{\mathbf{x}}_0)}
\left[
\left\|
{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
-
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\right\|^2
\right].
$$

如[定理 3.3.1](../03-score-based-perspective/03-denoising-score-matching.md#thm-sm-dsm)所示，$\mathcal{L}_{\text{SM}}$ 与 $\mathcal{L}_{\text{DSM}}$ 只相差一个与 $\boldsymbol{\phi}$ 无关的常数，因此具有相同的总体最优解：

$$
{\mathbf{s}}^*({\mathbf{x}}_t,t)
=
\mathbb{E}_{{\mathbf{x}}_0\sim p(\cdot|{\mathbf{x}}_t)}
\left[
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\right]
=
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t).
$$

同一条件化原理也可以直接在 PF-ODE 速度层面表述。定义条件速度

$$
{\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{x}}_0)
:=
f(t){\mathbf{x}}_t
-
\frac{1}{2}g^2(t)
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0).
$$

利用[式 (5.2.3)](02-flow-matching-framework.md#eq-marginal-oracle-score)，立即得到

$$
{\mathbf{v}}_t({\mathbf{x}}_t)
=
\mathbb{E}_{{\mathbf{x}}_0\sim p(\cdot|{\mathbf{x}}_t)}
\left[
{\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\right].
$$

因此，基于 score 的扩散通过可计算的条件目标学习 score，再用它构造 PF-ODE 速度。流匹配将直接对速度场应用同样的从条件到边缘的原理。

<span id="fig-conditional-marginal"></span>

![扩散中的条件视角与边缘视角](../../../en/assets/chapter-05/conditional-marginal.png)

**图 5.7：** **扩散中的条件与边缘视角示意图。** （本图受 [Lipman et al. (2024)](../../references.md#ref-lipman2024flow) 启发。）（1）条件 Gaussian 路径 $p_t(\cdot|\mathbf{x}_0)$，展示密度从固定 $\mathbf{x}_0$ 向先验扩张。（2）条件速度

$$
\mathbf{v}_t(\mathbf{x}_t|\mathbf{x}_0)=f(t){\mathbf{x}}_t - \tfrac{1}{2}g^2(t) \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t|{\mathbf{x}}_0).
$$

 （3）边缘密度 $p_t$，将数据分布 $p_{\mathrm{data}}$（橙色）输运到先验 $p_{\mathrm{prior}}$（灰色）。（4）边缘速度 $\mathbf{v}_t(\mathbf{x}_t)=f(t){\mathbf{x}}_t - \tfrac{1}{2}g^2(t) \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)$，对从 $\mathbf{x}_t$ 指向多个可能起点的条件方向（虚线）取平均，得到红色箭头。在采用单侧条件化 ${\mathbf{z}}={\mathbf{x}}_0$ 的 FM 框架中，同样的示意适用于 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 与 ${\mathbf{v}}_t({\mathbf{x}}_t)$，无须将它们显式写成 score $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 或 $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)$ 的形式。

来源：作者绘制。

**底层规则：Fokker–Planck 方程。** 边缘密度 $p_t$ 按 Fokker–Planck 方程演化：

$$
\begin{aligned}
\frac{\partial p_t({\mathbf{x}})}{\partial t} + \nabla \cdot \Bigg( \underbrace{\left(f(t){\mathbf{x}} - \frac{1}{2}g^2(t) \nabla_{\mathbf{x}} \log p_t({\mathbf{x}})\right)}_{{\mathbf{v}}_t({\mathbf{x}})}  p_t({\mathbf{x}}) \Bigg) = 0.
\end{aligned}
$$

这个 PDE 保证 PF-ODE 给出的密度匹配前向 SDE 的边缘分布。为说明这一点，回顾[式 (4.2.2)](../04-score-sde-framework/02-matching-marginals.md#eq-flow-map-def)定义的 PF-ODE *流映射* ${\boldsymbol{\Psi}}_{s \to t}({\mathbf{x}}_s)$，它将初始状态 ${\mathbf{x}}_s$ 从时刻 $s$ 直接输运到时刻 $t$ 的状态。从 $t=1$ 到 $t=0$ 反向运行 PF-ODE，以 ${\mathbf{x}}_1 \sim p_{\mathrm{prior}}$ 为初值，通过推前公式得到随时间变化的密度：

<span id="eq-pushforward-ode"></span>

$$
\begin{aligned}
p_t^{\mathrm{rev}}({\mathbf{x}})
    = \int \delta \left({\mathbf{x}} - {\boldsymbol{\Psi}}_{1 \to t}({\mathbf{x}}_1)\right)
      p_{\mathrm{prior}}({\mathbf{x}}_1) \mathrm{d}{\mathbf{x}}_1.
\end{aligned}
\tag{5.2.4}
$$

Fokker–Planck 方程保证所诱导的密度路径与同一个演化密度一致：

<span id="eq-p-t-fwd-rev"></span>

$$
\begin{aligned}
p_t^{\mathrm{rev}} = p_t.
\end{aligned}
\tag{5.2.5}
$$

特别地，这意味着 $p_0^{\mathrm{rev}} = p_0 = p_{\text{data}}$，从而在时刻 $t=0$ 恢复数据分布。由于 ODE 解映射是双向的，也可以类似地考虑在 ${\mathbf{x}}_0 \sim p_{\text{data}}$ 初始化，并沿时间正向求解 ODE，作相应分析。

## 5.2.2 流匹配框架 {#subsec-fm-framework}

[第 5.2.1 节](02-flow-matching-framework.md#subsec-lesson-score)的分析表明，扩散模型通过学习速度场，具体来说是 score，在满足边界条件的同时实现分布之间的输运。[式 (5.2.1)](02-flow-matching-framework.md#eq-lesson-dm-conditional-path)中方差 $\sigma_t^2$ 递增的 Gaussian 条件路径设计，隐式地将一个端点固定到 Gaussian 先验，同时允许条件密度定义在整个空间上，从而进行基于 score 的梯度计算。

本小节介绍 FM 框架，它建立在上述认识之上（[图 5.7](02-flow-matching-framework.md#fig-conditional-marginal)的同一示意也适用于 FM 框架），并将其扩展为学习连续流，在任意两个分布 $p_{\text{src}}$ 与 $p_{\text{tgt}}$ 之间输运样本。

**步骤 1：定义条件路径及其边缘密度。** 考虑任意源概率分布 $p_{\mathrm{src}}$ 与目标概率分布 $p_{\mathrm{tgt}}$，它们定义在 $\mathbb{R}^D$ 上。设[^2]

<span id="eq-marginal-endpoint"></span>

$$
\begin{aligned}
p_0(\mathbf{x}) = p_{\mathrm{src}}(\mathbf{x}), \quad p_1(\mathbf{x}) = p_{\mathrm{tgt}}(\mathbf{x}).
\end{aligned}
\tag{5.2.6}
$$

FM 隐式定义一族在两个端点之间插值的连续中间密度 $\{ p_t \}_{t \in [0,1]}$。每个边缘密度 $p_t$ 都通过潜变量 $\mathbf{z}$ 表示，该变量从已知分布 $\pi(\mathbf{z})$ 抽取，并配以条件分布 $p_t(\mathbf{x}_t | \mathbf{z})$：

<span id="eq-p-t"></span>

$$
p_t(\mathbf{x}_t) = \int p_t(\mathbf{x}_t | \mathbf{z}) \pi(\mathbf{z}) \mathop{}\!\mathrm{d} \mathbf{z},
\tag{5.2.7}
$$

其中选择 $(\pi(\mathbf{z}), \{p_t(\cdot|\mathbf{z})\})$，使其满足[式 (5.2.6)](02-flow-matching-framework.md#eq-marginal-endpoint)的边界条件。

一般而言，边缘密度 $p_t$ 不可计算，因为需要对 $\pi(\mathbf{z})$ 积分，而 $\pi(\mathbf{z})$ 与条件分布 $p_t(\mathbf{x}_t|\mathbf{z})$ 都可能很复杂。不过，以潜变量 $\mathbf{z}$ 为条件，使 FM 可以灵活建模比[第 5.2.1 节](02-flow-matching-framework.md#subsec-lesson-score)更广泛的插值路径。$\mathbf{z}$ 的常用选择包括：

-   **双侧条件化：** $\mathbf{z} = (\mathbf{x}_0, \mathbf{x}_1) \sim p_{\text{src}}(\mathbf{x}_0) p_{\text{tgt}}(\mathbf{x}_1)$，其中 $\pi$ 耦合源分布与目标分布。这使 FM 能在任意分布之间定义输运。

-   **单侧条件化：** $\mathbf{z} = \mathbf{x}_0$ 或 $\mathbf{z} = \mathbf{x}_1$。尤其在选择 Gaussian 源分布时，它恢复类似扩散的设定。

所有情况下，条件路径 $p_t(\mathbf{x}_t|\mathbf{z})$ 都应能够高效采样，相应的条件速度也应可计算。在下文先构造概率路径的方法中，我们还使用具有可计算闭式表达式的条件分布。具体构造见[第 5.3.2 节](03-probability-paths-and-velocities.md#subsec-fm-instantiations)，示意图见[图 5.9](03-probability-paths-and-velocities.md#fig-conditional-path)。

**步骤 2：速度场。** 概率路径 $\{p_t\}_{t\in[0,1]}$ 指定分布应如何随时间演化，但通常并不指定单个样本应如何运动。换言之，同一个边缘密度序列可以由不同速度场实现[^3]。

因此，FM 的目标是寻找速度场 $\mathbf{v}_t(\mathbf{x})$，使其诱导的 ODE

$$
\frac{\mathop{}\!\mathrm{d} \mathbf{x}(t)}{\mathop{}\!\mathrm{d} t}
=
\mathbf{v}_t(\mathbf{x}(t)),
\qquad
t\in[0,1],
$$

产生的边缘分布匹配指定的 $p_t$，覆盖每个时刻 $t$。这一条件成立时，ODE 可以将样本从 $\mathbf{x}(0)\sim p_{\text{src}}$ 正向输运到 $p_{\text{tgt}}$，或从 $\mathbf{x}(1)\sim p_{\text{tgt}}$ 反向输运到 $p_{\text{src}}$（更形式化的讨论见[第 5.2.4 节](02-flow-matching-framework.md#subsec-fm-theory)）。

这一要求由*连续性方程*[^4]刻画：

<span id="eq-continuity-eq"></span>

$$
\frac{\partial p_t(\mathbf{x})}{\partial t}
    +
    \nabla\cdot
    \bigl(
    \mathbf{v}_t(\mathbf{x})p_t(\mathbf{x})
    \bigr)
    =
    0.
\tag{5.2.8}
$$

任意满足[式 (5.2.8)](02-flow-matching-framework.md#eq-continuity-eq)的速度场 $\mathbf{v}_t$，结合初始条件 $p_0=p_{\mathrm{src}}$，在适当正则性条件下，都会生成一个 ODE，其演化边缘分布遵循指定路径 $\{p_t\}$。因此，求解 ODE 就在 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 之间提供逐样本输运，同时匹配中间分布。

连续性方程约束概率密度的演化，但通常不能唯一确定速度场。因此，不同样本轨迹可以产生完全相同的边缘密度演化。例如，在 $p_t>0$ 的区域上，若 $\mathbf{v}_t$ 满足[式 (5.2.8)](02-flow-matching-framework.md#eq-continuity-eq)，则

$$
\mathbf{v}_t
+
\frac{1}{p_t}\tilde{\mathbf{v}}_t
$$

对任意足够正则的无散度场 $\tilde{\mathbf{v}}_t$，其中 $\nabla\cdot\tilde{\mathbf{v}}_t=0$，也满足该方程。

因此，FM 寻找的是实现指定概率路径的*某一个*速度场。对于一般源分布与目标分布，无论边缘密度 $p_t$ 还是这样的速度场，通常都没有闭式形式。这引出下面的条件化构造：用可计算的条件路径与速度学习有效的边缘速度场。

作为具体示例，[第 5.3.1 节](03-probability-paths-and-velocities.md#subsec-special-fm-instantiations)考虑 Gaussian 到 Gaussian 的路径，以及一个特定的仿射流映射，其对应速度场可以显式计算。

**步骤 3：通过条件化策略学习。** 一旦选定概率路径及相应的边缘速度场 ${\mathbf{v}}_t$，FM 就使用神经网络 ${\mathbf{v}}_{\boldsymbol{\phi}}$ 直接学习这一速度场。理想的 FM 目标为

$$
\mathcal{L}_{\mathrm{FM}}(\boldsymbol{\phi})
=
\mathbb{E}_{t,\,{\mathbf{x}}_t\sim p_t}
\left[
\left\|
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
-
{\mathbf{v}}_t({\mathbf{x}}_t)
\right\|^2
\right].
$$

我们将这一神经网络参数化称为 *${\mathbf{v}}$ 预测*（速度预测），因为网络直接预测 ODE 的漂移。这与[第 5.2.1 节](02-flow-matching-framework.md#subsec-lesson-score)基于 score 的构造不同，后者先学习 score，再用它构造 PF-ODE 速度。

困难在于，边缘速度 ${\mathbf{v}}_t({\mathbf{x}}_t)$ 通常不可计算。与去噪 score 匹配一样，使用条件化策略可以解决这一问题。我们引入潜变量 ${\mathbf{z}}\sim\pi({\mathbf{z}})$，并构造可计算的条件概率路径 $p_t({\mathbf{x}}_t|{\mathbf{z}})$ 及相应的条件速度场 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})$。

若每个条件速度场都生成相应的条件概率路径，则对 ${\mathbf{z}}$ 边缘化可得到有效的边缘速度场：

<span id="eq-marginal-oracle-velocity"></span>

$$
\begin{aligned}
{\mathbf{v}}_t({\mathbf{x}}_t)
:=
\mathbb{E}_{{\mathbf{z}}\sim p(\cdot|{\mathbf{x}}_t)}
\left[
{\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})
\right].
\end{aligned}
\tag{5.2.9}
$$

因此，通常不可计算的边缘速度，是可计算条件速度的条件平均。这与[式 (5.2.3)](02-flow-matching-framework.md#eq-marginal-oracle-score)中的条件 score 恒等式直接对应。

这一观察导出*条件流匹配*（conditional flow matching，CFM）目标：

<span id="eq-fm-cfm"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CFM}}(\boldsymbol{\phi})
:=
\mathbb{E}_{t,\,{\mathbf{z}}\sim\pi({\mathbf{z}}),\,
{\mathbf{x}}_t\sim{\color{orange}p_t(\cdot|{\mathbf{z}})}}
\left[
\left\|
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
-
{\color{orange}{\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})}
\right\|^2
\right].
\end{aligned}
\tag{5.2.10}
$$

这里，${\mathbf{z}}$ 用于构造训练样本 ${\mathbf{x}}_t$ 与条件回归目标 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})$；它并不是神经网络 ${\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)$ 的额外输入。两个目标只相差一个常数 $C\leq0$，它与 $\boldsymbol{\phi}$ 无关：

$$
\mathcal{L}_{\mathrm{FM}}(\boldsymbol{\phi})
=
\mathcal{L}_{\mathrm{CFM}}(\boldsymbol{\phi})
+
C.
$$

因此，它们具有相同的总体最优解：

<span id="eq-minimum-velocity"></span>

$$
\begin{aligned}
{\mathbf{v}}^*({\mathbf{x}}_t,t)
=
{\mathbf{v}}_t({\mathbf{x}}_t).
\end{aligned}
\tag{5.2.11}
$$

换言之，尽管每个训练样例都以条件速度 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})$ 为目标，但在这些样例上取平均，就恢复了所需的边缘速度场。

为了使 CFM 训练可计算且无须模拟，需要满足两个实际条件：

-   能够轻易采样 ${\mathbf{x}}_t\sim p_t(\cdot|{\mathbf{z}})$，无须求解 ODE；

-   条件速度 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})$ 具有可计算表达式，可以用作回归目标。

[第 5.3.2 节](03-probability-paths-and-velocities.md#subsec-fm-instantiations)给出满足这些条件的具体构造。因此，正如去噪 score 匹配通过可计算的条件 score 学习不可计算的边缘 score，流匹配通过可计算的条件速度目标学习不可计算的边缘速度。

<span id="fig-cfm-illustration"></span>

![流匹配中的条件化技巧](../../../en/assets/chapter-05/cfm-illustration.png)

**图 5.8：** **流匹配中的条件化技巧。** 这里，${\mathbf{x}}_1\sim p_{\mathrm{tgt}}$ 表示目标数据样本。**（左）** 对固定中间样本 ${\mathbf{x}}_t$，边缘速度 ${\mathbf{v}}_t({\mathbf{x}}_t)$ 对所有可能目标样本 ${\mathbf{x}}_1$ 对应的条件速度取平均，这些目标样本都可能产生 ${\mathbf{x}}_t$。每个 ${\mathbf{x}}_1$ 的贡献按其与观测 ${\mathbf{x}}_t$ 的相容程度加权。直接计算这一平均，需要对整个目标分布积分，通常不可计算。**（右）** 然而，训练时特定目标样本 ${\mathbf{x}}_1$ 是已知的，它用于构造 ${\mathbf{x}}_t$。以该样本为条件，得到闭式速度 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{x}}_1) = b_t'{\mathbf{x}}_1 + \frac{a_t'}{a_t} \bigl({\mathbf{x}}_t-b_t{\mathbf{x}}_1\bigr)$，从而提供一个简单向量作为回归目标（[命题 5.3.1](03-probability-paths-and-velocities.md#closed-form-vf)）。虽然每个训练样例仅以一个目标样本为条件，但在数据上对条件训练目标取平均，会得到相同的最优边缘速度场；见[定理 5.2.1](02-flow-matching-framework.md#thm-fm-cfm)。因此，条件流匹配可以学得 ${\mathbf{v}}_t({\mathbf{x}}_t)$，无须直接对通常不可计算的边缘速度进行回归。这与变分视角（[定理 2.2.1](../02-variational-perspective/02-ddpm.md#thm-equiv-marginal-kl)）及去噪 score 匹配（[定理 3.3.1](../03-score-based-perspective/03-denoising-score-matching.md#thm-sm-dsm)）使用的是同一条件化原理。

来源：作者借助 AI 辅助编程绘制。

上述讨论总结如下：

<span id="thm-fm-cfm"></span>

**定理 5.2.1：$\mathcal{L}_{\mathrm{FM}}$ 与 $\mathcal{L}_{\mathrm{CFM}}$ 的等价性**

以下关系成立：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{FM}}(\boldsymbol{\phi}) = \mathcal{L}_{\mathrm{CFM}}(\boldsymbol{\phi}) + C,
\end{aligned}
$$

其中 $C$ 是与参数 $\boldsymbol{\phi}$ 无关的常数。此外，两个损失的最优解 ${\mathbf{v}}^*$ 满足

$$
{\mathbf{v}}^*({\mathbf{x}}_t, t) = {\mathbf{v}}_t({\mathbf{x}}_t), \quad \text{for almost every } {\mathbf{x}}_t \sim p_t,
$$

其中 ${\mathbf{v}}_t({\mathbf{x}}_t)$ 定义于[式 (5.2.9)](02-flow-matching-framework.md#eq-marginal-oracle-velocity)。

**定理证明。**

论证以及最优解的推导，与 score 匹配情形下[命题 4.3.1](../04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer)的推理完全相同。

这是条件化技巧第三次得到可计算的训练目标，如[图 5.8](02-flow-matching-framework.md#fig-cfm-illustration)所示。变分、基于 score 与基于流的三类方法，都体现了同一个底层原理。

**说明。**

取 $\pi = p_{\mathrm{data}}$，可以应用 Bayes 法则：

$$
p({\mathbf{x}}_0|{\mathbf{x}}_t) = \frac{p_t({\mathbf{x}}_t|{\mathbf{x}}_0) p_{\mathrm{data}}({\mathbf{x}}_0)}{p_t({\mathbf{x}}_t)},
$$

在基于 score 的模型中，也出现了与[式 (5.2.9)](02-flow-matching-framework.md#eq-marginal-oracle-velocity)类似的分解：

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
    &= \mathbb{E}_{{\mathbf{x}}_0 \sim p(\cdot|{\mathbf{x}}_t)} \left[ \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t|{\mathbf{x}}_0) \right] \\
    &= \mathbb{E}_{{\mathbf{x}}_0 \sim p_{\text{data}}} \left[ \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t|{\mathbf{x}}_0) \cdot \frac{p_t({\mathbf{x}}_t|{\mathbf{x}}_0)}{p_t({\mathbf{x}}_t)} \right],
\end{aligned}
$$

这对应[式 (5.2.9)](02-flow-matching-framework.md#eq-marginal-oracle-velocity)中的边缘化策略。

与[第 5.2.1 节](02-flow-matching-framework.md#subsec-lesson-score)一样，必须显式指定条件密度 $p_t({\mathbf{x}}_t|{\mathbf{z}})$ 与条件速度场 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})$，其中 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0) = \mathcal{N}({\mathbf{x}}_t; \alpha_t {\mathbf{x}}_0, \sigma_t^2 {\mathbf{I}})$，且

$$
{\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{x}}_0) = f(t){\mathbf{x}}_t - \frac{1}{2}g^2(t)\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t|{\mathbf{x}}_0),
$$

 一般条件流匹配框架也需要这两个组成部分。不过，我们尚未在一般情形中构造条件密度 $p_t({\mathbf{x}}_t|{\mathbf{z}})$ 或条件速度场 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})$。[第 5.3 节](03-probability-paths-and-velocities.md#sec-instance-fm)将介绍它们的几种常见实例。

## 5.2.3 扩散模型、一般流匹配与 NODE 的比较 {#subsec-compare-dm-general-fm}

**扩散模型与一般流匹配的比较。** [第 5.2.1 节](02-flow-matching-framework.md#subsec-lesson-score)的认识导向扩展后的 FM 框架，同时保留了相同的底层原理。为突出相似之处，我们在[表 5.1](02-flow-matching-framework.md#tab-framework-comparison)中总结二者。

<span id="tab-framework-comparison"></span>

**表 5.1：** 扩散模型（或 Gaussian FM）与一般 FM 框架的比较。这里的一般 FM 框架指双侧条件化设定，其中 $\mathbf{x}_0 \sim p_{\text{src}}$ 与 $\mathbf{x}_1 \sim p_{\text{tgt}}$ 独立采样。

<div style="overflow-x: auto;" markdown="1">
<table markdown="1">
<tr markdown="1">
<th markdown="block">

**比较项**

</th>
<th markdown="block">

**基于 score 的扩散模型**

</th>
<th markdown="block">

**一般 FM**

</th>
</tr>
<tr markdown="1">
<td markdown="block">

源分布 $p_{\mathrm{src}}$

</td>
<td markdown="block">

Gaussian 先验

</td>
<td markdown="block">

任意

</td>
</tr>
<tr markdown="1">
<td markdown="block">

目标分布 $p_{\mathrm{tgt}}$

</td>
<td markdown="block">

数据分布

</td>
<td markdown="block">

任意

</td>
</tr>
<tr markdown="1">
<td markdown="block">

潜变量分布 $\pi({\mathbf{z}})$

</td>
<td markdown="block">

$p_{\text{data}}$

</td>
<td markdown="block">

见[第 5.3.2 节](03-probability-paths-and-velocities.md#subsec-fm-instantiations)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

条件分布 $p_t({\mathbf{x}}_t|{\mathbf{z}})$

</td>
<td markdown="block">

$\mathcal{N}({\mathbf{x}}_t;\alpha_t{\mathbf{x}}_0,\sigma_t^2{\mathbf{I}})$

</td>
<td markdown="block">

见[第 5.3.2 节](03-probability-paths-and-velocities.md#subsec-fm-instantiations)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

边缘分布 $p_t({\mathbf{x}}_t)$

</td>
<td markdown="block">

$\int p_t({\mathbf{x}}_t|{\mathbf{x}}_0) p_{\mathrm{data}}({\mathbf{x}}_0) \mathop{}\!\mathrm{d}{\mathbf{x}}_0$

</td>
<td markdown="block">

$\int p_t({\mathbf{x}}_t|{\mathbf{z}}) \pi({\mathbf{z}}) \mathop{}\!\mathrm{d}{\mathbf{z}}$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

条件速度 ${\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}})$

</td>
<td markdown="block">

$f(t) {\mathbf{x}} - \frac{1}{2} g^2(t) \nabla \log p_t({\mathbf{x}}|{\mathbf{x}}_0)$

</td>
<td markdown="block">

见[第 5.3.2 节](03-probability-paths-and-velocities.md#subsec-fm-instantiations)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

边缘速度 ${\mathbf{v}}_t({\mathbf{x}})$

</td>
<td markdown="block">

$f(t) {\mathbf{x}} - \frac{1}{2} g^2(t) \nabla \log p_t({\mathbf{x}})$

</td>
<td markdown="block">

见[式 (5.2.9)](02-flow-matching-framework.md#eq-marginal-oracle-velocity)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

学习目标

</td>
<td markdown="block">

$\mathcal{L}_{\text{SM}}=\mathcal{L}_{\text{DSM}}+C$

</td>
<td markdown="block">

$\mathcal{L}_{\mathrm{FM}}=\mathcal{L}_{\mathrm{CFM}}+C$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

底层规则

</td>
<td colspan="2" markdown="block">

Fokker–Planck 方程／连续性方程

</td>
</tr>
</table>
</div>

假设 Gaussian 调度满足[引理 4.5.1](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#forward-sde)中的扩散 SDE 相容条件，Gaussian FM 本质上等价于对应的扩散形式（更多内容见[第 6 章](../06-unified-perspective/index.md#ch-unified-perspective)）；因此，除非明确说明，我们不区分二者。

**与 NODE 的联系。** FM 可视为[第 5.1.2 节](01-normalizing-flows-and-neural-odes.md#subsec-node)所介绍 NODE 的一种无须模拟的替代方案。CNF 在最大似然训练中需要求解 ODE，计算成本较高；FM 则通过简单回归损失直接回归指定速度场，绕过这一计算。关键在于，当连接源分布与目标分布的边缘密度路径固定时，训练中就不再需要精确模拟。

## 5.2.4 （选读）底层规则 {#subsec-fm-theory}

**连续性方程：质量守恒准则。** 类似于[第 5.2.1 节](02-flow-matching-framework.md#subsec-lesson-score)对 PF-ODE 与 Fokker–Planck 方程的分析，现在给出一个准则，验证 ODE 流诱导的密度路径是否与指定路径 $\{p_t\}_{t \in [0,1]}$ 一致。

考虑描述粒子在随时间变化速度场 ${\mathbf{v}}_t$ 下运动的 ODE：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{v}}_t\left({\mathbf{x}}(t)\right).
$$

如[式 (5.2.4)](02-flow-matching-framework.md#eq-pushforward-ode)所示，该 ODE 定义流映射 $\boldsymbol{\Psi}_{s\to t}({\mathbf{x}}_0)$，适用于任意 $s,t \in [0,1]$，特别地，它将初始点 ${\mathbf{x}}_0 \sim p_{\mathrm{src}}$ 从时刻 $0$ 输运到时刻 $t$ 的状态。时刻 $t$ 的诱导分布由推前给出：

<span id="eq-ode-pushforward"></span>

$$
\begin{aligned}
p_t^{\mathrm{fwd}}({\mathbf{x}})
    = \int \delta \left({\mathbf{x}} - \boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)\right)
      p_{\mathrm{src}}({\mathbf{x}}_0) \mathrm{d}{\mathbf{x}}_0=: \boldsymbol{\Psi}_{0\to t}\# p_{\mathrm{src}},
\end{aligned}
\tag{5.2.12}
$$

因此，$\boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0) \sim p_t^{\mathrm{fwd}}$ 在 ${\mathbf{x}}_0 \sim p_{\mathrm{src}}$ 时成立。类似地，也可以从 ${\mathbf{x}}_1 \sim p_{\mathrm{tgt}}$ 反向输运到 $p_{\mathrm{src}}$，所用映射为 $\boldsymbol{\Psi}_{1\to 0}({\mathbf{x}}_1)$。

假设给定指定的密度路径 $\{p_t\}_{t \in [0,1]}$，并构造速度场 $\{{\mathbf{v}}_t\}_{t \in [0,1]}$ 来定义粒子流。这自然引出以下问题：

<span id="question-5-2-1"></span>

**问题 5.2.1**

在什么条件下，流诱导的密度 $p_t^{\mathrm{fwd}}$ 与目标密度 $p_t$ 对所有 $t \in [0,1]$ 都完全匹配？

一旦两种密度演化一致，就可以通过求解 ODE，利用 ODE 流在 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 之间灵活输运样本。

与[式 (5.2.5)](02-flow-matching-framework.md#eq-p-t-fwd-rev)一样，可以通过*连续性方程*在理论上验证这种一致性；它刻画随时间演化密度中的质量守恒：

<span id="thm-continuity-mass"></span>

**定理 5.2.2：质量守恒准则**

流诱导密度 $p_t^{\mathrm{fwd}}$ 等于指定路径 $p_t$，对所有 $t \in [0,1]$ 都成立，即

$$
p_t^{\mathrm{fwd}} = p_t, \quad \text{for all } t \in [0,1],
$$

当且仅当二元组 $(p_t, {\mathbf{v}}_t)$ 满足连续性方程：

$$
\partial_t p_t({\mathbf{x}}) + \nabla_{\mathbf{x}} \cdot (p_t({\mathbf{x}}) {\mathbf{v}}_t({\mathbf{x}})) = 0,
$$

对所有 $t\in[0,1]$ 与 ${\mathbf{x}}$ 成立。

**定理证明。**

概念上的推导见[第 D.3.2 节](../../appendices/d-supplementary-proofs/03-flow-based-perspective.md#app-sec-mass-conservation)，更严格的处理见（[Villani and others, 2008](../../references.md#ref-villani2008optimal)）中的“质量守恒公式”。

**从条件路径到边缘路径。** 如[第 5.2.2 节](02-flow-matching-framework.md#subsec-fm-framework)所示，首先定义条件概率路径 $p_t(\cdot| {\mathbf{z}})$ 及对应条件速度场 ${\mathbf{v}}_t(\cdot| {\mathbf{z}})$，再通过下式构造边缘速度场：

$$
{\mathbf{v}}_t({\mathbf{x}})
    = \int {\mathbf{v}}_t({\mathbf{x}}| {\mathbf{z}}) \frac{p_t({\mathbf{x}}| {\mathbf{z}}) \pi({\mathbf{z}})}{p_t({\mathbf{x}})}   \mathop{}\!\mathrm{d} {\mathbf{z}},
$$

即[式 (5.2.9)](02-flow-matching-framework.md#eq-marginal-oracle-velocity)。不过，仍需保证所得边缘速度 ${\mathbf{v}}_t$ 诱导的 ODE 流，其密度路径与指定的 $p_t$ 一致。所幸，这一验证可以完全在条件层面完成：若每个条件速度场 ${\mathbf{v}}_t(\cdot| {\mathbf{z}})$ 都诱导对应的条件密度路径 $p_t(\cdot| {\mathbf{z}})$，那么所得边缘速度 ${\mathbf{v}}_t$ 也会诱导正确的边缘路径。下面正式陈述这一结果，并集中给出完整推导，明确展示边缘速度场为何具有这一形式：

<span id="marginal-vf"></span>

**命题 5.2.3：边缘向量场生成给定的边缘密度**

若条件速度场 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 诱导的条件密度路径与 $p_t(\cdot|{\mathbf{z}})$ 匹配（从 $p_0(\cdot|{\mathbf{z}})$ 出发），则[式 (5.2.9)](02-flow-matching-framework.md#eq-marginal-oracle-velocity)定义的边缘速度场 ${\mathbf{v}}_t(\cdot)$，诱导与 $p_t(\cdot)$ 一致的边缘密度路径，其初始分布为 $p_0(\cdot)$。

**命题证明。**

验证二元组 $(p_t, {\mathbf{v}}_t)$ 满足连续性方程，即可得到结果。我们以反向推导的方式论证，帮助理解边缘化速度场为何具有[式 (5.2.9)](02-flow-matching-framework.md#eq-marginal-oracle-velocity)的形式。由于条件速度场 ${\mathbf{v}}_t(\cdot|{\mathbf{z}})$ 诱导的密度路径与条件密度 $p_t(\cdot|{\mathbf{z}})$ 对于 ${\mathbf{z}} \sim \pi$ 匹配，每个条件二元组都满足连续性方程：

<span id="eq-continuity-conditional-pt"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} p_t({\mathbf{x}}|{\mathbf{z}}) =- \nabla_{\mathbf{x}} \cdot \big({\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}}) p_t({\mathbf{x}}|{\mathbf{z}})\big).
\end{aligned}
\tag{5.2.13}
$$

我们希望寻找速度场 ${\color{orange}{\mathbf{v}}_t(\cdot)}$，使其诱导的密度与边缘密度 $p_t$ 一致，即满足

<span id="eq-p-t-continuity"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} p_t({\mathbf{x}}) =- \nabla_{\mathbf{x}} \cdot \big({\color{orange}{\mathbf{v}}_t({\mathbf{x}})} p_t({\mathbf{x}})\big).
\end{aligned}
\tag{5.2.14}
$$

从[式 (5.2.7)](02-flow-matching-framework.md#eq-p-t)中 $p_t$ 的定义出发，

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} p_t({\mathbf{x}}) &=\int \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} p_t({\mathbf{x}}_t|{\mathbf{z}})\pi({\mathbf{z}})\mathop{}\!\mathrm{d} {\mathbf{z}} \\
  &=  - \int \nabla_{\mathbf{x}} \cdot\Big(  {\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}}) p_t({\mathbf{x}} | {\mathbf{z}})  \Big)\pi({\mathbf{z}}) \mathop{}\!\mathrm{d} {\mathbf{z}} \quad  \\
  &=  -\nabla_{\mathbf{x}} \cdot \Big( \int  {\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}}) p_t({\mathbf{x}} | {\mathbf{z}})
 \pi({\mathbf{z}}) \mathop{}\!\mathrm{d} {\mathbf{z}}\Big),
\end{aligned}
$$

其中第二个等号由应用[式 (5.2.13)](02-flow-matching-framework.md#eq-continuity-conditional-pt)得到。将其与[式 (5.2.14)](02-flow-matching-framework.md#eq-p-t-continuity)右侧比较可知，忽略一个无散度项，有

$$
{\color{orange}{\mathbf{v}}_t({\mathbf{x}})} p_t({\mathbf{x}}) = \int {\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}}) p_t({\mathbf{x}}|{\mathbf{z}}) \pi({\mathbf{z}}) \mathop{}\!\mathrm{d} {\mathbf{z}}.
$$

因此，可以定义

$$
{\color{orange}{\mathbf{v}}_t({\mathbf{x}})} := \int {\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}}) \frac{p_t({\mathbf{x}}|{\mathbf{z}})}{p_t({\mathbf{x}})} \pi({\mathbf{z}}) \mathop{}\!\mathrm{d} {\mathbf{z}},
$$

这恰好是[式 (5.2.9)](02-flow-matching-framework.md#eq-marginal-oracle-velocity)的形式。该定理的证明本质上就是将上述论证反向进行。

这一联系使我们能够将可能不可计算的边缘速度场的构造，归结为定义更简单的条件场 ${\mathbf{v}}_t(\cdot| {\mathbf{z}})$；这些条件场在构造上更易处理。

[^1]: 几种相关方法共享使用连续时间流在端点分布之间输运的核心思想，尽管其形式略有不同。这些方法包括流匹配（FM）（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)；[Neklyudov et al., 2023](../../references.md#ref-neklyudov2023action)）、Rectified Flow（RF）（[Liu, 2022](../../references.md#ref-liu2022rectified)；[Heitz et al., 2023](../../references.md#ref-heitz2023iterative)）以及 Stochastic Interpolants（[Albergo et al., 2023](../../references.md#ref-albergo2023stochastic)；[Albergo and Vanden-Eijnden, 2023](../../references.md#ref-albergobuilding)；[Ma et al., 2024](../../references.md#ref-ma2024sit)）。这里统一使用 FM 这一术语。

[^2]: 为与 FM 文献的标准记号一致，相比前面各节，我们将时间轴反转：$t=0$ 对应源分布，$t=1$ 对应目标分布。

[^3]: 关键区别在于，概率路径指定的是*分布*的演化，通常不指定实现它的粒子层面输运。只有增加输运结构，才会确定具体速度。例如，足够正则且可逆的流映射 ${\boldsymbol{\Psi}}_{0\to t}$ 通过 $\mathbf{v}_t=\partial_t{\boldsymbol{\Psi}}_{0\to t}\circ{\boldsymbol{\Psi}}_{0\to t}^{-1}$ 确定其 Euler 速度。同样，一旦指定前向扩散 SDE，对应 PF-ODE 就选择了特定速度 $\mathbf{v}_t(\mathbf{x}) = f(t)\mathbf{x} -\tfrac12g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$。因此，两种情形都在边缘路径 $\{p_t\}$ 本身之外，增加了粒子层面的输运结构。

[^4]: Fokker–Planck 方程的确定性对应形式，不含扩散项。
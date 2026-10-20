# 6.3 扩散模型中的等价性 {#sec-equivalent-parametrizations}

扩散模型的许多表述看起来各不相同，因为它们采用了不同的预测类型、训练目标、噪声调度或连续时间动力学。本节将明确说明，这些表述在严格理论层面上究竟在什么意义下可以等价，以及在实践中，学习得到的模型为何仍可能表现不同。

在精确解或 oracle 层面上，我们将遇到几种等价性概念：

-   **预测。** 给定带噪观测 ${\mathbf{x}}_t$ 和前向调度，干净数据、噪声、score 和速度的 oracle 包含相同的信息，并且可以通过代数运算相互转换。

-   **训练目标。** 以不同预测类型表述的平方误差目标之间可以相差一个依赖于时间的正因子，同时具有相同的逐点 oracle 最小化解。这些因子决定了训练时不同噪声水平所获得的权重大小。

-   **仿射前向过程。** 在粒子层面上，仿射前向过程

    $$
    {\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}
    $$

    可以通过状态缩放及相应的标准坐标，与标准仿射形式建立联系。当这一坐标随时间单调演变时，这种关系也定义了一种时间重参数化。

-   **边缘密度路径。** 在密度层面上，不同的确定性和随机动力学可以产生相同的边缘路径 $\{p_t\}_{t\in[0,T]}$。它们的样本轨迹可能不同，而 $p_t$ 的演变由相应的连续性方程或 Fokker-Planck 方程支配。

我们将从预测到采样逐步建立这些联系。[第 6.3.1 节](03-equivalent-parameterizations.md#subsec-four-equiv-para)首先联系四种常见的 oracle 预测类型及其训练损失。[第 6.3.2 节](03-equivalent-parameterizations.md#subsec-pf-ode-different-para)随后说明，这些预测如何导出同一 oracle PF–ODE 的不同表达式。接着，[第 6.3.3 节](03-equivalent-parameterizations.md#subsec-all-flows-are-equiv)将仿射概率路径与标准线性形式和三角函数形式联系起来。最后，[第 6.3.4 节](03-equivalent-parameterizations.md#subsec-concept-why-v-affine)进一步阐释标准线性调度与速度预测的直观含义，包括它们对训练和数值积分的影响。

这些精确关系提供了共同的数学基础，而实践中的具体选择仍然有意义。有限规模的神经网络可能以不同方式逼近等价的 oracle 量，损失加权、优化、模型架构和数值积分也可能导致不同的训练与采样行为。

## 6.3.1 四种预测类型的等价性 {#subsec-four-equiv-para}

我们首先分析[式（6.2.5）](02-training-objectives.md#eq-general-dm-loss-d)中组成部分（B）的设计选择。

我们已经看到，四种预测类型并非彼此独立的选择，而是对同一个底层量的不同描述。例如，噪声预测与干净数据预测直接相关（[第 2.2.4 节](../02-variational-perspective/02-ddpm.md#subsec-ddpm-prediction)），score 预测与噪声预测也是如此（[第 3.4 节](../03-score-based-perspective/04-noise-conditional-score-networks.md#sec-smld)）。这一反复出现的模式指向一个更深层的原理：四种参数化在代数上全部等价，可以通过简单的变换相互转换。为精确表述这一联系，我们给出以下命题，并在[图 6.1](03-equivalent-parameterizations.md#fig-equiv-parametrizations)中加以展示；该命题沿用（[Kingma et al., 2021](../../references.md#ref-kingma2021variational)）的结果。

<span id="equi-para"></span>

**命题 6.3.1：参数化的等价性**

设使各自目标最小化的最优预测为

$$
\boldsymbol{\epsilon}^*({\mathbf{x}}_t,t) , \quad {\mathbf{x}}^*({\mathbf{x}}_t,t), \quad {\mathbf{s}}^*({\mathbf{x}}_t,t) , \quad {\mathbf{v}}^*({\mathbf{x}}_t,t),
$$

它们分别对应噪声、干净数据、score 和速度参数化，并满足以下等价关系：

<span id="eq-predictions-equivalence"></span>

$$
\begin{aligned}
\boldsymbol{\epsilon}^*({\mathbf{x}}_t,t) &= -\sigma_t {\mathbf{s}}^*({\mathbf{x}}_t,t), \\
{\mathbf{x}}^*({\mathbf{x}}_t,t) &= \frac{1}{\alpha_t} {\mathbf{x}}_t
+ \frac{\sigma_t^2}{\alpha_t} {\mathbf{s}}^*({\mathbf{x}}_t,t), \\
{\mathbf{v}}^*({\mathbf{x}}_t,t) &= \alpha_t' {\mathbf{x}}^* + \sigma_t' \boldsymbol{\epsilon}^*
= f(t){\mathbf{x}}_t - \frac{1}{2} g^2(t){\mathbf{s}}^*({\mathbf{x}}_t,t).
\end{aligned}
\tag{6.3.1}
$$

这里，$f(t)$ 和 $g(t)$ 与 $\alpha_t$ 和 $\sigma_t$ 之间的关系由[引理 4.5.1](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#forward-sde)给出。此外，这些最小化解满足[式（6.2.1）](02-training-objectives.md#eq-noise-def)、[式（6.2.2）](02-training-objectives.md#eq-clean-def)、[式（6.2.3）](02-training-objectives.md#eq-score-def)和[式（6.2.4）](02-training-objectives.md#eq-velocity-def)中的恒等式。

**命题的证明。**

证明与[命题 4.3.1](../04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer)的证明类似；该命题分析了 DSM 目标下各种匹配损失的全局最优解。详见[第 D.4 节](../../appendices/d-supplementary-proofs/04-unified-perspective.md#app-sec-equiv-para)。

<span id="fig-equiv-parametrizations"></span>

![score、噪声、干净数据和速度的 oracle 预测之间的代数转换](../../../en/assets/chapter-06/equiv-parametrizations.png)

**图 6.1：** **四种 oracle 参数化之间的等价关系。** 利用前向调度与观测状态 ${\mathbf{x}}_t$，每一种 oracle 预测都可以转换为其他预测。例如，${\mathbf{v}}^*=\alpha_t'{\mathbf{x}}^*+\sigma_t'\boldsymbol{\epsilon}^*$，而 ${\mathbf{x}}$ 预测与 ${\boldsymbol{\epsilon}}$ 预测满足 ${\mathbf{x}}_t=\alpha_t{\mathbf{x}}^*+\sigma_t\boldsymbol{\epsilon}^*$。

来源：作者绘制。

这个命题最自然的理解是 *oracle 信息*的等价性。在任意噪声水平下，已知一种 oracle 预测，再结合带噪观测 ${\mathbf{x}}_t$ 和前向调度，就能通过[式（6.3.1）](03-equivalent-parameterizations.md#eq-predictions-equivalence)中的恒等式确定其他预测。例如，score 决定了以后验均值表示的干净数据预测，而这两者中的任意一个都能确定 PF-ODE 的速度。因此，在 oracle 层面上，这些参数化包含了关于底层扩散过程的相同信息。

然而，对于学习得到的模型，必须区分这种 oracle 等价性与神经网络直接预测的量。可以从以下各项中任选一项

$$
\boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t),
\quad
{\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t),
\quad
{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t),
\quad
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
$$

作为网络输出。一旦将某一种预测作为主要输出，就可以利用[式（6.3.1）](03-equivalent-parameterizations.md#eq-predictions-equivalence)中的同样代数转换来定义其他预测。特别地，对于参数为 $\boldsymbol{\phi}^\times$ 的已训练模型，这些转换后的预测由构造保证严格一致；例如，

$$
{\mathbf{x}}
=
\alpha_t
{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t)
+
\sigma_t
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t).
$$

这些转换在训练期间和训练结束后都有用。例如，噪声预测可以转换为用于反向 SDE 采样的 score，也可以转换为用于 PF-ODE 采样的速度。

网络预测的参数化还应与*训练损失*区分开来。如[式（6.2.5）](02-training-objectives.md#eq-general-dm-loss-d)的组成部分（B）所述，网络输出的量不必就是计算损失所用的量：可以先将预测转换为另一种参数化，再计算目标。此类转换通常会为平方误差目标引入依赖于时间的权重，后面的[式（6.3.6）](03-equivalent-parameterizations.md#eq-s-e-x-v-equi)将对此加以说明。

这些区别对有限规模的神经网络很重要。虽然 oracle 目标在信息上等价，但直接预测不同的量会使网络表示和优化不同的函数，相应的目标也可能对不同噪声水平赋予不同权重。因此，在不同预测参数化下独立训练的模型，不一定产生相同的经验预测或相同的 PF-ODE。本命题中的等价性所涉及的是理想 oracle 预测携带的信息；网络输出的参数化和训练目标仍是彼此独立且有实际意义的设计选择。

## 6.3.2 不同参数化下的 PF-ODE {#subsec-pf-ode-different-para}

PF-ODE 具有若干等价的参数化形式（score、噪声、去噪结果和速度）。尽管原则上可以相互替换，但具体选择会产生实际影响：它会改变向量场的刚性、离散化误差的表现以及优化的难易程度。利用高级 ODE 求解器进行快速采样时（见[第 9 章](../09-numerical-solvers/index.md#ch-numerical-solvers)），实践中常采用 ${\boldsymbol{\epsilon}}$ 或 ${\mathbf{x}}$ 预测，因为它们与求解器的输入形式相适应，并能减少误差累积。训练仅需少量函数求值的生成器时（见[第 11 章](../11-fast-generators/index.md#ch-fast-scratch)），${\mathbf{x}}$ 或 ${\mathbf{v}}$ 预测通常会带来更平滑的目标和更好的步间一致性。

我们将写出各个参数化下的 PF-ODE，并利用[式（6.3.1）](03-equivalent-parameterizations.md#eq-predictions-equivalence)明确给出转换关系。结果归纳在以下命题中。

<span id="pf-ode-para"></span>

**命题 6.3.2：不同参数化下的 PF-ODE**

设 $\alpha_t$ 和 $\sigma_t$ 为前向扰动调度，并将时间导数记为 $\alpha_t' := \tfrac{\mathop{}\!\mathrm{d} \alpha_t}{\mathop{}\!\mathrm{d} t}$ 和 $\sigma_t' := \tfrac{\mathop{}\!\mathrm{d} \sigma_t}{\mathop{}\!\mathrm{d} t}$。则经验 PF-ODE 具有以下等价形式

<span id="eq-clean-pf-ode"></span>

$$
\begin{aligned}
    \frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
&= \frac{\alpha_t'}{\alpha_t} {\mathbf{x}}(t)
 - \sigma_t \left(\frac{\alpha_t'}{\alpha_t}-\frac{\sigma_t'}{\sigma_t}\right)
   \boldsymbol{\epsilon}^*({\mathbf{x}}(t), t) \\
&= \frac{\sigma_t'}{\sigma_t} {\mathbf{x}}(t)
 + \alpha_t \left(\frac{\alpha_t'}{\alpha_t}-\frac{\sigma_t'}{\sigma_t}\right)
   {\mathbf{x}}^*({\mathbf{x}}(t), t) \\
&= \frac{\alpha_t'}{\alpha_t} {\mathbf{x}}(t)
 + \sigma_t^{2} \left(\frac{\alpha_t'}{\alpha_t}-\frac{\sigma_t'}{\sigma_t}\right)
   {\mathbf{s}}^*({\mathbf{x}}(t), t) \\
   &= \alpha_t'{\mathbf{x}}^*({\mathbf{x}}(t), t) + \sigma_t' {\boldsymbol{\epsilon}}^*({\mathbf{x}}(t), t)\\
&= {\mathbf{v}}^*({\mathbf{x}}(t), t).
\end{aligned}
\tag{6.3.2}
$$

为得到 Score SDE 的记法，回顾[引理 4.5.1](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#forward-sde)。若令

$$
f(t) = \frac{\alpha_t'}{\alpha_t},
\quad
g^2(t) = \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \big(\sigma_t^2\big)
         - 2 \frac{\alpha_t'}{\alpha_t} \sigma_t^2
       = 2\sigma_t\sigma_t' - 2 \frac{\alpha_t'}{\alpha_t} \sigma_t^2,
$$

则 PF-ODE 可以写成我们熟悉的 Score SDE 形式：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
= f(t) {\mathbf{x}}(t)  -  \frac{1}{2} g^2(t) {\mathbf{s}}^*({\mathbf{x}}(t),t).
$$

为具体说明如何将 PF-ODE 离散化以进行采样，我们将在[第 9.2 节](../09-numerical-solvers/02-ddim.md#sec-revisit-ddim)介绍一种广泛使用的扩散 ODE 采样器——DDIM 方法——的更新规则。这个例子将展示 Euler 离散化与 PF-ODE 如何自然地联系起来。

## 6.3.3 所有仿射流的等价性 {#subsec-all-flows-are-equiv}

接下来，我们分析[式（6.2.5）](02-training-objectives.md#eq-general-dm-loss-d)中组成部分（A）的设计选择。

**状态层面的等价性。** FM（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)）和 RF（[Liu, 2022](../../references.md#ref-liu2022rectified)）采用的一种方便的标准插值为

$$
{\mathbf{x}}_t^{\mathrm{FM}}=(1-t) {\mathbf{x}}_0+t {\boldsymbol{\epsilon}} = {\mathbf{x}}_{0}+t({\boldsymbol{\epsilon}}-{\mathbf{x}}_0),
$$

其速度为常向量 ${\boldsymbol{\epsilon}}-{\mathbf{x}}_0$。本小节的关键在于，这一选择在形式上的简单性并非本质特征：任意仿射插值

$$
{\mathbf{x}}_t=\alpha_t {\mathbf{x}}_0+\sigma_t {\boldsymbol{\epsilon}}
$$

都可以写成对标准路径进行时间重参数化和缩放后的形式。定义

$$
c(t):=\alpha_t+\sigma_t,
\qquad
\tau(t):=\frac{\sigma_t}{\alpha_t+\sigma_t}
\quad\big(c(t)\neq 0\big).
$$

直接进行代数改写可得

$$
\begin{aligned}
{\mathbf{x}}_t
&=\alpha_t {\mathbf{x}}_0+\sigma_t {\boldsymbol{\epsilon}}
\\
&=\big(\alpha_t+\sigma_t\big) \left(\frac{\alpha_t}{\alpha_t+\sigma_t}{\mathbf{x}}_0+\frac{\sigma_t}{\alpha_t+\sigma_t}{\boldsymbol{\epsilon}}\right) \\
&=c(t) \left(\big(1-\tau(t)\big){\mathbf{x}}_0+\tau(t){\boldsymbol{\epsilon}}\right)
=c(t) {\mathbf{x}}_{\tau(t)}^{\mathrm{FM}}.
\end{aligned}
$$

因此，只要 $c(t)\neq0$，每条仿射路径都能逐点表示为标准路径乘以一个标量缩放因子。若进一步要求 $\tau(t)$ 在所考虑的时间区间内连续且严格单调，那么 $t\mapsto\tau(t)$ 也就构成了有效的时间变量变换，两条路径通过时间重参数化和缩放 $c(t)$ 建立联系。对于通常采用的前向扩散调度，信噪比单调衰减意味着 $\tau(t)$ 单调演变。

对于相应的速度，对 ${\mathbf{x}}_t=c(t){\mathbf{x}}_{\tau(t)}^{\mathrm{FM}}$ 应用链式法则：

$$
\begin{aligned}
{\mathbf{v}}({\mathbf{x}}_t,t)
&:= \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\left(\alpha_t{\mathbf{x}}_0 +\sigma_t {\boldsymbol{\epsilon}}\right)
\\&=\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\big(c(t){\mathbf{x}}_{\tau(t)}^{\mathrm{FM}}\big)
\\
&=c'(t) {\mathbf{x}}_{\tau(t)}^{\mathrm{FM}}+c(t) \tau'(t) \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{x}}^{\mathrm{FM}}_{s}\bigg|_{s=\tau(t)} \\
&=c'(t) {\mathbf{x}}_{\tau(t)}^{\mathrm{FM}}+c(t) \tau'(t) {\mathbf{v}}^{\mathrm{FM}} \left({\mathbf{x}}_{\tau(t)}^{\mathrm{FM}},\tau(t)\right),
\end{aligned}
$$

因为沿标准路径有 ${\mathbf{v}}^{\mathrm{FM}}({\mathbf{x}}^{\mathrm{FM}}_{\tau},\tau)=-{\mathbf{x}}_0+{\boldsymbol{\epsilon}}$。

我们将上述推导概括为以下命题中的正式表述。

<span id="equiv-interpolation"></span>

**命题 6.3.3：仿射流的等价性**

设 ${\mathbf{x}}_t^{\mathrm{FM}}=(1-t){\mathbf{x}}_0+t{\boldsymbol{\epsilon}}$ 且 ${\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}$，其中 $c(t):=\alpha_t+\sigma_t\neq0$ 且 $\tau(t):=\sigma_t/(\alpha_t+\sigma_t)$。则

<span id="eq-fm-general-affine-formula"></span>

$$
\begin{aligned}
{\mathbf{x}}_t
&=
c(t){\mathbf{x}}_{\tau(t)}^{\mathrm{FM}},
\\
{\mathbf{v}}({\mathbf{x}}_t,t)
&=
c'(t){\mathbf{x}}_{\tau(t)}^{\mathrm{FM}}
+
c(t)\tau'(t)
{\mathbf{v}}^{\mathrm{FM}}
\left(
{\mathbf{x}}_{\tau(t)}^{\mathrm{FM}},\tau(t)
\right).
\end{aligned}
\tag{6.3.3}
$$

因此，每个满足 $c(t)\neq0$ 的仿射插值都具有这种标准的状态层面表示。若 $\tau(t)$ 还连续且严格单调，那么它也可以解释为时间重参数化与标量缩放 $c(t)$ 的结合。

**与三角函数流的等价性。** 另一种广泛使用的仿射流是三角函数插值（[Salimans and Ho, 2021](../../references.md#ref-salimans2021progressive)；[Albergo et al., 2023](../../references.md#ref-albergo2023stochastic)；[Lu and Song, 2024](../../references.md#ref-lu2024simplifying)）。作为一个具体例子，我们也将说明，*任意*仿射流都可以表示为这种形式。三角函数路径定义为

<span id="eq-trig-para"></span>

$$
\begin{aligned}
{\mathbf{x}}^{\mathrm{Trig}}_u := \cos(u) {\mathbf{x}}_0 + \sin(u){\boldsymbol{\epsilon}} .
\end{aligned}
\tag{6.3.4}
$$

令 $R_t := \sqrt{\alpha_t^2 + \sigma_t^2}$，并假设 $R_t>0$。选择角度 $\tau_t$ 使得

$$
\cos \tau_t = \frac{\alpha_t}{R_t},
\qquad
\sin \tau_t = \frac{\sigma_t}{R_t}.
$$

则每个仿射插值 ${\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}$ 都能表示为逐点缩放并重新指定时间的三角函数路径：

<span id="eq-any-to-trig"></span>

$$
\begin{aligned}
{\mathbf{x}}_t
= \alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}
= R_t \left(\frac{\alpha_t}{R_t}{\mathbf{x}}_0+\frac{\sigma_t}{R_t}{\boldsymbol{\epsilon}}\right)
= R_t {\mathbf{x}}^{\mathrm{Trig}}_{\tau_t}.
\end{aligned}
\tag{6.3.5}
$$

$(\alpha_t,\sigma_t)$ 是平面上的一个点。用 $R_t$ 归一化后，该点位于单位圆上，由此确定角度 $\tau_t$，进而确定状态 ${\mathbf{x}}^{\mathrm{Trig}}_{\tau_t}$；半径 $R_t$ 则给出整体尺度。若还能将 $\tau_t$ 选为在所考虑区间内关于 $t$ 连续且严格单调的函数，那么这种状态层面的表示也定义了一种有效的时间重参数化。

对 ${\mathbf{x}}^{\mathrm{Trig}}_u$ 关于 $u$ 求导，可得其速度，

$$
{\mathbf{v}}^{\mathrm{Trig}}_u = -\sin(u) {\mathbf{x}}_0 + \cos(u) {\boldsymbol{\epsilon}}.
$$

通过与[式（6.3.5）](03-equivalent-parameterizations.md#eq-any-to-trig)相同的变量变换，这一关系给出了速度的闭式转换（其他参数化也可以类似地转换）。

综上所述，我们得到以下结论：

<span id="conclusion-6-3-1"></span>

**结论 6.3.1**

无论调度 $(\alpha_t,\sigma_t)$ 采用何种形式，包括 VE、VP（例如三角函数形式）、FM 或 RF，仿射插值都可以通过适当的时间变量变换和标量缩放相互转换。

**四种参数化的训练目标。** 设 ${\mathbf{x}}_t=\alpha_t {\mathbf{x}}_0+\sigma_t {\boldsymbol{\epsilon}}$，其中 $\sigma_t>0$，且 $(\alpha_t,\sigma_t)$ 可微并满足 $\alpha_t'\sigma_t-\alpha_t\sigma_t'\neq 0$。考虑以下 oracle 目标

$$
{\boldsymbol{\epsilon}}^*({\mathbf{x}}_t,t)={\mathbb{E}}[{\boldsymbol{\epsilon}}|{\mathbf{x}}_t],\quad
{\mathbf{x}}_0^*({\mathbf{x}}_t,t)={\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_t],\quad
{\mathbf{v}}^*({\mathbf{x}}_t,t)={\mathbb{E}}[\alpha_t'{\mathbf{x}}_0+\sigma_t'{\boldsymbol{\epsilon}}|{\mathbf{x}}_t].
$$

由[命题 6.3.1](03-equivalent-parameterizations.md#equi-para)，它们满足

$$
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)
= -\frac{1}{\sigma_t} {\boldsymbol{\epsilon}}^*({\mathbf{x}}_t,t)
= \frac{\alpha_t}{\sigma_t^2} \left({\mathbf{x}}_0^*({\mathbf{x}}_t,t)-\frac{{\mathbf{x}}_t}{\alpha_t}\right),
\quad
{\mathbf{v}}^*=\alpha_t' {\mathbf{x}}_0^*+\sigma_t' {\boldsymbol{\epsilon}}^*.
$$

采用以下输出头转换

$$
{\mathbf{s}}_{{\boldsymbol{\phi}}} \equiv -\frac{1}{\sigma_t} {\boldsymbol{\epsilon}}_{{\boldsymbol{\phi}}}
\equiv \frac{\alpha_t}{\sigma_t^2} \left({\mathbf{x}}_{{\boldsymbol{\phi}}}-\frac{{\mathbf{x}}_t}{\alpha_t}\right),
$$

以及从速度到 score 的转换

$$
{\mathbf{s}}_{{\boldsymbol{\phi}}}
= \frac{\alpha_t}{\sigma_t(\alpha_t'\sigma_t-\alpha_t\sigma_t')} {\mathbf{v}}_{{\boldsymbol{\phi}}}
- \frac{\alpha_t'}{\sigma_t(\alpha_t'\sigma_t-\alpha_t\sigma_t')} {\mathbf{x}}_t,
$$

各样本的平方损失之间仅相差依赖于时间的权重：

<span id="eq-s-e-x-v-equi"></span>

$$
\begin{aligned}
\big\|{\mathbf{s}}_{{\boldsymbol{\phi}}}-\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)\big\|_2^2
&=\frac{1}{\sigma_t^2} \big\|{\boldsymbol{\epsilon}}_{{\boldsymbol{\phi}}}-{\boldsymbol{\epsilon}}^*\big\|_2^2 \\
&=\frac{\alpha_t^2}{\sigma_t^4} \big\|{\mathbf{x}}_{{\boldsymbol{\phi}}}-{\mathbf{x}}_0^*\big\|_2^2 \\
&=\left(\frac{\alpha_t}{\sigma_t(\alpha_t'\sigma_t-\alpha_t\sigma_t')}\right)^{ 2}
\big\|{\mathbf{v}}_{{\boldsymbol{\phi}}}-{\mathbf{v}}^*\big\|_2^2 .
\end{aligned}
\tag{6.3.6}
$$

根据[命题 6.3.3](03-equivalent-parameterizations.md#equiv-interpolation)，任意仿射流 ${\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}$ 都可以通过 ${\mathbf{x}}_t=c(t){\mathbf{x}}^{\mathrm{FM}}_{\tau(t)}$ 转换为标准 FM 路径，其中 $c(t)=\alpha_t+\sigma_t$ 且 $\tau(t)=\sigma_t/(\alpha_t+\sigma_t)$。求导可得

$$
{\mathbf{v}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)=c'(t){\mathbf{x}}^{\mathrm{FM}}_{\tau(t)}
+c(t)\tau'(t){\mathbf{v}}^{\mathrm{FM}}_{{\boldsymbol{\phi}}} \left({\mathbf{x}}_{\tau(t)}^{\mathrm{FM}},\tau(t)\right),
\qquad
{\mathbf{x}}_{\tau(t)}^{\mathrm{FM}}=\frac{{\mathbf{x}}_t}{c(t)},
$$

同样的关系也适用于 ${\mathbf{v}}^*$。因此，速度损失按如下方式变换

$$
\begin{aligned}
\big\|{\mathbf{v}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)-{\mathbf{v}}^*({\mathbf{x}}_t,t)\big\|_2^2
= \big(c(t) \tau'(t)\big)^2
\Big\|{\mathbf{v}}^{\mathrm{FM}}_{{\boldsymbol{\phi}}} \left(\frac{{\mathbf{x}}_t}{c(t)},\tau(t)\right)
-\big({\mathbf{v}}^{\mathrm{FM}}\big)^* \left(\frac{{\mathbf{x}}_t}{c(t)},\tau(t)\right)\Big\|_2^2.
\end{aligned}
$$

根据以上观察，我们得到以下结论：

<span id="conclusion-6-3-2"></span>

**结论 6.3.2**

score、噪声、干净数据和速度训练目标在理论上等价，其差别仅在于时间相关权重（对于速度，还涉及包含 ${\mathbf{x}}_t$ 的仿射输出头转换），这些权重由 $(\alpha_t,\sigma_t)$ 决定。

## 6.3.4 （选读）参数化与标准流的分析 {#subsec-concept-why-v-affine}

尽管前面几节已经说明，四种参数化在数学上全部等价并且可以相互转换，前向仿射加噪流也等价于标准形式

$$
{\mathbf{x}}_t^{\mathrm{FM}} = (1-t) {\mathbf{x}}_0 + t {\boldsymbol{\epsilon}},
$$

本小节仍将提供进一步的直观解释，并分析将 ${\mathbf{v}}$ 预测参数化与这种标准仿射流结合使用的潜在优势。

本小节关注一个简单的问题：*不同的参数化和调度如何影响模型所学习的内容以及采样方式？* 我们从两个概念视角展开：

-   **回归目标与调度。** 我们重点说明，为何将 ${\mathbf{v}}$ 预测与标准线性调度 $(\alpha_t,\sigma_t)=(1-t,t)$ 结合是自然的选择：它使条件速度目标的尺度不随时间改变，并消除了速度分解中显式出现的调度曲率项。

-   **对求解器的影响。** 我们考察这种参数化与数值积分之间的相互作用，而将求解器的详细分析留到[第 9 章](../09-numerical-solvers/index.md#ch-numerical-solvers)。

在继续讨论之前，我们先区分两类速度场，以避免歧义。作为易于计算的训练目标，*条件速度*定义为

$$
{\mathbf{v}}_t({\mathbf{x}}_t |{\mathbf{z}}) = {\mathbf{x}}_t' = \alpha_t' {\mathbf{x}}_0 + \sigma_t' {\boldsymbol{\epsilon}},
\quad \text{where } {\mathbf{z}} = ({\mathbf{x}}_0, {\boldsymbol{\epsilon}}),
$$

而在推断时求解 PF-ODE、用于移动样本的 *oracle（边缘化）速度*为

$$
{\mathbf{v}}^*({\mathbf{x}}, t) = {\mathbb{E}}\big[{\mathbf{v}}_t(\cdot |{\mathbf{z}})  \big|  {\mathbf{x}}_t = {\mathbf{x}}\big].
$$

**视角 1：为何 $(\alpha_t,\sigma_t)=(1-t,t)$ 是一种自然的调度。** 记 $\sigma_t:=\zeta(t)$ 且 $\alpha_t:=1-\zeta(t)$，其中 $\zeta(t)$ 随时间变化，则条件速度变为

$$
{\mathbf{v}}_t({\mathbf{x}}_t |{\mathbf{z}})
=\zeta'(t)({\boldsymbol{\epsilon}}-{\mathbf{x}}_0),
\quad\text{where } {\mathbf{z}}=({\mathbf{x}}_0,{\boldsymbol{\epsilon}}).
$$

**单位尺度的回归目标。** 对于标准调度 $\zeta(t)=t$，条件速度 ${\mathbf{v}}_t(\cdot |{\mathbf{z}})$ 满足

<span id="eq-unit-velocity"></span>

$$
\begin{aligned}
\mathbb{E}\left[\|{\mathbf{v}}_t(\cdot |{\mathbf{z}})\|_2^2\right]
={\mathbb{E}}_{{\boldsymbol{\epsilon}}}\left\lVert{\boldsymbol{\epsilon}}\right\rVert_2^2+{\mathbb{E}}_{{\mathbf{x}}_0}\left\lVert{\mathbf{x}}_0\right\rVert^2_2 = D + \underbrace{\operatorname{Tr}\operatorname{Cov}[{\mathbf{x}}_0]}_{\text{total variance}}+\underbrace{\left\lVert{\mathbb{E}}{\mathbf{x}}_0\right\rVert^2_2}_{\text{mean}}.
\end{aligned}
\tag{6.3.7}
$$

因此，目标的期望大小不随 $t$ 变化。将数据标准化为零均值和单位协方差（即 $\operatorname{Cov}[{\mathbf{x}}_0]={\mathbf{I}}$）后，两个分量 $\alpha_t'{\mathbf{x}}_0$ 和 $\sigma_t'{\boldsymbol{\epsilon}}$ 在所有 $t$ 下的贡献相当，从而避免端点附近的梯度爆炸或消失。为说明这一点，我们考虑扩散模型的训练目标：

$$
\mathcal{L}_{\mathrm{velocity}}({\boldsymbol{\phi}})
= {\mathbb{E}}_{t}  {\mathbb{E}}_{{\mathbf{x}}_0,{\boldsymbol{\epsilon}}}\left[\big\|{\mathbf{v}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t) - {\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})\big\|_2^2
\right].
$$

应用链式法则可得，该损失关于模型参数 ${\boldsymbol{\phi}}$ 的梯度为

$$
\nabla_{{\boldsymbol{\phi}}}\mathcal{L}_{\mathrm{velocity}}({\boldsymbol{\phi}})
= 2  {\mathbb{E}}_{t}  {\mathbb{E}}_{{\mathbf{x}}_0,{\boldsymbol{\epsilon}}} \left[
  \partial_{{\boldsymbol{\phi}}}{\mathbf{v}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)^{\top} \left({\mathbf{v}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t) - {\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})\right)
\right].
$$

因此，目标的尺度 $\|{\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{z}})\|_2$ 会影响梯度的稳定性：在其他条件相同的情况下，若它衰减到 $0$（或发散），那么在相应的 $t$ 处，梯度就倾向于消失（或爆炸）。采用标准选择 $\zeta(t)=t$ 时，[式（6.3.7）](03-equivalent-parameterizations.md#eq-unit-velocity)给出了不依赖于 $t$ 的目标大小，因此，回归信号不会导致端点（$t=0$ 或 $t=1$）处的衰减或发散（假设 ${\mathbb{E}}\|\partial_{{\boldsymbol{\phi}}}{\mathbf{v}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)\|^2$ 以及所有时间权重均受到控制）。

**标准调度与 ${\mathbf{v}}$ 预测的相互作用。** 对于仿射路径 ${\mathbf{x}}_t=\alpha_t {\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}$，oracle 速度可以分解为

$$
{\mathbf{v}}^*({\mathbf{x}}, t)=\alpha_t' {\mathbf{x}}^*({\mathbf{x}},t)+\sigma_t' {\boldsymbol{\epsilon}}^*({\mathbf{x}},t),
$$

其中 ${\mathbf{x}}^*={\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_t={\mathbf{x}}]$ 且 ${\boldsymbol{\epsilon}}^*={\mathbb{E}}[{\boldsymbol{\epsilon}}|{\mathbf{x}}_t={\mathbf{x}}]$。固定 ${\mathbf{x}}$ 求导可得

$$
\partial_t {\mathbf{v}}^*
=\underbrace{\alpha_t'' {\mathbf{x}}^*+\sigma_t'' {\boldsymbol{\epsilon}}^*}_{\text{schedule curvature}}
+ \alpha_t' \partial_t {\mathbf{x}}^*+\sigma_t' \partial_t {\boldsymbol{\epsilon}}^*.
$$

在线性调度 $\alpha_t=1-t,\ \sigma_t=t$ 下，曲率项消失（$\alpha_t''=\sigma_t''=0$），因此 ${\mathbf{v}}^*$ 的时间变化主要反映后验的演变 $(\partial_t {\mathbf{x}}^*,\partial_t {\boldsymbol{\epsilon}}^*)$，而非调度本身。这一效果在 ${\mathbf{v}}$ 预测下尤其清楚：系数 $\alpha_t',\sigma_t'$ 为常数（$-1$ 和 $+1$），避免了在漂移中额外引入依赖于 $t$ 的缩放。相比之下，score、${\mathbf{x}}_0$ 或 ${\boldsymbol{\epsilon}}$ 参数化通常会引入 $\sigma_t'/\sigma_t$ 或 $\alpha_t'/\alpha_t$ 这样的比值，即使采用线性调度，这些比值也可能在端点附近剧烈变化。因此，尽管这些性质在原则上并非其独有，线性 $(1-t,t)$ 调度与 $v$ 预测相结合，仍为 oracle 速度提供了尤为稳定且清晰的时间依赖关系。

**最小化条件能量。** 接下来，我们采用更偏理论的最优传输视角（见[第 7 章](../07-optimal-transport/index.md#ch-ot-eot)）。这里，*条件动能*量化了条件速度沿前向路径的总期望运动量，即从 ${\mathbf{x}}_0$ 移动到 ${\boldsymbol{\epsilon}}$ 所需的瞬时运动量（或运动耗费）：

$$
\mathcal K[\zeta]
:=\int_0^1 {\mathbb{E}}_{{\mathbf{x}}_0,{\boldsymbol{\epsilon}}}\!\big[\|{\mathbf{v}}_t(\cdot |{\mathbf{z}})\|_2^2\big]\mathop{}\!\mathrm{d} t
=\Big(D+\operatorname{Tr}\operatorname{Cov}[{\mathbf{x}}_0]+\|{\mathbb{E}}{\mathbf{x}}_0\|_2^2\Big)
\int_0^1 \big(\zeta'(t)\big)^2\mathop{}\!\mathrm{d} t.
$$

因此，最小化 $\mathcal K[\zeta]$ 对应于在以下仿射插值族中，寻找期望意义下最平滑、能量最低的路径

$$
{\mathbf{x}}_t=(1-\zeta(t)){\mathbf{x}}_0+\zeta(t){\boldsymbol{\epsilon}}.
$$

在边界条件 $\zeta(0)=0$ 和 $\zeta(1)=1$ 下，Euler–Lagrange 方程 $\zeta''(t)=0$ 给出最小化解 $\zeta(t)=t$，对应一条直线条件路径。这意味着，在该族的所有光滑调度 $\zeta$ 中，标准流 $\zeta(t)=t$ 是从 ${\mathbf{x}}_0$ 移动到 ${\boldsymbol{\epsilon}}$ 最节省能量的方式。我们将在[命题 7.5.1](../07-optimal-transport/05-probability-flow-odes-and-optimal-transport.md#ot-conditional-flow)中再次讨论这一点，并给出更详细的处理。

**关于 oracle 速度能量的说明。** 线性调度 $\zeta(t)=t$ 总能使上面推导的*条件*速度能量最小，但不一定能使边缘化后的 oracle 速度 ${\mathbf{v}}^*$ 的能量最小。原因在于，${\mathbf{v}}^*$ 的大小可能沿概率路径变化，因此，使能量最小的调度可能以不同速度通过路径的不同部分。

为明确说明这种依赖关系，令

$$
{\mathbf{y}}_u=(1-u){\mathbf{x}}_0+u{\boldsymbol{\epsilon}},
\qquad u\in[0,1],
$$

其中 $u$ 表示沿仿射概率路径的位置，并定义

$$
\mathbf{b}_u({\mathbf{y}})
:=
{\mathbb{E}}\!\left[{\boldsymbol{\epsilon}}-{\mathbf{x}}_0|{\mathbf{y}}_u={\mathbf{y}}\right],
\qquad
\kappa(u)
:=
{\mathbb{E}}_{{\mathbf{y}}_u}
\left[
\|\mathbf{b}_u({\mathbf{y}}_u)\|_2^2
\right].
$$

这里，$\kappa(u)$ 衡量沿路径每移动一个单位时，oracle 速度大小平方的期望。

调度 $u=\zeta(t)$ 指定了通过这条路径的快慢。由于 ${\mathbf{x}}_t={\mathbf{y}}_{\zeta(t)}$，oracle 速度变为

$$
{\mathbf{v}}^*({\mathbf{x}},t)
=
\zeta'(t)\mathbf{b}_{\zeta(t)}({\mathbf{x}}),
$$

其动能为

$$
\mathcal{K}_{\mathrm{marg}}[\zeta]
=
\int_0^1
\kappa(\zeta(t))
\bigl(\zeta'(t)\bigr)^2
\mathop{}\!\mathrm{d} t.
$$

假设 $\kappa(u)>0$ 并满足充分的正则性条件，最小化这一能量可得

$$
\kappa(\zeta(t))
\bigl(\zeta'(t)\bigr)^2
=
\text{constant},
\qquad\text{or equivalently}\qquad
\zeta'(t)
\propto
\frac{1}{\sqrt{\kappa(\zeta(t))}}.
$$

因此，使能量最小的调度会在 oracle 速度的期望大小较大处移动得更慢，在其大小较小处移动得更快。

由此，$\zeta(t)=t$ 只有在 $\kappa(u)$ 沿路径保持常数时，才能使边缘速度能量最小。这凸显了它与上述条件速度结果的区别：对于后者，线性调度始终是仿射族内恒速且能量最小的选择。

**视角 2：为何速度预测可以视为采样的自然选择。**

利用 PF-ODE 采样最终需要其速度场。因此，${\mathbf{v}}$ 预测尤为直接：网络预测的量就是 ODE 右端出现的量。相比之下，${\mathbf{x}}$、${\boldsymbol{\epsilon}}$ 和 ${\mathbf{s}}$ 预测通过当前状态与网络预测的组合来表示同一个 oracle 速度。

在 oracle 层面，根据[式（6.3.2）](03-equivalent-parameterizations.md#eq-clean-pf-ode)，这些是同一个 ${\mathbf{v}}^*({\mathbf{x}},t)$ 可以精确相互转换的形式；因此，底层的 oracle PF-ODE 不会因参数化选择而改变。这里，我们重点考察学习得到的 PF-ODE 在采样时的表达方式；用于训练相应预测的损失可以另行选择。

**${\mathbf{x}}$、${\boldsymbol{\epsilon}}$ 和 ${\mathbf{s}}$ 预测下的 PF-ODE。** [式（6.3.2）](03-equivalent-parameterizations.md#eq-clean-pf-ode)中的前三个恒等式具有共同形式

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
\underbrace{L(t){\mathbf{x}}(t)}_{\text{known linear term}}
+
\underbrace{{\mathbf{N}}_{\boldsymbol{\phi}^{\times}}({\mathbf{x}}(t),t)}_{\text{network-dependent term}},
$$

其中，$L(t)$ 表示由噪声调度确定的相应已知标量系数，${\mathbf{N}}_{\boldsymbol{\phi}^{\times}}$ 包含干净数据、噪声或 score 预测与相应时间相关系数的乘积。例如，在 ${\boldsymbol{\epsilon}}$ 预测下，

$$
L(t)
=
\frac{\alpha_t'}{\alpha_t},
\qquad
{\mathbf{N}}_{\boldsymbol{\phi}^{\times}}({\mathbf{x}},t)
=
-\sigma_t
\left(
\frac{\alpha_t'}{\alpha_t}
-
\frac{\sigma_t'}{\sigma_t}
\right)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^{\times}}({\mathbf{x}},t).
$$

这种形式便于数值积分，因为已知线性项可以解析处理，而依赖网络的项则通过数值方法近似。基于指数积分器的扩散求解器，例如 DPM-Solver（[Lu et al., 2022](../../references.md#ref-lu2022dpm)），正是利用了这一结构。我们将在[第 9 章](../09-numerical-solvers/index.md#ch-numerical-solvers)详细阐述这一求解器视角。

**${\mathbf{v}}$ 预测下的 PF-ODE。** 采用 ${\mathbf{v}}$ 预测时，PF-ODE 具有以下直接而简单的形式

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
{\mathbf{v}}_{\boldsymbol{\phi}^{\times}}({\mathbf{x}}(t),t)
\approx
{\mathbf{v}}^*({\mathbf{x}}(t),t).
$$

因此，一次网络求值就能直接提供标准 ODE 求解器所需的量，无需先将预测与单独的线性项组合。

标准线性调度使这一联系尤其清晰。对于

$$
{\mathbf{x}}_t=(1-t){\mathbf{x}}_0+t{\boldsymbol{\epsilon}},
$$

一次速度预测就能确定相应的端点预测

$$
\widehat{{\mathbf{x}}}_0
=
{\mathbf{x}}_t-t{\mathbf{v}}_{\boldsymbol{\phi}^{\times}}({\mathbf{x}}_t,t),
\qquad
\widehat{{\boldsymbol{\epsilon}}}
=
{\mathbf{x}}_t+(1-t){\mathbf{v}}_{\boldsymbol{\phi}^{\times}}({\mathbf{x}}_t,t).
$$

利用这一次预测从 $t$ 移动到 $s$，得到

$$
(1-s)\widehat{{\mathbf{x}}}_0+s\widehat{{\boldsymbol{\epsilon}}}
=
{\mathbf{x}}_t
+
(s-t){\mathbf{v}}_{\boldsymbol{\phi}^{\times}}({\mathbf{x}}_t,t),
$$

这恰好是标准线性调度下，对学习得到的速度场所作的普通 Euler 更新。因此，${\mathbf{v}}$ 预测与线性调度相结合，在模型预测与 PF-ODE 的数值积分之间建立了尤为直接的联系。关于数值求解器和更一般调度的详细讨论，将留到[第 9 章](../09-numerical-solvers/index.md#ch-numerical-solvers)。

综合来看，${\mathbf{v}}$ 预测直接提供 PF-ODE 所需的速度，而其他预测类型通过依赖于调度的变换来表示同样的 oracle 动力学。这种直接对应关系，是将速度预测视为采样的自然选择的主要依据。

**结论。** 尽管 ${\mathbf{v}}$ 预测与标准线性调度相结合具有某些理论优势，例如目标大小恒定且不存在调度曲率，但这些性质不一定使它在所有情况下都更优。在实践中，模型性能取决于一系列相互作用的因素，包括网络架构、归一化方案、随时间变化的损失权重、采样器与离散化步数的选择、引导强度、正则化策略、数据缩放以及总体训练预算。不同的数据集和目标可能更适合其他参数化或调度，最优配置最终是一个经验问题，应通过验证和消融研究来确定。

# 6.4 底层原理：Fokker–Planck 方程 {#sec-comparison-connection}

<span id="fig-unified-framework"></span>

![连续性方程与 Fokker–Planck 方程连接变分、SDE 和 ODE 描述](../../../en/assets/chapter-06/unified-framework.png)

**图 6.2：** **通过连续性方程连接变分、SDE 与 ODE 形式的统一视角，所有 $p_t({\mathbf{x}})$ 都在共同的动力学下演化。** 速度场 ${\mathbf{v}}^*({\mathbf{x}}, t) = f(t){\mathbf{x}} - \frac{1}{2}g^2(t){\mathbf{s}}^*({\mathbf{x}}, t)$ 由 score 函数 ${\mathbf{s}}^*({\mathbf{x}}, t) := \nabla_{\mathbf{x}} \log p_t({\mathbf{x}})$ 决定。系数 $f(t)$、$g(t)$、$\sigma_t$ 与 $\alpha_t$ 是预定义的时间函数，而 $\gamma(t)$ 是可调的时变超参数。

来源：作者绘制。

本节说明，扩散模型的三个主要视角——变分、基于 score 与基于流——都源于同一统一原理：连续性（Fokker–Planck）方程，它支配所选前向过程下的密度演化。

首先回顾，[第 4.6 节](../04-score-sde-framework/06-fokker-planck-and-reverse-sde.md#eq-heuristic-fpe-sde)的分析，将基于离散核与 Bayes 法则的变分视角，与基于连续动力学的 score SDE 视角统一起来。我们通过说明变分模型是底层前向和逆向 SDE 的相容离散化，建立这一联系。具体而言，通过离散核逐步计算的边缘密度，其演化方式与支配连续时间动力学的 Fokker–Planck 方程一致。这确认了两个视角在基础层面的等价性。

接下来连接基于流与基于 score 的视角。[第 6.4.1 节](04-fokker-planck-connections.md#subsec-flow-score)说明，ODE 流确定一条密度路径，其边缘分布总可以由一族随机过程实现。这将确定性流与随机 SDE 纳入同一族中。

这些结果共同将三个视角统一在同一框架下（见[图 6.2](04-fokker-planck-connections.md#fig-unified-framework)）。最后，[第 6.5 节](05-closing-remarks.md#subsec-takeaway-dm-gaussianfm)总结本章。

## 6.4.1 基于流的方法与 Score SDE 的联系 {#subsec-flow-score}

扩散模型的一个显著特点是，不同的动力系统，无论确定性还是随机性，都可以沿着相同的概率分布演化路径运行。本节揭示[第 5.2 节](../05-flow-based-perspective/02-flow-matching-framework.md#sec-flow-matching-framework)中基于 ODE 的流与 Score SDE 之间自然、简洁的联系。具体而言，定义生成 ODE 的速度场，可以转化为遵循相同 Fokker–Planck 动力学的随机对应形式，为确定性插值与随机采样建立具有理论依据的桥梁。由此得到从 ODE 到 SDE 的连续模型族，生成同一条数据分布路径。

考虑连续时间设定，其扰动核为

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0) = \mathcal{N}({\mathbf{x}}_t; \alpha_t {\mathbf{x}}_0, \sigma_t^2 {\mathbf{I}}),
$$

其中 ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$。与通常一样，该条件分布诱导边缘密度路径 $p_t({\mathbf{x}}_t) = \mathbb{E}_{{\mathbf{x}}_0 \sim p_{\mathrm{data}}}[p({\mathbf{x}}_t|{\mathbf{x}}_0)]$，满足 $p_T\approx p_{\mathrm{prior}}$。

为匹配这一密度路径，考虑 ODE

<span id="eq-ode-velocity"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = \mathbf{v}_t({\mathbf{x}}(t)), \quad t \in [0,T],
\end{aligned}
\tag{6.4.1}
$$

其中 $\mathbf{v}_t({\mathbf{x}}) ={\mathbb{E}}\left[\alpha_t'{\mathbf{x}}_0+\sigma_t'{\boldsymbol{\epsilon}}|{\mathbf{x}}\right]$ 是[式 (5.2.9)](../05-flow-based-perspective/02-flow-matching-framework.md#eq-marginal-oracle-velocity)所示的 oracle 速度（注意时间已翻转，以遵循扩散模型约定）。从 ${\mathbf{x}}(T)\sim p_{\mathrm{prior}}$ 反向积分[式 (6.4.1)](04-fokker-planck-connections.md#eq-ode-velocity)，得到来自 $p_0$ 的样本。

虽然这一 ODE 已足以生成高质量样本，但加入随机性可能改善样本多样性。这引出以下问题：

<span id="question-6-4-1"></span>

**问题 6.4.1**

是否存在一个 SDE，从 $p_{\mathrm{prior}}$ 出发，其动力学产生与[式 (6.4.1)](04-fokker-planck-connections.md#eq-ode-velocity)中的 ODE 相同的边缘密度？

下面的结果肯定了这一点：存在一族逆时间 SDE，诱导与对应 PF-ODE 相同的边缘密度路径。这些 SDE 诱导的密度满足这条路径的同一 Fokker–Planck 方程，因此它们的单时刻边缘分布与指定插值 $\{p_t\}_{t\in[0,T]}$ 一致[^1]。

<span id="rev-sde-interpolant"></span>

**命题 6.4.1：逆时间 SDE 生成与插值相同的边缘分布**

令 $\gamma(t) \geq 0$ 为任意随时间变化的系数。考虑以下逆时间 SDE，使用原始时间变量 $t$ 表示，并从 $T$ 反向积分到 $0$：

<span id="eq-general-reverse-sde"></span>

$$
\begin{aligned}
\mathop{}\!\mathrm{d} \bar{\mathbf{x}}(t)
    =
    \Big[
        {\mathbf{v}}^*(\bar{\mathbf{x}}(t), t)
        - \tfrac{1}{2}\gamma^2(t){\mathbf{s}}^*(\bar{\mathbf{x}}(t), t)
    \Big]\mathop{}\!\mathrm{d} t
    + \gamma(t)\mathop{}\!\mathrm{d}\bar{{\mathbf{w}}}(t),
    \qquad t:T\to 0 .
\end{aligned}
\tag{6.4.2}
$$

终端条件为 $\bar{\mathbf{x}}(T)\sim p_T$。则过程 $\{\bar{\mathbf{x}}(t)\}_{t\in[0,T]}$ 匹配 ODE 密度路径诱导的指定边缘分布 $\{p_t\}_{t\in[0,T]}$。这里，${\mathbf{s}}^*({\mathbf{x}},t):=\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$ 是 score 函数，它与速度场 ${\mathbf{v}}^*({\mathbf{x}},t)$ 的关系为

<span id="eq-score-velocity"></span>

$$
\begin{aligned}
{\mathbf{v}}^*({\mathbf{x}}, t)
    =
    f(t){\mathbf{x}}
    -
    \frac{1}{2}g^2(t){\mathbf{s}}^*({\mathbf{x}},t),
    \quad
    {\mathbf{s}}^*({\mathbf{x}},t)
    =
    \frac{1}{\sigma_t}
    \frac{\alpha_t{\mathbf{v}}^*({\mathbf{x}},t)-\alpha_t'{\mathbf{x}}}
    {\alpha_t'\sigma_t-\alpha_t\sigma_t'} .
\end{aligned}
\tag{6.4.3}
$$

**命题证明。**

当过程反向积分时，相对于原始时间变量 $t$ 写出的逆时间 Fokker–Planck 方程为

$$
\partial_t p_t
=
-\nabla\cdot
\Big(
\big[
{\mathbf{v}}^*
-
\tfrac{1}{2}\gamma^2{\mathbf{s}}^*
\big]p_t
\Big)
-
\tfrac{1}{2}\gamma^2\Delta p_t .
$$

利用恒等式 $\nabla\cdot({\mathbf{s}}^*p_t)=\Delta p_t$（因为 ${\mathbf{s}}^*=\nabla\log p_t$），二阶项抵消，得到

$$
\partial_t p_t
=
-\nabla\cdot({\mathbf{v}}^*p_t),
$$

即对应 PF-ODE 的一阶连续性方程。因此，逆时间 SDE 与 ODE 诱导相同的边缘密度路径 $\{p_t\}$。见 [Ma et al. (2024)](../../references.md#ref-ma2024sit) 的附录 A.2—A.3。

在[命题 6.4.1](04-fokker-planck-connections.md#rev-sde-interpolant)的 oracle 层面，可以任意选择 $\gamma(t)$ 而不改变指定边缘密度路径：依赖 $\gamma$ 的漂移项与扩散项在 Fokker–Planck 方程中精确抵消。代表性选择包括：

-   设 $\gamma(t) = 0$，恢复[式 (6.4.1)](04-fokker-planck-connections.md#eq-ode-velocity)中的 ODE。

-   当 $\gamma(t) = g(t)$ 时，[式 (6.4.2)](04-fokker-planck-connections.md#eq-general-reverse-sde)成为[式 (4.1.6)](../04-score-sde-framework/01-score-sde-principles.md#eq-sde-backward)中的逆时间 SDE，因为 oracle 速度 ${\mathbf{v}}({\mathbf{x}}, t)$ 满足（见[命题 6.3.1](03-equivalent-parameterizations.md#equi-para)）：

    $$
    {\mathbf{v}}^*({\mathbf{x}}, t) = f(t){\mathbf{x}} - \tfrac{1}{2}g^2(t){\mathbf{s}}^*({\mathbf{x}}, t).
    $$

-   也有研究探索 $\gamma(t)$ 的其他选择；例如，[Ma et al. (2024)](../../references.md#ref-ma2024sit) 选择 $\gamma(t)$，以最小化 $p_{\mathrm{data}}$ 与 $t=0$ 时密度之间的 KL 差距，后者通过从 $t=T$ 求解[式 (6.4.2)](04-fokker-planck-connections.md#eq-general-reverse-sde)获得。

沿用 Score SDE 的做法，训练好的速度场 ${\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}, t)$ 可以通过[式 (6.4.3)](04-fokker-planck-connections.md#eq-score-velocity)转换为参数化 score 函数 ${\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}, t)$。将其代入[式 (6.4.2)](04-fokker-planck-connections.md#eq-general-reverse-sde)，定义*经验逆时间 SDE*，从 $t=T$ 出发、以 $\bar{\mathbf{x}}(T) \sim p_{\mathrm{prior}}$ 为初始条件进行数值积分，即可采样。

这一命题展示扩散模型的显著灵活性：一旦前向过程诱导出*边缘密度路径* $\{p_t\}_{t\in[0,T]}$，就可以由一整族逆时间动力学实现，包括 PF-ODE 与以下逆时间 SDE：

$$
\mathop{}\!\mathrm{d}\bar{\mathbf{x}}(t)
=
\big[
{\mathbf{v}}^*(\bar{\mathbf{x}},t)
-\tfrac12\gamma^2(t){\mathbf{s}}^*(\bar{\mathbf{x}},t)
\big]\mathop{}\!\mathrm{d} t
+
\gamma(t)\mathop{}\!\mathrm{d}\bar{\mathbf{w}}(t),
\quad
\gamma(t)\geq0,
$$

它们从 $t=T$ 演化到 $t=0$。虽然 $\gamma(t)$ 改变单条轨迹的随机动力学，但对于指定密度路径，相应的 Fokker–Planck 方程都化为同一个支配密度的方程：

$$
\partial_t p_t
=
-\nabla_{{\mathbf{x}}}\cdot
\bigl({\mathbf{v}}^*({\mathbf{x}},t)p_t({\mathbf{x}})\bigr).
$$

事实上，利用 ${\mathbf{s}}^*({\mathbf{x}},t)=\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$，额外的、依赖 $\gamma$ 的漂移项与扩散项在密度层面抵消。因此，$\gamma(t)$ 控制样本轨迹的随机性，同时保留相同的 oracle 单时刻边缘分布，将确定性 PF-ODE 与一族随机 SDE 对应形式连接起来，如[图 6.2](04-fokker-planck-connections.md#fig-unified-framework)所示。

[^1]: 为完整起见，前向 SDE 表示（这里不需要使用）为

    $$
    \mathop{}\!\mathrm{d} {\mathbf{x}}(t)=f(t) {\mathbf{x}}(t) \mathop{}\!\mathrm{d} t+g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}(t),
    $$

    其中 $f(t)$ 与 $g(t)$ 通过[式 (4.5.2)](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#eq-kernel-sde-equiv)与 $(\alpha_t,\sigma_t)$ 联系。
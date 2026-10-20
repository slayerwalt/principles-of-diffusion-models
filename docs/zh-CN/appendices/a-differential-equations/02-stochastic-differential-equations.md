# A.2 随机微分方程基础 {#app-sde-intro}

随机微分方程（SDE）在常微分方程（ODE）的基础上加入随机性，为受不确定性影响的系统提供数学建模框架。本章从 ODE 的离散化出发，扩展到 SDE 的离散化，最后讨论一般 SDE，包括 Itô 微积分和 Itô 公式。

## A.2.1 从 ODE 到 SDE：直观介绍 {#from-odes-to-sdes}

先考虑一个描述状态变量 ${\mathbf{x}}(t) \in \mathbb{R}^D$ 确定性演化的 ODE：

<span id="eq-ode"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t), \quad {\mathbf{x}}(0) = {\mathbf{x}}_0.
\tag{A.2.1}
$$

这里，${\mathbf{f}}: \mathbb{R}^D \times [0,T] \rightarrow \mathbb{R}^D$ 是随时间变化的速度场，支配 ${\mathbf{x}}(t)$ 的动力学。该 ODE 的解是光滑轨迹 $t \mapsto {\mathbf{x}}(t)$，完全由初始条件 ${\mathbf{x}}_0$ 决定。

**离散化视角。** 为建立直观认识，考虑以小时间步 $\Delta t$ 对[公式 (A.2.1)](02-stochastic-differential-equations.md#eq-ode)进行 Euler 离散化：

$$
{\mathbf{x}}_{t+\Delta t} = {\mathbf{x}}_t + {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t.
$$

随着 $\Delta t \to 0$，这一近似更加准确，并在关于 ${\mathbf{f}}$ 的标准正则性条件下收敛到 ODE 的精确解。

**引入随机性：从 ODE 到 SDE。** 在许多现实系统中，完全了解动力学并不现实。噪声、不确定性或未建模的相互作用都可能影响演化。为纳入这种随机性，在 ODE 中加入随机项：

<span id="eq-euler-sde"></span>

$$
{\mathbf{x}}_{t+\Delta t} = {\mathbf{x}}_t + {\mathbf{f}}({\mathbf{x}}_t, t)\Delta t + g(t)\sqrt{\Delta t} \cdot \boldsymbol{\epsilon}_t,
\tag{A.2.2}
$$

其中

-   $g: [0,T] \to \mathbb{R}$ 是扩散系数（可能同时依赖状态和时间，不过这里假设只依赖时间）；

-   $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}_D)$ 是独立同分布的标准 Gaussian 向量。

这一修改后的更新规则既反映确定性漂移，也反映按 $\sqrt{\Delta t}$ 缩放的随机扰动。这种缩放确保随机扰动在极限 $\Delta t \to 0$ 下保持有限。该表述在 $\Delta t \to 0$ 时产生一个*连续时间随机过程*，由此引出 SDE 框架。

**随机微分方程。** 形式上，离散更新[公式 (A.2.2)](02-stochastic-differential-equations.md#eq-euler-sde)在 $\Delta t \to 0$ 时的极限定义 SDE：

<span id="eq-sde"></span>

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}(t) = {\mathbf{f}}({\mathbf{x}}(t), t)\mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}(t).
\tag{A.2.3}
$$

这里，${\mathbf{w}}(t) \in \mathbb{R}^D$ 是 *Wiener 过程*（标准 Brownian 运动），一个具有以下性质的连续时间随机过程：

-   **初始状态：** ${\mathbf{w}}(0) = 0$ 几乎必然成立；

-   **独立增量：** 对于 $0 \leq s < t$，增量 ${\mathbf{w}}(t) - {\mathbf{w}}(s)$ 与过去独立；

-   **Gaussian 增量：**

    <span id="eq-wiener-gaussian"></span>

    $$
    \begin{aligned}
    {\mathbf{w}}(t) - {\mathbf{w}}(s) \sim \mathcal{N}\bigl(\boldsymbol{0}, (t - s){\mathbf{I}}_D \bigr)
    \end{aligned}
    \tag{A.2.4}
    $$

-   **连续性：** 样本路径 $t \mapsto {\mathbf{w}}(t)$ 几乎必然连续，但处处不可微。

此外，常使用记号

$$
\mathop{}\!\mathrm{d} {\mathbf{w}}(t) := {\mathbf{w}}(t + \mathop{}\!\mathrm{d} t) - {\mathbf{w}}(t)
$$

表示 Wiener 过程的无穷小增量。

虽然这一记号具有启发性，但它只是直观记号，不应理解为经典意义（例如 Riemann 或 Lebesgue 意义）的微分，因为 Brownian 路径几乎必然处处不可微。它是表示 Gaussian 增量性质的形式化简写：

$$
\mathop{}\!\mathrm{d} {\mathbf{w}}(t) \sim \mathcal{N}(0, \mathop{}\!\mathrm{d} t\, \mathrm{I}_D),
$$

即在长度为 $\mathop{}\!\mathrm{d} t$ 的无穷小时间区间内，Wiener 过程的增量表现为均值为零、协方差为 $\mathop{}\!\mathrm{d} t\, \mathrm{I}_D$ 的 Gaussian 随机变量。

## A.2.2 对[公式 (A.2.3)](02-stochastic-differential-equations.md#eq-sde)的进一步解释 {#further-explanation-of-sde}

[公式 (A.2.3)](02-stochastic-differential-equations.md#eq-sde)中的 SDE 应理解为其*积分形式*：

<span id="eq-sde-integral-form"></span>

$$
\begin{aligned}
{\mathbf{x}}(t) = {\mathbf{x}}(0) + \int_0^t {\mathbf{f}}({\mathbf{x}}(s), s) \mathop{}\!\mathrm{d} s + \int_0^t g(s) \mathop{}\!\mathrm{d} {\mathbf{w}}(s),
\end{aligned}
\tag{A.2.5}
$$

并在 *Itô 意义*下解释。这里，第一项是经典（Riemann 或 Lebesgue）积分，表示累积的确定性漂移；第二项是 *Itô 随机积分*，相对于 Wiener 过程 ${\mathbf{w}}(t)$ 积分。这里不完整给出 Itô 积分的严格构造，而提供以下直观理解。

**Itô 积分的直观理解。** Itô 积分可理解为以下形式的离散和的极限[^1]：

$$
\sum_i g(t_i)\bigl(w(t_{i+1})-w(t_i)\bigr),
$$

其中被积函数 $g(t)$ 在每个子区间的*左*端点 $t_i$ 求值。随着划分变细，这些和收敛到 Itô 积分。左端点求值规则使 Itô 积分区别于可以使用中点或其他求值规则的经典积分。由于 Brownian 路径连续却几乎必然处处不可微，经典积分无法直接应用。Itô 积分处理了这一不规则性，刻画随机波动随时间累积的作用。

**微分记号的使用。** $\mathop{}\!\mathrm{d} {\mathbf{x}}(t)$、$\mathop{}\!\mathrm{d} t$ 和 $\mathop{}\!\mathrm{d} {\mathbf{w}}(t)$ 等表达式不是经典微分，而是表示相应过程无穷小增量的形式记号。尽管带有启发性质，它们因便于仿照 ODE 表示 SDE 而被广泛使用，也便于在 Itô 微积分中作形式运算。

[附录 C](../c-ito-calculus/index.md#app-ito)将说明 Itô 微积分如何应用于扩散模型。

**与 ODE 的比较。** 在 ODE 中，例如

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t),
$$

其积分形式

$$
{\mathbf{x}}(t) = {\mathbf{x}}(0) + \int_0^t {\mathbf{f}}({\mathbf{x}}(\tau), \tau) \mathop{}\!\mathrm{d} \tau
$$

由*微积分基本定理*保证，该定理确保可微函数可以由其导数恢复。

相比之下，[公式 (A.2.3)](02-stochastic-differential-equations.md#eq-sde)等 SDE 没有这一定理的直接对应形式，因为 Brownian 运动不具可微性，随机积分也不遵循经典链式法则。Itô 微积分引入其他工具（例如 Itô 引理）来分析和处理随机动力学。

因此，尽管 SDE 的微分记号紧凑且直观，严格理解仍需要通过使用 Itô 积分的积分表述来解释。

## A.2.3 SDE 的数值求解器 {#a-numerical-solver-for-sde}

与 ODE 类似，如果 $\mathbf{f}(\cdot, t)$ 和 $g(\cdot)$ 满足一定光滑性条件，[公式 (A.2.3)](02-stochastic-differential-equations.md#eq-sde)中的 SDE 也有唯一解[^2]：$\mathbf{f}(\cdot, t)$ 关于 $\mathbf{x}$ 满足 Lipschitz 条件和线性增长条件，$g(\cdot)$ 平方可积。

对于[公式 (A.2.3)](02-stochastic-differential-equations.md#eq-sde)这样的一般 SDE，通常没有闭式解，因此需要数值方法。常用方法是 *Euler–Maruyama 方法*，它推广了 ODE 的 Euler 方法，实际上已经在[公式 (A.2.2)](02-stochastic-differential-equations.md#eq-euler-sde)中出现。它近似漂移项 ${\mathbf{f}}({\mathbf{x}}(t), t)$ 在时间步 $\Delta t$ 上的作用，并模拟随机噪声 $g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}(t)$，使用的 Gaussian 增量为 $\sqrt{\Delta t} \, \boldsymbol{\epsilon}_t$，其中 $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

后文[第 C.1.5 节](../c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-linear-sde)将说明，线性 SDE 具有闭式解。

[^1]: 对于平方可积的被积函数，这种收敛在 $L^2$（均方）意义下成立，进而蕴含依概率收敛。$L^2$ 视角是自然的，因为它直接联系到 Itô 等距性质，这是 Itô 积分标准严格构造背后的关键恒等式。该等距性质在期望意义下将随机积分与普通积分关联起来（${\mathbb{E}}\bigl[\bigl\|\int_0^T \boldsymbol{\psi}(t)\,\mathrm{d}{\mathbf{w}}_t \bigr\|^2\bigr] = {\mathbb{E}}\bigl[\int_0^T \|\boldsymbol{\psi}(t)\|_F^2\,\mathrm{d}t\bigr]$，其中使用 Frobenius 范数 $\|\boldsymbol{\psi}(t)\|_F^2$）；我们将在[第 C.1.5 节](../c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-linear-sde)陈述并使用它。

[^2]: 这里的解是*强*解，即 $\mathbf{x}(t)$ 满足 SDE 的积分形式（见[公式 (A.2.5)](02-stochastic-differential-equations.md#eq-sde-integral-form)），相对于固定概率空间上的给定 Brownian 运动 $\mathbf{w}(t)$ 而言。此处省略详细的技术定义。

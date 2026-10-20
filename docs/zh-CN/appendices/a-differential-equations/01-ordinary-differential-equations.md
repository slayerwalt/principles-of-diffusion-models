# A.1 常微分方程基础 {#app-ode-intro}

本节介绍 ODE 的基础理论，重点讨论给定初始条件时解的唯一性，也介绍使用数值求解器求解 ODE 的实用方法。

## A.1.1 常微分方程的直观理解 {#intuition-of-ordinary-differential-equation}

这一确定性过程称为*常微分方程*（ODE）。在多变量情形中，考虑以下形式的系统：

<span id="eq-general-ode"></span>

$$
\frac{\mathop{}\!\mathrm{d} \mathbf{x}(t)}{\mathop{}\!\mathrm{d} t} = \mathbf{v}(\mathbf{x}(t), t),
\tag{A.1.1}
$$

其中 $\mathbf{x}(t) \in \mathbb{R}^D$ 是向量值函数，表示系统在时刻 $t$ 的状态，$\mathbf{v}: \mathbb{R}^D \times \mathbb{R} \to \mathbb{R}^D$ 是向量场，指定每个时空点处变化的方向与大小。

**求解 ODE 的总体直观理解。**

<span id="fig-ode-illustration"></span>

![速度场与数值更新和精确 ODE 轨迹的比较](../../../en/assets/appendix-a/ode-illustration.png)

**图 A.1：** **ODE 示意图。** 速度场 ${\mathbf{v}}({\mathbf{x}},t)$ 为每一点指定一个漂移向量。解轨迹 ${\mathbf{x}}(t)$ 是一条切向量始终与局部漂移一致的路径。左图展示逐步求解器更新（点和箭头）如何近似路径，右图展示与速度场一致流动的精确轨迹（黑色）。如果不指定初始状态 ${\mathbf{x}}(0)$，就有无穷多条轨迹，其瞬时变化都与同一速度场匹配。然而，一旦固定 ${\mathbf{x}}(0)$，ODE 就确定沿漂移流动的唯一路径 ${\mathbf{x}}(t)$。

来源：作者绘制。

为建立直观认识，可以把向量场 $\mathbf{v}(\mathbf{x}, t)$ 想象为一幅随时间变化的箭头图，告诉我们点 $\mathbf{x}$ 在任意给定时刻 $t$ 应如何移动。求解微分方程，就是在这一场中描出曲线 $\mathbf{x}(t)$，使曲线上任一点的切向量（即瞬时速度）都与 $\mathbf{v}(\mathbf{x}(t), t)$ 给出的向量一致。

-   **向量场视角：** 函数 $\mathbf{v}(\mathbf{x}, t)$ 定义对象应如何移动，为运动或变化提供局部“指令”。

-   **轨迹视角：** 解 $\mathbf{x}(t)$ 是粒子在每个瞬间都遵循向量场 $\mathbf{v}$ 所定规则时走过的路径。

因此，求解 ODE 就像把粒子放入流场，观察它随时间去往何处。

## A.1.2 常微分方程的存在性与唯一性 {#existence-and-uniqueness-of-ordinary-differential-equations}

目前我们已经看到，求解 ODE 就是寻找一条在每一点都遵循向量场所给方向的路径。直观上，这像是在跟踪粒子沿速度定义的流运动时形成的轨迹。

但这一图景引出了一个重要问题：

**问题 A.1.1**

选定起点后，能否确定确实存在一条遵循这些方向的路径？如果存在，这条路径是否唯一，还是粒子可能突然跳到另一条轨迹上？

回答这些问题十分必要，因为它告诉我们能否从起始位置可靠地预测系统行为。*存在唯一性定理*给出向量场需要满足的条件，以保证从任意给定初始点出发恰有一条路径。这确保解的行为一致，是 ODE 理论的基石。

**局部（时间上）存在唯一性定理。**

下面给出定理的*局部*版本：对于给定初始条件，解在初始时刻的某个邻域内存在且唯一。

<span id="thm-ode-local"></span>

**定理 A.1.1：局部存在性与唯一性**

设 $\mathbf{v}(\mathbf{x}, t)$ 关于 $\mathbf{x}$ 和 $t$ 在区域 $D \subseteq \mathbb{R}^D \times \mathbb{R}$ 中连续。如果 $\mathbf{v}$ 关于 $\mathbf{x}$ 满足 Lipschitz 条件：

$$
\|\mathbf{v}(\mathbf{x}_1, t) - \mathbf{v}(\mathbf{x}_2, t)\| \leq L \|\mathbf{x}_1 - \mathbf{x}_2\| \quad \forall (\mathbf{x}_1, t), (\mathbf{x}_2, t) \in D,
$$

其中 $L > 0$ 为常数，则对于每个初始条件 $\mathbf{x}(t_0) = \mathbf{x}_0$，[公式 (A.1.1)](01-ordinary-differential-equations.md#eq-general-ode)都存在唯一解 $\mathbf{x}(t)$，定义在某个区间 $[t_0 - \delta, t_0 + \delta]$ 上。

**定理的证明。**

（证明提纲）存在唯一性定理可通过 Picard–Lindelöf 迭代法构造性地证明。该方法生成函数序列 $\{\mathbf{x}_n(t)\}$，使其收敛到解 $\mathbf{x}(t)$。迭代定义为：

$$
\mathbf{x}_{n+1}(t) = \mathbf{x}_0 + \int_{t_0}^t \mathbf{v}(\mathbf{x}_n(s), s) \mathop{}\!\mathrm{d} s.
$$

-   从初始猜测 $\mathbf{x}_0(t) = \mathbf{x}_0$ 开始。

-   利用积分形式迭代改进 $\mathbf{x}_n(t)$。

-   在 Lipschitz 条件下，应用压缩映射定理保证收敛。

证明的本质植根于 Picard–Lindelöf 迭代法，其核心思想也在[第 9.6 节](../../chapters/09-numerical-solvers/06-paradigms.md#sec-picard)中用于加速扩散模型的采样。

**全局（时间上）存在唯一性定理。** 局部存在唯一性定理保证小时间区间上解的存在，而“全局（时间上）存在唯一性定理”在额外正则性条件下，将这一结果扩展到整个区间 $[t_0, T]$。这类结果中的一个著名例子是 *Carathéodory 定理*，它在两个关键假设下保证 ODE 解的全局存在性与唯一性：关于状态变量的局部 Lipschitz 连续性，以及线性增长界。

1.  **关于 $\mathbf{x}$ 的局部 Lipschitz 条件：** 存在函数 $\mathrm{Lip}(t)$，在 $[0, T]$ 上可积，使得对所有 $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^D$，

    $$
    \|{\mathbf{v}}(\mathbf{x}_1, t) - {\mathbf{v}}(\mathbf{x}_2, t)\| \leq \mathrm{Lip}(t) \|\mathbf{x}_1 - \mathbf{x}_2\|.
    $$

2.  **线性增长条件：** 存在函数 $M(t)$，在 $[0, T]$ 上可积，使得对所有 $\mathbf{x} \in \mathbb{R}^D$，

    $$
    \|{\mathbf{v}}(\mathbf{x}, t)\| \leq M(t)(1 + \|\mathbf{x}\|).
    $$

关于定理假设、正式表述及详细证明的完整讨论，参见（[Reid, 1971](../../references.md#ref-reid1971ordinary)）。

**注。**

将这些定理应用于扩散模型的概率流 ODE（见[公式 (4.1.7)](../../chapters/04-score-sde-framework/01-score-sde-principles.md#eq-prob-ode-gt)）时，可能需要对 score 函数 $\nabla_{\mathbf{x}} \log p_t({\mathbf{x}})$ 施加额外假设，例如条件 (i) 和 (ii)。不关注技术细节的读者可以合理接受这些假设，无须进一步论证。

因此，对于由随时间变化的速度场定义的 ODE，给定初始条件后，粒子流的轨迹便唯一确定。

**唯一性意味着解不相交。** 局部存在唯一性定理保证的 ODE 解的唯一性蕴含一条基本性质：从不同初始条件出发的两条不同解轨迹不能相交。这反映了 ODE 的确定性本质，确保每个状态沿唯一路径演化。以下推论将这一结果形式化。

**推论 A.1.1：解不相交**

考虑 ODE 的两个解 ${\mathbf{x}}_1(t)$ 和 ${\mathbf{x}}_2(t)$，

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{v}}\left({\mathbf{x}}(t),t\right), \quad t \in [0,T].
$$

假设它们有不同初值 ${\mathbf{x}}_1(0) \neq {\mathbf{x}}_2(0)$，则这两个解在 $[0,T]$ 上不相交，即

$$
{\mathbf{x}}_1(t) \neq {\mathbf{x}}_2(t) \quad \text{for all } t \in [0,T].
$$

**推论的证明。**

反证，假设存在某个 $t^* \in (0,T]$，使得

$$
\mathbf{x}_1(t^*) = \mathbf{x}_2(t^*).
$$

将两个解首次相遇的时刻定义为

$$
t_0 := \inf \{ t \in [0,T]|\mathbf{x}_1(t) = \mathbf{x}_2(t) \}.
$$

由于 $\mathbf{x}_1(0) \neq \mathbf{x}_2(0)$，且 $t^*$ 属于这一集合，可得 $t_0 > 0$。根据 $\mathbf{x}_1$ 和 $\mathbf{x}_2$ 的连续性，有

$$
\mathbf{x}_1(t_0) = \mathbf{x}_2(t_0).
$$

考虑初值问题

$$
\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t} = \mathbf{v}(\mathbf{x}(t), t), \quad \mathbf{x}(t_0) = \mathbf{x}_1(t_0).
$$

根据 ODE 的唯一性定理，$\mathbf{x}_1$ 和 $\mathbf{x}_2$ 在区间 $[t_0, T]$ 上必须重合。类似地，逆时间应用唯一性意味着两解在 $[0, t_0]$ 上也重合。因此，两解满足

$$
\mathbf{x}_1(t) = \mathbf{x}_2(t) \quad \text{for all } t \in [0, T],
$$

这与假设 $\mathbf{x}_1(0) \neq \mathbf{x}_2(0)$ 矛盾。因此可得

$$
\mathbf{x}_1(t) \neq \mathbf{x}_2(t) \quad \text{for all } t \in [0, T].
$$

通过保证解路径不相交，这一定理为流映射模型提供了隐含但关键的支持（见[第 10 章](../../chapters/10-distillation/index.md#ch-distillation)和[第 11 章](../../chapters/11-fast-generators/index.md#ch-fast-scratch)）。

## A.1.3 指数积分因子 {#app-exp-int-factor}

由一般时变速度 ${\mathbf{v}}$ 确定的 ODE 未必有闭式解，但在某些特殊情况下，可以解析求解，或将其化为结构更好的形式。

**示例。** 考虑以下线性标量 ODE：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = L(t){\mathbf{x}}(t),
$$

其中 $L(t) \in \mathbb{R}$ 是连续函数。这一方程有众所周知的闭式解（对于任意 $s$ 和 $t$）：

$$
{\mathbf{x}}(t) = {\mathbf{x}}(s) \cdot \exp\left( \int_s^t L(\tau) \,\mathop{}\!\mathrm{d} \tau \right).
$$

这一公式展示了解如何按照一个指数因子演化，该因子累积了时变系数 $L(t)$ 的作用。由此引出*指数积分因子*：

<span id="eq-exp-int-factor"></span>

$$
\begin{aligned}
\mathcal{E}(s \shortrightarrow t) := \exp\left( \int_s^t L(\tau)\mathop{}\!\mathrm{d} \tau \right),
\end{aligned}
\tag{A.1.2}
$$

尤其适用于动力学同时包含线性与非线性分量的更一般设定。

**半线性 ODE 与指数积分因子。** 现在考虑一类更广泛的 ODE，称为*半线性* ODE。这类方程将动力学分成线性部分（关于状态变量）和非线性余项：

<span id="eq-semilinear-ode"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = L(t){\mathbf{x}}(t) + {\mathbf{N}}({\mathbf{x}}(t), t),
\end{aligned}
\tag{A.1.3}
$$

其中 ${\mathbf{x}}(t) \in \mathbb{R}^D$ 是状态向量，$L(t)$ 是标量值连续函数，${\mathbf{N}}: \mathbb{R}^D \times [0,T] \to \mathbb{R}^D$ 是非线性向量场。这种半线性结构自然出现在许多物理与工程系统中。特别地，它也出现在扩散模型的概率流 ODE 表述中（见[公式 (4.1.7)](../../chapters/04-score-sde-framework/01-score-sde-principles.md#eq-prob-ode-gt)）。识别这一结构后，可使用积分因子表示，以解析方式处理已知的线性动力学，仅对剩余项作数值近似。这一原则在[第 9 章](../../chapters/09-numerical-solvers/index.md#ch-numerical-solvers)研究的扩散 ODE 求解器中起核心作用。

**步骤 1：通过积分因子分离非线性项。** 注意，从[公式 (A.1.3)](01-ordinary-differential-equations.md#eq-semilinear-ode)的半线性 ODE 中减去线性漂移，就能分离非线性部分：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} - L(t){\mathbf{x}}(t) = {\mathbf{N}}({\mathbf{x}}(t), t).
$$

为吸收线性项，在两侧乘以积分因子的逆：

$$
\mathcal{E}^{-1}(s \shortrightarrow t) = \exp\left(-\int_s^t L(\tau)\mathop{}\!\mathrm{d} \tau \right).
$$

现在对左侧应用乘积法则：

$$
\begin{aligned}
\mathcal{E}^{-1}(s \shortrightarrow t) \left( \frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} - L(t){\mathbf{x}}(t) \right)
&= \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \left[ \mathcal{E}^{-1}(s \shortrightarrow t) {\mathbf{x}}(t) \right].
\end{aligned}
$$

因此，方程变为：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \left[ \mathcal{E}^{-1}(s \shortrightarrow t) {\mathbf{x}}(t) \right]
= \mathcal{E}^{-1}(s \shortrightarrow t) {\mathbf{N}}({\mathbf{x}}(t), t).
$$

这一变换通过分离非线性分量简化原方程，使我们能够在变换后的坐标系中专注于非线性动力学。

**步骤 2：对时间积分。** 将两侧从 $s$ 积分到 $t$：

$$
\int_s^t \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} \tau} \left[ \mathcal{E}^{-1}(s \shortrightarrow \tau) {\mathbf{x}}(\tau) \right] \mathop{}\!\mathrm{d} \tau
= \int_s^t \mathcal{E}^{-1}(s \shortrightarrow \tau) {\mathbf{N}}({\mathbf{x}}(\tau), \tau)\mathop{}\!\mathrm{d} \tau.
$$

左侧就是变换后变量在 $t$ 和 $s$ 处的取值之差：

$$
\mathcal{E}^{-1}(s \shortrightarrow t) {\mathbf{x}}(t) - {\mathbf{x}}(s).
$$

因此，得到：

$$
\mathcal{E}^{-1}(s \shortrightarrow t) {\mathbf{x}}(t)
= {\mathbf{x}}(s) + \int_s^t \mathcal{E}^{-1}(s \shortrightarrow \tau) {\mathbf{N}}({\mathbf{x}}(\tau), \tau)\mathop{}\!\mathrm{d} \tau.
$$

**步骤 3：解出 ${\mathbf{x}}(t)$。** 在两侧乘以指数流 $\mathcal{E}(s \shortrightarrow t)$，得到解：

<span id="eq-exp-int-factor-integration"></span>

$$
\begin{aligned}
{\mathbf{x}}(t) = \underbrace{\mathcal{E}(s \shortrightarrow t)  {\mathbf{x}}(s)}_{\text{linear part}} + \underbrace{\int_s^t \mathcal{E}(\tau \shortrightarrow t) {\mathbf{N}}({\mathbf{x}}(\tau), \tau)\mathop{}\!\mathrm{d} \tau}_{\text{nonlinear part}} .
\end{aligned}
\tag{A.1.4}
$$

解自然分成线性与非线性两部分。指数积分器利用这一结构，以精确闭式求解线性部分，只离散化非线性余项。这使步长由非线性动力学决定，而非由可能很大的线性系数决定（例如，可参看[第 9.1 节](../../chapters/09-numerical-solvers/01-prologue.md#sec-prologue-solvers)中指数 Euler 更新与普通 Euler 更新的比较）。

## A.1.4 常微分方程的数值求解器 {#app-de-solver}

考虑[公式 (A.1.1)](01-ordinary-differential-equations.md#eq-general-ode)中的 ODE，

$$
\frac{\mathop{}\!\mathrm{d}\mathbf{x}(t)}{\mathop{}\!\mathrm{d} t}
=
\mathbf{v}(\mathbf{x}(t),t),
$$

并给定初始条件。求解 ODE 意味着寻找连续轨迹 $\mathbf{x}(t)$，使其每一时刻的瞬时变化都与速度场 $\mathbf{v}$ 一致。

等价地，解满足积分方程

<span id="eq-integral-form"></span>

$$
\mathbf{x}(t)
=
\mathbf{x}(0)
+
\int_0^t
\mathbf{v}(\mathbf{x}(\tau),\tau)\mathop{}\!\mathrm{d}\tau.
\tag{A.1.5}
$$

若该积分无法以闭式计算，就通过在有限个点计算速度场来数值近似轨迹。

**从连续轨迹到数值更新。** 数值求解器首先选择离散时间网格

$$
t_0,t_1,\ldots,t_M
$$

并产生相应的近似值

$$
\mathbf{x}_0,\mathbf{x}_1,\ldots,\mathbf{x}_M
$$

来近似连续轨迹。

间隔

$$
h_i:=t_{i+1}-t_i
$$

称为*步长*。较小的步长通常能提供更多关于轨迹的局部信息，但需要更多函数计算。因此，数值求解器的目标是高效利用已有的 $\mathbf{v}$ 计算，以相对较少的步骤获得准确轨迹。

不同求解器主要区别在于*下一次更新所用信息来自何处*。由此可以作如下分类。

**数值求解器的结构视角。** 对于本书涉及的方法，区分两种主要数值视角很有帮助。

-   **时间步进方法：** 从当前状态出发，将轨迹从一个时间点推进到下一个时间点。Euler、Runge–Kutta 和 Adams–Bashforth 方法是代表性例子。

    在时间步进方法内部，还可进一步区分：

    -   *单步方法*，用当前状态以及当前区间内产生的额外计算结果构造下一状态；

    -   *多步方法*，还会复用此前已完成步骤的信息。

-   **时间并行视角：** 可以将轨迹计算表述为同时更新多个时刻的暂定状态，而不必严格逐个完成所有时间点。Picard 迭代等不动点构造为这一思想提供了基本机制。

单步方法不应与单次*函数计算*混淆。例如，Euler 每步计算一次速度；Runge–Kutta 方法则可以在中间*阶段*多次计算速度，但仍属于单步方法。决定性区别在于是否需要此前已完成步骤的信息。

**直观图景**

后文用于扩散采样的数值 ODE 求解器思想可组织如下：

<span id="eq-classic-solvers"></span>

$$
\begin{aligned}
\begin{array}{c}
\begin{array}{rcl}
\textbf{Time-stepping}
&\rightarrow&
\begin{cases}
\text{single-step: }
    & \text{Euler, Runge--Kutta},\\
\text{multistep}
    & \text{Adams--Bashforth},
\end{cases}
\\[12pt]
\textbf{Parallel-in-time}
&\rightarrow&
\text{Picard fixed-point iteration}.
\end{array}
\\[6pt]
\downarrow
\\[-2pt]
\text{The same numerical ideas can be applied forward or backward in time.}
\end{array}
\end{aligned}
\tag{A.1.6}
$$

这一区分对后续扩散采样尤其有用：Runge–Kutta 型构造通过观察当前步骤*内部*获取更丰富的信息；Adams–Bashforth 型构造则通过*回看*先前计算的值获取信息。

**正向与逆时间求解 ODE。** 上述分类不依赖 ODE 的积分方向。普通正向积分采用

$$
t_{n+1}>t_n,
\qquad
h_n=t_{n+1}-t_n>0.
$$

扩散采样通常从含噪终止状态出发，使用递减时间网格：

$$
t_{n+1}<t_n,
\qquad
h_n=t_{n+1}-t_n<0.
$$

将步长视为有符号量时，同样的数值公式适用于任一方向。因此，ODE 的逆时间积分无须新的数值求解器族；Euler、Runge–Kutta、多步方法及相关方法都可用于递减时间网格。

这种简单的反向积分应与 SDE 的时间反转区分，后者的随机动力学需要额外分析。下一节将讨论这一问题。

## A.1.5 代表性数值求解器 {#app-de-solver-examples}

上述分类说明了求解器从何处获取下一次更新的信息。现在通过几个代表性例子具体说明这些思想。

这里有意只选少量例子。Euler 展示最简单的单步更新；Runge–Kutta 方法展示如何在当前步骤内使用额外阶段；Adams–Bashforth 展示如何复用历史；Picard 迭代则展示时间并行方法背后的不动点视角。

以下例子中，令

$$
t_{n+1}=t_n+h.
$$

步长 $h$ 带符号：正向积分时 $h>0$，逆向积分时 $h<0$。

**Euler 方法。** Euler 方法是最简单的单步方法，仅使用当前区间起点处的速度：

$$
\mathbf{x}_{n+1}
=
\mathbf{x}_n
+
h\,\mathbf{v}(\mathbf{x}_n,t_n).
$$

几何上，Euler 观察当前运动方向，并沿此方向推进完整一步。

在标准光滑性假设下，Euler 的局部截断误差为 $\mathcal O(h^2)$，全局误差为 $\mathcal O(h)$。其简单性使之成为有用的一阶基线，但当速度在一步内显著变化时，精度会下降。

**Runge–Kutta 方法。** Euler 只使用区间起点处的速度。Runge–Kutta（RK）方法通过在同一步内生成的额外*阶段*计算速度，获取更多信息。

因此，它们属于*单步、多阶段*方法：构造 $\mathbf{x}_{n+1}$ 所用的全部信息都由当前起点 $\mathbf{x}_n$ 生成，但一个步骤可以包含多次函数计算。

下面用两个二阶方法说明 RK 的基本原则。

**显式中点法。** 显式中点法先用当前速度预测区间中点的状态：

$$
\mathbf{x}_{\mathrm{mid}}
=
\mathbf{x}_n
+
\frac{h}{2}
\mathbf{v}(\mathbf{x}_n,t_n).
$$

随后在预测的中点计算速度，并用该方向完成整个更新：

$$
\mathbf{x}_{n+1}
=
\mathbf{x}_n
+
h\,
\mathbf{v}
\left(
\mathbf{x}_{\mathrm{mid}},
t_n+\frac{h}{2}
\right).
$$

因此，中点法先“查看”步骤内部，再决定如何完成更新，而不假设起始速度在整个区间都具有代表性。

**Heun 方法（改进 Euler）。** Heun 方法是另一种两阶段、二阶 Runge–Kutta 方法。它不考察中点，而是先用 Euler 方法预测终点：

$$
\mathbf{x}_{\mathrm{pred}}
=
\mathbf{x}_n
+
h\,\mathbf{v}(\mathbf{x}_n,t_n).
$$

随后在预测终点计算速度，并对起始速度与预测的终点速度取平均：

$$
\mathbf{x}_{n+1}
=
\mathbf{x}_n
+
\frac{h}{2}
\left[
\mathbf{v}(\mathbf{x}_n,t_n)
+
\mathbf{v}(\mathbf{x}_{\mathrm{pred}},t_n+h)
\right].
$$

显式中点法和 Heun 方法均使用两个阶段，在标准光滑性假设下，局部截断误差为 $\mathcal O(h^3)$，全局误差为 $\mathcal O(h^2)$。二者只在步骤内额外信息的收集与组合方式上不同。

高阶 Runge–Kutta 方法使用更多阶段扩展同一原则；这里省略其标准公式。在扩散采样中，[Karras et al. (2022)](../../references.md#ref-karras2022elucidating) 直接采用 Heun 方法作为 ODE 求解器。

**Adams–Bashforth 方法。** Runge–Kutta 方法通过查看当前步骤*内部*获取额外信息。多步方法则*回看*并复用此前已完成步骤的信息。

例如，在均匀步长 $h$ 下，二阶 Adams–Bashforth 方法（AB2）为

$$
\mathbf{x}_{n+1}
=
\mathbf{x}_n
+
h
\left[
\frac{3}{2}
\mathbf{v}(\mathbf{x}_n,t_n)
-
\frac{1}{2}
\mathbf{v}(\mathbf{x}_{n-1},t_{n-1})
\right].
$$

因此，第二条信息来自保存的上一步速度，而非当前区间内的新计算。

在标准光滑性假设及足够准确的初始化下，AB2 的局部截断误差为 $\mathcal O(h^3)$，全局误差为 $\mathcal O(h^2)$。由于第一步没有历史信息，多步方法必须先使用另一求解器初始化。

上述公式假设网格均匀。在非均匀网格上，多步系数依赖相邻步长。这一复用历史信息的原则，将在[第 9.3 节](../../chapters/09-numerical-solvers/03-deis.md#sec-exponential)和[第 9.5.4 节](../../chapters/09-numerical-solvers/05-dpm-solver-plus-plus.md#subsec-dpm-solver--multi)中针对扩散设计的多步求解器中再次出现。

**Picard 迭代。** 上述方法通过构造局部时间步进更新来近似轨迹。Picard 迭代采用另一种思想：通过 ODE 的积分形式，反复改进整条候选轨迹。

从初始猜测 $\mathbf{x}^{(0)}(t)$ 出发，并要求其满足 $\mathbf{x}^{(0)}(0)=\mathbf{x}(0)$，Picard 迭代定义为

$$
\mathbf{x}^{(k+1)}(t)
=
\mathbf{x}(0)
+
\int_0^t
\mathbf{v}
\bigl(
\mathbf{x}^{(k)}(s),s
\bigr)
\mathop{}\!\mathrm{d} s.
$$

因此，每次迭代都沿上一条候选轨迹计算速度，再使用这些计算结果构造新轨迹。

这一迭代的不动点满足原始 ODE 积分方程。在 Picard–Lindelöf 定理的标准条件下，迭代局部收敛到 ODE 解。

对于后文的用途，关键观察在计算层面：一次 Picard 迭代中，每个时刻的速度都依赖*上一条*候选轨迹。因此，不同时间点的计算可以独立进行，为时间并行计算提供可能。我们将在[第 9.6 节](../../chapters/09-numerical-solvers/06-paradigms.md#sec-picard)重访这一视角。

# 4.2 前向／逆时间 SDE 与 PF-ODE 的边缘分布匹配 {#subsec-fpe-sde-ode}

<span id="fig-score-sde-three-dynamics"></span>

![前向 SDE、逆时间 SDE 与概率流 ODE 共享边缘密度](../../../en/assets/chapter-04/score-sde-three-dynamics.png)

**图 4.5：** **共享相同边缘分布 $p_t$ 的三种动力学。** **上图：加噪方向。** 前向 SDE 作为基准参考，通过将数据扩散向 Gaussian 先验来定义边缘分布 $p_t$。PF-ODE 在*沿时间正向运行*时，确定性地遵循相同的边缘分布路径。**下图：去噪方向。** 逆时间 SDE 与*沿时间反向运行*的 PF-ODE 以相反顺序产生相同的边缘分布，这由 Fokker–Planck 方程保证。逆时间 SDE 给出随机去噪过程，PF-ODE 则给出确定性去噪过程。因此，重复出现的密度快照是有意安排的：这些过程在单个时刻的边缘分布层面一致，但样本层面的动力学不同。PF-ODE 同时出现在上下两图，因为它是可以沿任一时间方向积分的确定性流。

来源：作者借助 AI 辅助编程绘制。

本节将 *Fokker–Planck 方程*作为支配扩散模型的核心原理来介绍。它描述概率密度如何在含噪动力学下演化，并且是存在随机性时变量替换公式的自然对应物（见[附录 B](../../appendices/b-density-evolution/index.md#app-continuity)）。从定义边缘分布 $p_t$ 的前向扩散过程出发，我们解释逆时间 SDE 与 PF-ODE 如何被设计为重现同一个边缘分布族。随后，通过一个 Gaussian 示例将这一原理具体化，其中逆时间 SDE 与 PF-ODE 都可以直接、显式地计算。

## 4.2.1 用 Fokker–Planck 方程保证边缘密度一致 {#alignment-of-marginal-densities}

扩散模型中的一个核心概念是，*不同过程可以产生相同的边缘分布序列*（本小节后面将加以说明）。目标是构造一个将 $p_{\mathrm{prior}}$ 转化为 $p_{\mathrm{data}}$ 的过程，使各时刻的边缘分布一致，尤其是在 $t=0$ 时。只要过程可计算且允许高效采样，其具体形式就是次要的。这自然引出一个基本问题：

<span id="question-4-2-1"></span>

**问题 4.2.1**

如何保证不同过程产生相同的边缘分布？

回到我们的设定，一旦指定前向 SDE，它就定义了从 $p_{\mathrm{data}}$ 到 $p_{\mathrm{prior}}$ 的边缘密度演化。随后构造逆时间 SDE 与 PF-ODE，使其轨迹产生的边缘分布与前向过程完全一致。这一对应关系的关键是 Fokker–Planck 方程，它支配扩散过程下边缘密度的演化。[图 4.5](02-matching-marginals.md#fig-score-sde-three-dynamics)给出示意。以下定理（[Anderson, 1982](../../references.md#ref-anderson1982reverse)；[Song et al., 2020](../../references.md#ref-song2020score)）奠定了这一联系的基础：

<span id="thm-fpe"></span>

**定理 4.2.1：通过 Fokker–Planck 方程匹配边缘分布**

考虑前向 SDE

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}(t)
=
{\mathbf{f}}({\mathbf{x}}(t),t)\mathop{}\!\mathrm{d} t
+
g(t)\mathop{}\!\mathrm{d} {\mathbf{w}}(t),
\qquad
{\mathbf{x}}(0)\sim p_0=p_{\mathrm{data}}.
$$

在标准正则性条件下，其边缘密度 $p_t$ 按 Fokker–Planck 方程演化：

<span id="eq-fp-general"></span>

$$
\begin{aligned}
\partial_t p_t({\mathbf{x}})
&=
-\nabla_{{\mathbf{x}}}\cdot
\bigl[{\mathbf{f}}({\mathbf{x}},t)p_t({\mathbf{x}})\bigr]
+
\tfrac12 g^2(t)\Delta_{{\mathbf{x}}}p_t({\mathbf{x}})
\\
&=
-\nabla_{{\mathbf{x}}}\cdot
\bigl[{\mathbf{v}}({\mathbf{x}},t)p_t({\mathbf{x}})\bigr],
\end{aligned}
\tag{4.2.1}
$$

其中 $\Delta_{{\mathbf{x}}}$ 表示 Laplace 算子，且

$$
{\mathbf{v}}({\mathbf{x}},t)
:=
{\mathbf{f}}({\mathbf{x}},t)
-
\tfrac12 g^2(t)
\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}).
$$

同一族边缘分布 $\{p_t\}_{t\in[0,T]}$ 可以由两种不同的动力学实现：

-   **(i)** *PF-ODE*

    $$
    \frac{\mathop{}\!\mathrm{d}\tilde{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
    =
    {\mathbf{v}}(\tilde{\mathbf{x}}(t),t).
    $$

    从 $\tilde{\mathbf{x}}(0)\sim p_0$ 出发，自 $0$ 正向求解到 $T$，得到 $\tilde{\mathbf{x}}(t)\sim p_t$。等价地，从 $\tilde{\mathbf{x}}(T)\sim p_T$ 出发，对同一个 ODE 自 $T$ 反向求解到 $0$，就会按相反顺序遵循同样的边缘分布。

-   **(ii)** *逆时间 SDE*

    $$
    \mathop{}\!\mathrm{d}\bar{\mathbf{x}}(t)
    =
    \Bigl[
    {\mathbf{f}}(\bar{\mathbf{x}}(t),t)
    -
    g^2(t)\nabla_{{\mathbf{x}}}\log p_t(\bar{\mathbf{x}}(t))
    \Bigr]\mathop{}\!\mathrm{d} t
    +
    g(t)\mathop{}\!\mathrm{d}\bar{\mathbf{w}}(t).
    $$

    从 $\bar{\mathbf{x}}(T)\sim p_T$ 出发，该 SDE 沿时间反向从 $t=T$ 求解到 $t=0$。沿这一逆向演化，它在时刻 $t$ 的边缘分布为

    $$
    \bar{\mathbf{x}}(t)\sim p_t.
    $$

因此，虽然逆时间 SDE 是随机的、PF-ODE 是确定性的，二者仍可以在每个噪声水平上具有相同的边缘分布。

**定理证明。**

证明见[第 D.2.5 节](../../appendices/d-supplementary-proofs/02-score-based-perspective.md#app-sec-scoresde-fpe-proof)；[第 4.6.1 节](06-fokker-planck-and-reverse-sde.md#subsec-fpe-reason)则利用概率的边缘化技巧，进一步解释 Fokker–Planck 方程背后的直觉。

**固定边缘分布下的多种条件分布。** 为理解 PF-ODE 如何沿时间正向输运 $p_{\mathrm{data}}$（或等价地，反向输运 $p_{\mathrm{prior}}$），考虑*流映射* ${\boldsymbol{\Psi}}_{s \to t}: \mathbb{R}^D \to \mathbb{R}^D$，其中 ${\boldsymbol{\Psi}}_{s \to t}({\mathbf{x}}_s)$ 表示 PF-ODE 在时刻 $t$ 的解，该解以 ${\mathbf{x}}_s$ 为初值，初始化时刻为 $s$，适用于任意时刻 $s, t \in[0,T]$。换言之，这一映射接收初始状态 ${\mathbf{x}}_s$，直接跳到其在 $t$ 时的状态：

<span id="eq-flow-map-def"></span>

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{s \to t}({\mathbf{x}}_s) \coloneqq {\mathbf{x}}_s + \int_s^t {\mathbf{v}}({\mathbf{x}}_\tau, \tau)\mathrm{d}\tau,
\end{aligned}
\tag{4.2.2}
$$

其速度场为

$$
\begin{aligned}
{\mathbf{v}}({\mathbf{x}}, \tau) := {\mathbf{f}}({\mathbf{x}}, \tau) - \frac{1}{2} g^2(\tau)\nabla_{{\mathbf{x}}}\log p_\tau({\mathbf{x}}).
\end{aligned}
$$

这里的积分表示沿 PF-ODE 轨迹累积的净位移。*流映射*本来就是微分方程、动力系统与连续介质力学中的经典术语：它表示在给定动力学下，将一个状态从时刻 $s$ 的位置输运到时刻 $t$ 的位置的映射。速度场 ${\mathbf{v}}({\mathbf{x}},t)$ 描述瞬时运动，流映射 ${\boldsymbol{\Psi}}_{s\to t}$ 则记录有限时间区间内的累积运动（[Arnold, 1988](../../references.md#ref-arnold1988geometrical)；[Chorin and Marsden, 1993](../../references.md#ref-chorin1993mathematical)）[^1]。本书始终在这一标准数学与物理意义下使用该术语，尤其是在[第 10 章](../10-distillation/index.md#ch-distillation)与[第 11 章](../11-fast-generators/index.md#ch-fast-scratch)讨论基于扩散的快速生成器时。在 ${\mathbf{v}}$ 满足温和的光滑性假设下，流映射 ${\boldsymbol{\Psi}}_{s \to t}: \mathbb{R}^D \to \mathbb{R}^D$ 是光滑双射[^2]。

对任意 $t\in[0,T]$，*推前密度*定义为

$$
p_t^{\mathrm{fwd}}({\mathbf{x}}_t) := \int \delta\bigl({\mathbf{x}}_t - {\boldsymbol{\Psi}}_{0 \to t}({\mathbf{x}}_0)\bigr)\, p_{\mathrm{data}}({\mathbf{x}}_0) \,\mathop{}\!\mathrm{d} {\mathbf{x}}_0,
$$

记作 $\boldsymbol{\Psi}_{0\to t} \sharp p_{\mathrm{data}}$，表示时刻 $t$ 的分布，对应映射为 $\boldsymbol{\Psi}_{0\to t}$。[定理 4.2.1](02-matching-marginals.md#thm-fpe)保证 $p_t^{\mathrm{fwd}} = p_t$，其中 $p_t$ 是前向 SDE 的边缘密度，从而将确定性 PF-ODE 与随机核联系起来：

$$
p_t({\mathbf{x}}_t) = \int p_t({\mathbf{x}}_t |{\mathbf{x}}_0)\, p_{\mathrm{data}}({\mathbf{x}}_0)\,\mathop{}\!\mathrm{d} {\mathbf{x}}_0
= \int \delta\bigl({\mathbf{x}}_t - \boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)\bigr)\, p_{\mathrm{data}}({\mathbf{x}}_0)\,\mathop{}\!\mathrm{d} {\mathbf{x}}_0.
$$

这意味着，无穷多种条件分布 $Q_t({\mathbf{x}}_t |{\mathbf{x}}_0)$ 可以产生相同的 $p_t({\mathbf{x}}_t)$，例如：

-   **随机形式（无须模拟）：** $Q_t({\mathbf{x}}_t |{\mathbf{x}}_0) = p_t({\mathbf{x}}_t |{\mathbf{x}}_0)$，

-   **确定性形式（需要求解 ODE）：** $Q_t({\mathbf{x}}_t |{\mathbf{x}}_0) = \delta\bigl({\mathbf{x}}_t - \boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)\bigr)$，

-   **混合形式：**

    $$
    Q_t({\mathbf{x}}_t |{\mathbf{x}}_0) = \lambda p_t({\mathbf{x}}_t |{\mathbf{x}}_0) + (1-\lambda)\delta\bigl({\mathbf{x}}_t - \boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)\bigr),
    $$

    $\lambda\in[0,1]$.

$Q_t({\mathbf{x}}_t |{\mathbf{x}}_0)$ 的这种非唯一性，源于边缘约束不能唯一确定条件分布。这一概念将在[第 5.2.2 节](../05-flow-based-perspective/02-flow-matching-framework.md#subsec-fm-framework)与[第 9.2.3 节](../09-numerical-solvers/02-ddim.md#subsec-ddim)再次出现。特别是，存在一整族与同一边缘分布 $p_t$ 一致的逆时间 SDE。

<span id="observation-4-2-1"></span>

**观察 4.2.1：匹配指定的边缘密度**

多种过程可以产生相同的边缘密度序列；关键在于满足 Fokker–Planck 方程。这一基本认识，使我们能够灵活设计从 $p_{\mathrm{prior}}$ 转化为 $p_{\mathrm{data}}$，或沿相反方向转化的生成过程。

Fokker–Planck 方程处于扩散模型的核心，其根源是概率密度的基本*变量替换公式*（系统介绍见[附录 B](../../appendices/b-density-evolution/index.md#app-continuity)）。这一原则远非无关紧要的技术细节，它贯穿后续论述，尤其会在[第 5.2 节](../05-flow-based-perspective/02-flow-matching-framework.md#sec-flow-matching-framework)再次出现。

## 4.2.2 可计算的示例：Gaussian 动力学的演化 {#subsec-example-gaussian}

当 $p_{\mathrm{data}}$ 是正态分布（或 Gaussian 混合分布）时，score 函数具有闭式表达式。这是建立扩散过程直觉的理想设定：仅使用基础微积分，就能显式推导逆时间 SDE 与 PF-ODE，无须借助高深的数学工具。本小节在这一可计算情形中说明这些方程的行为。

**Gaussian 情形下逆时间 SDE 的精确计算。** 当 $p_{\mathrm{data}}$ 为 Gaussian 分布时，可以直接推导[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)，无须依赖 [Anderson (1982)](../../references.md#ref-anderson1982reverse) 的一般理论及证明。为说明核心思想，我们考虑一维情形；更高维情形可以同样扩展。

从前向 SDE 开始：

$$
\mathop{}\!\mathrm{d} x(t)  =  f(t)  x(t)  \mathop{}\!\mathrm{d} t  +  g(t)  \mathop{}\!\mathrm{d} w_t,
$$

执行步长为 $\Delta t>0$ 的一个小 Euler 步：

$$
x_{t+\Delta t}  =  a  x_t  +  r  \epsilon,
$$

其中 $a:=1+f(t)\Delta t$、$r:=g(t)\sqrt{\Delta t}$ 且 $\epsilon\sim\mathcal N(0,1)$。等价地，前向单步转移核为 Gaussian：

$$
x_{t+\Delta t}   |   x_t  \sim  \mathcal N\!\big(\cdot;a  x_t,   r^2\big).
$$

由于假设 $p_{\mathrm{data}}$ 为 Gaussian 分布，时刻 $t$ 的当前边缘分布也为 Gaussian，具有以下形式：

$$
x_t \sim \mathcal N(\cdot;m_t,   s_t^2),
$$

其中 $m_t$ 与 $s_t$ 为标量。因此，条件化只需将两个 Gaussian 分布相乘并重新归一化，代数计算仍很初等。

由 Bayes 法则，忽略一个常数，条件密度为先验与转移核之积：

$$
\begin{aligned}
p(x_t   |   x_{t+\Delta t})  &\propto p(x_{t+\Delta t} | x_t   ) p_t({\mathbf{x}}_t) \\&\propto
\exp\Big(-\frac{(x_t-m_t)^2}{2s_t^2}\Big)
\exp\Big(-\frac{(x_{t+\Delta t}-a  x_t)^2}{2r^2}\Big).
\end{aligned}
$$

指数是关于 $x_t$ 的二次式。展开两个平方并合并同类项，即可明确看出哪些系数起作用：

$$
-2\log p(x_t   |   x_{t+\Delta t})
 =
A  x_t^2  -  2 B  x_t  +  \text{const},
$$

其中

$$
A:=\frac{1}{s_t^2}+\frac{a^2}{r^2},\quad
B:=\frac{m_t}{s_t^2}+\frac{a  x_{t+\Delta t}}{r^2}.
$$

这里，$A$ 是精度之和（先验精度加上通过 $a$ 输运的转移核精度），而 $B$ 是相应目标的精度加权和。有了这些量，通过配方即可直接得到后验：

$$
A x_t^2 - 2 B x_t
  =
A\Big(x_t-\frac{B}{A}\Big)^2  -  \frac{B^2}{A},
$$

因此，条件分布为 Gaussian，方差为 $1/A$，均值为 $B/A$：

$$
\operatorname{Var}(x_t   |   x_{t+\Delta t})   =   \frac{1}{\frac{1}{s_t^2}+\frac{a^2}{r^2}},
\qquad
\mathbb E[x_t   |   x_{t+\Delta t}]   =
\frac{\frac{m_t}{s_t^2}+\frac{a  x_{t+\Delta t}}{r^2}}{\frac{1}{s_t^2}+\frac{a^2}{r^2}}.
$$

这些闭式形式已经描述了任意小 $\Delta t$ 下的逆向转移。为了得到逆时间 SDE，我们现在对小 $\Delta t$ 展开它们。

使用 $a=1+f(t)\Delta t$ 与 $r^2=g^2(t)\Delta t$。当 $\Delta t\to 0$ 时，$\frac{a^2}{r^2}\sim \frac{1}{g^2(t)\Delta t}$ 的贡献在精度中占主导，因此方差变为

$$
\operatorname{Var}(x_t   |   x_{t+\Delta t})
= \left(\frac{1}{s_t^2}+\frac{a^2}{r^2}\right)^{-1}
= g^2(t)  \Delta t  +  \mathcal{O}(\Delta t^2),
$$

这说明逆向步骤的扩散尺度 $g(t)$ 与前向步骤相同。对于均值，将比值 $B/A$ 展开到一阶：

$$
\mathbb E[x_t   |   x_{t+\Delta t}]
  =  x_{t+\Delta t}
 +  \Delta t\left[
-\left(f(t)+\frac{g^2(t)}{s_t^2}\right)  x_{t+\Delta t}
 +  \frac{g^2(t)}{s_t^2}  m_t
\right]  +  \mathcal{O}(\Delta t^2).
$$

将均值与方差合在一起，得到单步逆向转移核

$$
x_t    |    x_{t+\Delta t}
 \sim
\mathcal N\!\left(
x_{t+\Delta t} + \Delta t\!\left[-\left(f+\frac{g^2}{s_t^2}\right)  x_{t+\Delta t} + \frac{g^2}{s_t^2}  m_t\right],
 g^2  \Delta t\right)
 +  \mathcal{O}(\Delta t^2).
$$

这正是从 $t+\Delta t$ 到 $t$ 反向运行的 Euler–Maruyama 更新：

$$
x_t - x_{t+\Delta t}
= \Delta t\!\left[-\left(f+\frac{g^2}{s_t^2}\right) x_{t+\Delta t}
+ \frac{g^2}{s_t^2} m_t\right]
+ g\sqrt{\Delta t}\,\epsilon.
$$

这一逆向漂移可通过 score 表示，因为对于 Gaussian 边缘分布 $p_t=\mathcal N(m_t,s_t^2)$，

$$
\partial_x \log p_t(x) = -\frac{x-m_t}{s_t^2}
 \Longrightarrow
-\left(f+\frac{g^2}{s_t^2}\right)x
+\frac{g^2}{s_t^2}m_t
=
-\left[f x-g^2\partial_x\log p_t(x)\right].
$$

因此，令 $\Delta t\to0$，得到原始扩散时钟下的逆时间 SDE：

$$
\mathop{}\!\mathrm{d} \bar x(t)
=
\left[
f(t)\bar x(t)
-
g^2(t)\partial_x\log p_t(\bar x(t))
\right]\mathop{}\!\mathrm{d} t
+
g(t)\mathop{}\!\mathrm{d}\bar w(t),
\qquad t:T\to0.
$$

这与[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)中的一般逆时间 SDE 完全一致。在向量形式中，标量导数 $\partial_x\log p_t$ 被替换为 $\nabla_{{\mathbf{x}}}\log p_t$。

**Gaussian 情形下 PF-ODE 的精确计算。** 假设数据分布为 Gaussian 时，也可以直接推导 PF-ODE 公式，无须使用 Fokker–Planck 方程等复杂工具。最终可以看到，PF-ODE 的边缘密度与前向 SDE、逆时间 SDE 的边缘密度都一致，从而对[第 4.2 节](02-matching-marginals.md#subsec-fpe-sde-ode)所讨论的 Fokker–Planck 理论给出构造性的验证。

假设 $x_t\sim\mathcal N(\cdot;m_t,s_t^2)$，对应时刻为 $t$。一个大小为 $\Delta t$ 的确定性小步可写作光滑映射

$$
x_{t+\Delta t}=\Phi_{t,\Delta t}(x_t)
= x_t + \Delta t v_t(x_t) + \mathcal{O}(\Delta t^2),
$$

这只是关于 $\Delta t$ 的一阶 Taylor 展开。我们的目标是构造一个速度场，使其确定性流重现同样的 Gaussian 边缘分布演化。一个特别简单的选择是仿射速度场

$$
v_t(x)=a_t x+b_t.
$$

由于仿射动力学保持 Gaussian 性，我们可以通过匹配均值和方差与前向 SDE 的演化，确定 $a_t$ 与 $b_t$。将其代回该步骤，得到

$$
x_{t+\Delta t}=(1+\alpha_t \Delta t) x_t+\beta_t \Delta t+\mathcal{O}(\Delta t^2),
\qquad
\alpha_t:=a_t,\ \ \beta_t:=b_t.
$$

现在通过这一映射推前 $x_t\sim\mathcal N(\cdot;m_t,s_t^2)$，并跟踪均值与方差到一阶：

$$
\begin{aligned}
\mathbb E[x_{t+\Delta t}]
&= m_t+\Delta t(\alpha_t m_t+\beta_t)+\mathcal{O}(\Delta t^2),
\\
\operatorname{Var}(x_{t+\Delta t})
&= s_t^2+\Delta t (2\alpha_t s_t^2)+\mathcal{O}(\Delta t^2).
\end{aligned}
$$

另一方面，前向 SDE $\mathop{}\!\mathrm{d} x=f(t) x \mathop{}\!\mathrm{d} t+g(t) \mathop{}\!\mathrm{d} w_t$ 具有以下初等矩公式（见[式 (4.4.3)](04-sde-instantiations.md#eq-mean-var-evolution)）：

$$
m_t' = f(t) m_t,
\qquad
(s_t^2)' = 2 f(t) s_t^2 + g^2(t).
$$

匹配 $\Delta t$ 的系数，得到

$$
\alpha_t = f(t)+\frac{g^2(t)}{2 s_t^2},
\qquad
\beta_t = - \frac{g^2(t)}{2 s_t^2} m_t.
$$

采用这些选择后，步骤变为

$$
x_{t+\Delta t}
= x_t + \Delta t\!\left[\Big(f(t)+\frac{g^2(t)}{2s_t^2}\Big)x_t
-\frac{g^2(t)}{2s_t^2} m_t\right]+\mathcal{O}(\Delta t^2).
$$

由于对于 Gaussian 分布 $p_t=\mathcal N(m_t,s_t^2)$ 有 $\partial_x\log p_t(x)=-(x-m_t)/s_t^2$，可以将方括号改写为 $f(t) x_t - \tfrac12 g^2(t) \partial_x\log p_t(x_t)$。因此，

$$
x_{t+\Delta t}
= x_t + \Delta t\Big[f(t) x_t - \tfrac12 g^2(t) \partial_x\log p_t(x_t)\Big] + \mathcal{O}(\Delta t^2).
$$

最后，除以 $\Delta t$ 并令 $\Delta t\to 0$，得到 PF-ODE

$$
x'(t)  =  f(t) x(t)  -  \tfrac12 g^2(t) \partial_x\log p_t\!\big(x(t)\big).
$$

要理解这个 ODE 为什么具有与前向 SDE（以及逆时间 SDE）相同的边缘分布，只需注意上面的漂移是线性项加上平移项。因此，$x(t)$ 仿射依赖于 $x(0)$，而仿射映射将 Gaussian 分布映射为 Gaussian 分布。此外，根据我们的匹配，沿这一 ODE 的均值 $m_t$ 与方差 $s_t^2$，满足与前向 SDE 完全相同的两个标量 ODE，初值也相同。因此，两种演化的 $p_t=\mathcal N(m_t,s_t^2)$ 在每个时刻 $t$ 都相同。

[^1]: 这一映射背后的物理思想历史更久。18 世纪的流体力学就已经通过速度场和跟踪物质粒子的演化来研究运动，尤其体现在 Euler 与 Lagrange 的奠基性工作中（[Euler, 1757](../../references.md#ref-euler1757fluid)；[Lagrange, 1783](../../references.md#ref-lagrange1783fluid)）。

[^2]: 提前说明：PF-ODE 流映射 ${\boldsymbol{\Psi}}_{s\to t}$ 恰好就是将 $p_s$ 输运到 $p_t$ 的归一化流（NF）双射（详见[第 5.1 节](../05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#sec-flow-based-method)）。区别在于，一旦前向 SDE 及其边缘密度固定，PF-ODE 公式就唯一指定了其对应的速度场 ${\mathbf{v}}({\mathbf{x}},t) = {\mathbf{f}}({\mathbf{x}},t) -\frac{1}{2}g^2(t)\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$，而归一化流或连续时间归一化流直接参数化并学习速度场。二者都依赖相同的连续变量替换原理。
# 4.4 SDE 的具体实例 {#sec-instances}

[Song et al. (2020)](../../references.md#ref-song2020score) 根据演化过程中方差的行为，将前向 SDE 的漂移项 ${\mathbf{f}}({\mathbf{x}}, t)$ 与扩散项 $g(t)$ 分为三类。这里重点介绍两种常用类型：*方差爆炸（<u>V</u>ariance <u>E</u>xplosion，VE）* SDE 与*方差保持（<u>V</u>ariance <u>P</u>reserving，VP）* SDE。虽然也可以自定义噪声调度，但其设计会显著影响实际性能。[表 4.1](04-sde-instantiations.md#tb-sde-instances)总结了这两个 SDE 实例。

<span id="tb-sde-instances"></span>

**表 4.1：前向 SDE 总结**

|  | **VE SDE** | **VP SDE** |
| --- | --- | --- |
| $\mathbf{f}({\mathbf{x}}, t)$ | $\mathbf{0}$ | $-\frac{1}{2}\beta(t){\mathbf{x}}$ |
| $g(t)$ | $\sqrt{\frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t}}$ | $\sqrt{\beta(t)}$ |
| SDE | $\mathop{}\!\mathrm{d}{\mathbf{x}}(t) = g(t) \mathop{}\!\mathrm{d}{\mathbf{w}}(t)$ | $\mathop{}\!\mathrm{d}{\mathbf{x}}(t) = -\frac{1}{2}\beta(t) {\mathbf{x}}(t) \mathop{}\!\mathrm{d} t + \sqrt{\beta(t)} \mathop{}\!\mathrm{d}{\mathbf{w}}(t)$ |
| $p_t({\mathbf{x}}_t\vert {\mathbf{x}}_0)$ | $\mathcal{N}\left({\mathbf{x}}_t ; {\mathbf{x}}_0, \left(\sigma^2(t) - \sigma^2(0)\right){\mathbf{I}}\right)$ | $\mathcal{N}\left({\mathbf{x}}_t ; {\mathbf{x}}_0 e^{-\frac{1}{2}\int_{0}^{t} \beta(\tau)\mathop{}\!\mathrm{d} \tau}, {\mathbf{I}} - {\mathbf{I}} e^{-\int_{0}^{t} \beta(\tau)\mathop{}\!\mathrm{d} \tau}\right)$ |
| $p_{\mathrm{prior}}$ | $\mathcal{N}(\boldsymbol{0}, \sigma^2(T){\mathbf{I}})$ | $\mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$ |

## 4.4.1 VE SDE {#ve-sde}

VE SDE 包含以下组成部分：

-   **漂移项：** 零漂移项 $\mathbf{f}=\boldsymbol{0}$。

-   **扩散项：** $g(t)=\sqrt{\frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t}}$，其中 $\sigma(t)$ 为某个函数。

前向 SDE 因而具有以下形式：

<span id="eq-vesde"></span>

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}(t) =  \sqrt{\frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t}} \mathop{}\!\mathrm{d}{\mathbf{w}}(t).
\tag{4.4.1}
$$

类似地，[第 4.4.3 节](04-sde-instantiations.md#subsec-scoresde-perturbation-kernel)的结果给出 VE SDE 的扰动核，并指导选择适当的先验分布：

-   **扰动核：**

    $$
    \begin{aligned}
    p_t({\mathbf{x}}_t | {\mathbf{x}}_0) &= \mathcal{N}\left({\mathbf{x}}_t ; {\mathbf{x}}_0, \left(\sigma^2(t) - \sigma^2(0)\right){\mathbf{I}}\right)
    \end{aligned}
    $$

-   **先验分布：** 假设 $\sigma(t)$ 在 $t \in [0,T]$ 上是递增函数，且 $\sigma^2(T) \gg \sigma^2(0)$。先验分布为：

    $$
    p_{\mathrm{prior}} := \mathcal{N}(\boldsymbol{0}, \sigma^2(T){\mathbf{I}}).
    $$

VE SDE 的一个典型实例是采用以下设计的 NCSN：

$$
\sigma(t):=\sigma_{\text{min}}\left(\frac{\sigma_{\text{max}}}{\sigma_{\text{min}}} \right)^t,\quad \text{for } t\in(0,1],
$$

其中 $\sigma_{\text{min}}$ 与 $\sigma_{\text{max}}$ 是预先指定的常数。也就是说，方差序列被设计为等比数列。由此，NCSN 可以视为 VE SDE 的离散化版本，正如[第 4.1.1 节](01-score-sde-principles.md#subsec-scoresde-motivation)所述。

## 4.4.2 VP SDE {#vp-sde}

令 $\beta\colon [0,T] \rightarrow \mathbb{R}_{\geq 0 }$ 为 $t$ 的非负函数。VP SDE 由以下组成部分定义：

-   **漂移项：** 线性漂移 $\mathbf{f}({\mathbf{x}}, t) = -\frac{1}{2}\beta(t){\mathbf{x}}$。

-   **扩散项：** $g(t) = \sqrt{\beta(t)}$。

因此，前向 SDE 表示为：

<span id="eq-forward-linear-sde"></span>

$$
\begin{aligned}
\mathop{}\!\mathrm{d}{\mathbf{x}}(t) = -\frac{1}{2}\beta(t) {\mathbf{x}}(t) \mathop{}\!\mathrm{d} t + \sqrt{\beta(t)} \mathop{}\!\mathrm{d}{\mathbf{w}}(t).
\end{aligned}
\tag{4.4.2}
$$

利用[第 4.4.3 节](04-sde-instantiations.md#subsec-scoresde-perturbation-kernel)的结果，可以推导 VP SDE 的扰动核，并选择适当的先验分布：

-   **扰动核：**

    $$
    p_t({\mathbf{x}}_t|{\mathbf{x}}_0) = \mathcal{N}\left({\mathbf{x}}_t ; {\mathbf{x}}_0 e^{-\frac{1}{2}\int_{0}^{t} \beta(\tau)\mathop{}\!\mathrm{d} \tau}, {\mathbf{I}} - {\mathbf{I}} e^{-\int_{0}^{t} \beta(\tau)\mathop{}\!\mathrm{d} \tau}\right).
    $$

-   **先验分布：** $p_{\mathrm{prior}} := \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$。

由于扰动核是均值和协方差已知的 Gaussian 分布，我们可以应用[式 (D.2.5)](../../appendices/d-supplementary-proofs/02-score-based-perspective.md#eq-gaussian-score)计算其 score 函数。

VP SDE 的经典示例是 DDPM，其中噪声调度 $\beta(t)$ 定义为：

$$
\beta(t) := \beta_{\text{min}} + t(\beta_{\text{max}} - \beta_{\text{min}}), \quad \text{for all } t \in [0,1].
$$

这里，$\beta_{\text{min}}$ 与 $\beta_{\text{max}}$ 是预定义常数。由此，DDPM 可解释为 VP SDE 的离散化，正如[第 4.1.1 节](01-score-sde-principles.md#subsec-scoresde-motivation)所述。

## 4.4.3 （选读）如何推导扰动核 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$？ {#subsec-scoresde-perturbation-kernel}

若前向 SDE [式 (4.1.3)](01-score-sde-principles.md#eq-sde-forward)中的漂移项关于 ${\mathbf{x}}$ 是线性的，形式为

$$
\begin{aligned}
{\mathbf{f}}({\mathbf{x}}, t) = f(t) {\mathbf{x}},
\end{aligned}
$$

其中 $f(t) \in \mathbb{R}$ 为某个随时间变化的标量值函数，则[式 (4.1.3)](01-score-sde-principles.md#eq-sde-forward)成为线性 SDE：

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}(t) = f(t){\mathbf{x}}(t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d}{\mathbf{w}}(t).
$$

即使初始分布 $p_{\mathrm{data}}$ 不是 Gaussian 分布，漂移的线性性仍保证条件过程保持 Gaussian 形式。特别地，对于 $t>0$，转移核具有以下形式：

$$
p_t( {\mathbf{x}}_t | {\mathbf{x}}_0 ) = \mathcal{N} \left( {\mathbf{x}}_t;  {\mathbf{m}}(t),  P(t) \mathbf{I}_D \right),
$$

其中 ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$，而 ${\mathbf{m}}(t) \in \mathbb{R}^D$、$P(t)\in\mathbb{R}_{\ge 0}$ 分别表示给定 ${\mathbf{x}}_0$ 时的条件均值与标量方差，定义为：

$$
{\mathbf{m}}(t) = \mathbb{E} \left[{\mathbf{x}}_t | {\mathbf{x}}(0) = {\mathbf{x}}_0\right], \qquad
P(t) \mathbf{I}_D = \operatorname{Cov} \left[{\mathbf{x}}_t | {\mathbf{x}}(0) = {\mathbf{x}}_0\right].
$$

这些一阶和二阶矩按以下 ODE 演化（[Särkkä and Solin, 2019](../../references.md#ref-sarkka2019applied)）：

<span id="eq-mean-var-evolution"></span>

$$
\begin{aligned}
    \frac{\mathop{}\!\mathrm{d} {\mathbf{m}}(t)}{\mathop{}\!\mathrm{d} t} &= f(t) {\mathbf{m}}(t), \\
    \frac{\mathop{}\!\mathrm{d} P(t)}{\mathop{}\!\mathrm{d} t} &= 2f(t) P(t) + g^2(t),
\end{aligned}
\tag{4.4.3}
$$

这里要求初始均值 ${\mathbf{m}}(0)$ 与方差 $P(0)$ 有限。

由于两个 ODE 都是线性的，可以通过积分因子法得到闭式解。给定初始条件 ${\mathbf{x}}_0$，均值与方差的演化为

<span id="eq-mean-var-solution"></span>

$$
\begin{aligned}
{\mathbf{m}}(t) = \mathcal{E}(0  \to  t) {\mathbf{x}}_0,
\qquad
P(t) = \int_{0}^{t} \mathcal{E}^{2}(s  \to  t)  g(s)^{2} \mathop{}\!\mathrm{d} s,
\end{aligned}
\tag{4.4.4}
$$

其中 ${\mathbf{m}}(0)={\mathbf{x}}_0$ 且 $P(0)=0$。这里 $\mathcal{E}(s \to t)$ 表示指数积分因子

$$
\mathcal{E}(s  \to  t) := \exp \left(\int_{s}^{t} f(u) \mathop{}\!\mathrm{d} u\right),
$$

它刻画从时刻 $s$ 到 $t$ 漂移的累积影响。因此，转移核 $p_t({\mathbf{x}}_t | {\mathbf{x}}_0)$ 也具有闭式表达式。

关于 $p_t({\mathbf{x}}_t |{\mathbf{x}}_0)$ 的条件协方差为各向同性的证明，即 $\operatorname{Cov}[{\mathbf{x}}_t | {\mathbf{x}}_0] = P(t) \mathbf I_D$，这里使用 $D$ 维、具有独立坐标且扩散为 $g(t)\mathbf I_D$ 的 Wiener 过程，以及[式 (4.4.3)](04-sde-instantiations.md#eq-mean-var-evolution)的推导，我们放在[第 C.1.5 节](../../appendices/c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-linear-sde)，其中使用 Itô 微积分。

**示例：VE SDE 的转移核**

在 VE SDE 的特殊情形下，${\mathbf{f}} \equiv 0$ 且 $g(t) = \sqrt{\frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t}}$，SDE 解的均值与协方差按如下方式演化。

**均值。**

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{m}}(t)}{\mathop{}\!\mathrm{d} t} = 0,
\quad \text{with} \quad {\mathbf{m}}(0) = {\mathbf{x}}_0
 \Longrightarrow
{\mathbf{m}}(t) = {\mathbf{x}}_0.
$$

**方差。**

$$
\frac{\mathop{}\!\mathrm{d} P(t)}{\mathop{}\!\mathrm{d} t} = \frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t},
\quad \text{with} \quad P(0) = 0
 \Longrightarrow
P(t) = \sigma^2(t) - \sigma^2(0).
$$

因此，

$$
p_t({\mathbf{x}}_t | {\mathbf{x}}_0)=\mathcal{N} \left({\mathbf{x}}_t;  {\mathbf{x}}_0,  \big(\sigma^2(t)-\sigma^2(0)\big)\mathbf{I}_D\right).
$$

**示例：VP SDE 的转移核**

对于漂移为 ${\mathbf{f}}({\mathbf{x}}, t) = -\tfrac{1}{2}\beta(t) {\mathbf{x}}$、扩散为 $g(t) = \sqrt{\beta(t)}$ 的 VP SDE：

**均值 ${\mathbf{m}}(t)$。**

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{m}}}{\mathop{}\!\mathrm{d} t}
= -\frac{1}{2}\beta(t) {\mathbf{m}}(t),
\quad
B(t) \coloneqq \int_{0}^{t} \beta(s) \mathop{}\!\mathrm{d} s,
\quad
{\mathbf{m}}(t) = e^{-\frac{1}{2}B(t)} {\mathbf{x}}_0.
$$

**方差 $P(t)$。** 方差满足

$$
\frac{\mathop{}\!\mathrm{d} P}{\mathop{}\!\mathrm{d} t} = -\beta(t) P(t) + \beta(t).
$$

应用积分因子 $e^{B(t)}$，其中 $B(t) = \int_0^t \beta(s) \mathop{}\!\mathrm{d} s$，得到

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \left[P(t)e^{B(t)}\right] = \beta(t) e^{B(t)}.
$$

对两边积分，得到

$$
P(t) = 1 - e^{-B(t)}.
$$

因此，协方差是各向同性的，且

$$
{\mathbf{P}}(t) = P(t) \mathbf I_D = \bigl(1 - e^{-B(t)}\bigr)\mathbf I_D.
$$

**最终的闭式转移核。**

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
= \mathcal{N} \left(
   {\mathbf{x}}_t;
   \underbrace{e^{-\frac{1}{2} B(t)} {\mathbf{x}}_0}_{{\mathbf{m}}(t)},
   \underbrace{\big(1 - e^{-B(t)}\big)\mathbf{I}_D}_{P(t)\mathbf{I}_D}
\right),
\quad
B(t) = \int_0^t \beta(s) \mathop{}\!\mathrm{d} s.
$$

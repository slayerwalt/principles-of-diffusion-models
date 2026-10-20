# 4.5 通过累积扰动核的直接视角 {#sec-variational-perspective}

介绍 DDPM 与 Score SDE 时，通常先指定噪声如何在时间上局部添加：DDPM 通过离散前向转移 $p({\mathbf{x}}_t|{\mathbf{x}}_{t-\Delta t})$，Score SDE 则通过连续时间 SDE。然而，对于训练，一个特别有用的对象是*累积扰动核*

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0),
$$

它直接以原始数据样本为条件，描述时刻 $t$ 的分布。这恰好是在 DDPM 和连续时间去噪目标[式 (2.2.8)](../02-variational-perspective/02-ddpm.md#eq-ddpm-mean-loss)与[式 (4.3.1)](03-training-and-sampling.md#eq-conti-dsm-loss-real)中自然出现的形式。

这带来一种互补视角：不必从局部噪声注入开始再推导 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$，而可以直接指定累积扰动核。对于下文考虑的线性 Gaussian 过程，一个允许的累积核决定相应的线性 SDE，反之亦然。这一视角明确给出每个时刻的信号量和噪声量，直接联系到实际训练目标，也将成为通向[第 5 章](../05-flow-based-perspective/index.md#ch-flow-based)流匹配概率路径视角的有用桥梁。

<span id="fig-vdm-sde"></span>

![增量式线性 SDE 与累积扰动核描述同一个仿射前向过程](../../../en/assets/chapter-04/vdm-sde.png)

**图 4.8：** **同一个仿射前向过程的两种视角。** 对于允许的仿射扩散调度，通过连续时间线性 SDE 增量式地注入噪声，与直接指定[式 (4.5.1)](05-accumulated-perturbation-kernels.md#eq-forward-kernel)的累积扰动核，描述相同的条件分布 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$。

来源：作者绘制。

**一般仿射前向过程 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$。** 首先定义累积前向扰动核

<span id="eq-forward-kernel"></span>

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
:=
\mathcal{N}\bigl(
{\mathbf{x}}_t;
\alpha_t{\mathbf{x}}_0,
\sigma_t^2\mathbf{I}
\bigr),
\tag{4.5.1}
$$

其中 ${\mathbf{x}}_0\sim p_{\mathrm{data}}$，$\alpha_t$ 与 $\sigma_t$ 是 $t\in[0,T]$ 的非负标量函数。在标准前向扩散设定中，取

$$
\alpha_0=1,
\qquad
\sigma_0=0,
$$

使过程精确地从数据出发，并在使用对应比值时，假设 $\alpha_t>0$ 与 $\sigma_t>0$ 对 $t\in(0,T]$ 成立。

等价地，[式 (4.5.1)](05-accumulated-perturbation-kernels.md#eq-forward-kernel)中的样本可写作

$$
{\mathbf{x}}_t
=
\alpha_t{\mathbf{x}}_0
+
\sigma_t\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

这一简单形式包含几种熟悉的概率路径：

-   **VE（NCSN）：** $\alpha_t\equiv1$，其中噪声尺度 $\sigma_t$ 随时间增加；

-   **VP（DDPM）：** $\alpha_t^2+\sigma_t^2=1$；

-   **线性 Gaussian 插值：** $\alpha_t=1-t$ 且 $\sigma_t=t$，定义于 $t\in[0,1]$ 上。当采用从数据到噪声的时间约定时，这对应流匹配与 Rectified Flow 中常用的线性路径。

## 4.5.1 与 Score SDE 的联系 {#connection-to-score-sde}

对于 Score SDE，先指定累积扰动核 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 再恢复相应 SDE 系数，通常比从漂移项和扩散项出发、求解矩的 ODE 更直观（见[第 4.4.3 节](04-sde-instantiations.md#subsec-scoresde-perturbation-kernel)）。

对于允许的仿射调度，相应的前向 SDE 具有[式 (4.4.2)](04-sde-instantiations.md#eq-forward-linear-sde)中关于 ${\mathbf{x}}$ 线性的形式：

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}}(t)
    =
    \underbrace{f(t){\mathbf{x}}(t)}_{{\mathbf{f}}({\mathbf{x}}(t),t)}
    \mathop{}\!\mathrm{d} t
    +
    g(t)\mathop{}\!\mathrm{d}{\mathbf{w}}(t),
\end{aligned}
$$

其中 $f,g\colon[0,T]\rightarrow\mathbb{R}$ 为时间的实值函数。

系数 $f(t)$ 与 $g(t)$ 可以从 $\alpha_t$ 与 $\sigma_t$ 解析地恢复。并非任意选择的 $\alpha_t$ 与 $\sigma_t$ 都能定义有效扩散：其隐含的扩散方差 $g^2(t)$ 必须保持非负。等价地，在线性 Gaussian 设定中，信噪比 $\alpha_t^2/\sigma_t^2$ 不能随时间增加。以下引理精确表述这一关系。

<span id="forward-sde"></span>

**引理 4.5.1：前向扰动核 $\Leftrightarrow$ 线性 SDE**

假设 $\alpha_0=1$、$\sigma_0=0$，并且 $\alpha_t>0$ 与 $\sigma_t>0$ 对于 $t\in(0,T]$ 可微。定义 $\lambda_t:=\log\tfrac{\alpha_t}{\sigma_t}$。

给定前向扰动核

$$
{\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}},
\qquad
{\boldsymbol{\epsilon}}\sim\mathcal N(\mathbf 0,{\mathbf{I}}),
$$

存在相应的线性 SDE

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}(t)=f(t){\mathbf{x}}(t)\mathop{}\!\mathrm{d} t+g(t)\mathop{}\!\mathrm{d}{\mathbf{w}}(t)
$$

具有相同的条件核 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0) =\mathcal N({\mathbf{x}}_t;\alpha_t{\mathbf{x}}_0,\sigma_t^2{\mathbf{I}})$，当且仅当系数

<span id="eq-kernel-sde-equiv"></span>

$$
\begin{aligned}
f(t)
&=\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\log\alpha_t,\\
g^2(t)
&=\frac{\mathop{}\!\mathrm{d}\sigma_t^2}{\mathop{}\!\mathrm{d} t}
-2\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\log\alpha_t\,\sigma_t^2
=-2\sigma_t^2\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\lambda_t
\end{aligned}
\tag{4.5.2}
$$

满足 $g^2(t)\geq0$，适用于 $t\in(0,T]$。等价地，$\lambda_t$ 必须单调不增。

反之，若线性 SDE 的条件核为 $\mathcal N({\mathbf{x}}_t;\alpha_t{\mathbf{x}}_0,\sigma_t^2{\mathbf{I}})$，且满足 $\alpha_t>0$ 与 $\sigma_t>0$，则其系数必然满足[式 (4.5.2)](05-accumulated-perturbation-kernels.md#eq-kernel-sde-equiv)。

**引理证明。**

根据[第 4.4.3 节](04-sde-instantiations.md#subsec-scoresde-perturbation-kernel)，将均值和协方差的 ODE ${\mathbf{m}}'(t)=f(t){\mathbf{m}}(t)$ 与 ${\mathbf{P}}'(t)=2f(t){\mathbf{P}}(t)+g^2(t){\mathbf{I}}$，分别与 ${\mathbf{m}}(t)=\alpha_t{\mathbf{x}}_0$ 和 ${\mathbf{P}}(t)=\sigma_t^2{\mathbf{I}}$ 匹配，即可得到结果。

**说明。**

要使终端条件分布与 ${\mathbf{x}}_0$ 无关，且等于指定的 Gaussian 先验，需要 $\alpha_T=0$，并且 $\sigma_T^2$ 等于先验方差。由于

$$
\alpha_t
=
\exp\Big(\int_0^t f(u)\mathop{}\!\mathrm{d} u\Big),
$$

若要在有限终端时刻实现 $\alpha_T=0$，则需要

$$
\int_0^T f(u)\mathop{}\!\mathrm{d} u=-\infty.
$$

因此，漂移引起的收缩必须在终端时刻附近变得奇异；特别地，$f$ 不可能在 $[0,T]$ 上保持有界。

扩散系数与指定方差之间的关系为

$$
g^2(t)
=
(\sigma_t^2)'
-
2\frac{\alpha_t'}{\alpha_t}\sigma_t^2.
$$

对于通常的正则调度，$\sigma_t^2$ 趋于有限非零的终端方差，$(\sigma_t^2)'$ 保持有界，且 $\alpha_t'/\alpha_t\to-\infty$，此时还有

$$
g^2(t)\to\infty
\qquad\text{as }t\to T.
$$

更一般地，$g(t)$ 的具体行为取决于所选调度。

若 $f$ 与 $g$ 在有限区间 $[0,T]$ 上保持有界，则 $\alpha_T>0$，因而无法在有限 $T$ 时精确遗忘。实践中，因此会选择一个终端时刻，使对 ${\mathbf{x}}_0$ 的残余依赖可忽略，且 $p_T\approx p_{\mathrm{prior}}$；或者采用无限时间形式，使 $\alpha_t\to0$ 渐近成立。

根据上述引理，通过系数为 $f(t)$ 与 $g(t)$ 的线性 SDE 指定增量噪声注入，在数学上等价于通过参数 $\alpha_t$ 与 $\sigma_t$ 定义扰动核。扩散模型文献中常交替使用这两种视角。因此，我们得到：

<span id="observation-4-5-1"></span>

**观察 4.5.1**

对于有效的仿射扩散调度，两种描述等价：既可以指定累积扰动核 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$，以 $\alpha_t$ 与 $\sigma_t$ 为参数，也可以通过 $f(t)$ 与 $g(t)$ 指定相应的线性 SDE。

## 4.5.2 与基于变分的扩散模型的联系 {#subsec-connection-vdm}

重新考察 DDPM 中通过 Bayes 法则得到的一个核心恒等式：

<span id="eq-ddpm-reverse-condition-kernel"></span>

$$
\begin{aligned}
p({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t, {\mathbf{x}}) = p({\mathbf{x}}_t|{\mathbf{x}}_{t-\Delta t}) \cdot \frac{p_{t-\Delta t}({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}})}{p_t({\mathbf{x}}_t|{\mathbf{x}})},
\end{aligned}
\tag{4.5.3}
$$

它适用于任意 ${\mathbf{x}}$（通常 ${\mathbf{x}}\sim p_{\mathrm{data}}$）。逆向条件分布 $p({\mathbf{x}}_{t-\Delta t} | {\mathbf{x}}_t, {\mathbf{x}})$ 对建模至关重要，使训练目标可计算，并支持高效采样。

尽管 DDPM 通常先定义增量核 $p({\mathbf{x}}_t|{\mathbf{x}}_{t-\Delta t})$，累积转移 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 往往提供更具可解释性、更实用的形式，尤其适用于先验与损失设计。

**推导转移核。** 现在将其扩展到连续时间。令 $0 \leq t < s \leq T$ 为两个连续时间点。给定扰动核 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$，可以通过应用[式 (4.5.3)](05-accumulated-perturbation-kernels.md#eq-ddpm-reverse-condition-kernel)[^1]，计算逆向条件分布 $p({\mathbf{x}}_t|{\mathbf{x}}_s, {\mathbf{x}})$，适用于任意 ${\mathbf{x}}$，并以 $p({\mathbf{x}}_s|{\mathbf{x}}_t)$ 为中间前向核。以下引理总结这一推导，将[引理 2.2.2](../02-variational-perspective/02-ddpm.md#ddpm-reverse-kernel)扩展到无须假设 $\alpha_t^2 + \sigma_t^2 = 1$ 的情形。

<span id="reverse-kernel"></span>

**引理 4.5.2：逆向条件转移核**

令 $0 \leq t < s \leq T$。逆向*条件*转移核为：

$$
p({\mathbf{x}}_t|{\mathbf{x}}_s, {\mathbf{x}}) = \mathcal{N}\left({\mathbf{x}}_t; \boldsymbol{\mu}({\mathbf{x}}_s, {\mathbf{x}}; s, t), \sigma^2(s, t) {\mathbf{I}}\right),
$$

其中

<span id="eq-p-backward-mean-var"></span>

$$
\begin{aligned}
    \boldsymbol{\mu}({\mathbf{x}}_s, {\mathbf{x}}; s, t) := \frac{\alpha_{s|t} \sigma_t^2}{\sigma_s^2} {\mathbf{x}}_s + \frac{\alpha_t \sigma^2_{s|t}}{\sigma_s^2} {\mathbf{x}}, \quad
    \sigma^2(s, t) := \sigma^2_{s|t} \frac{\sigma_t^2}{\sigma_s^2}.
\end{aligned}
\tag{4.5.4}
$$

这里，$\alpha_{s|t}$ 与 $\sigma_{s|t}$ 定义为：

$$
\alpha_{s|t} := \frac{\alpha_s}{\alpha_t}, \qquad \sigma^2_{s|t} := \sigma_s^2 - \alpha_{s|t}^2 \sigma_t^2.
$$

**引理证明。**

首先计算前向转移核：

<span id="eq-zt-given-zs"></span>

$$
p({\mathbf{x}}_s|{\mathbf{x}}_t) = \mathcal{N}\left({\mathbf{x}}_s; \alpha_{s|t} {\mathbf{x}}_t, \sigma^2_{s|t} {\mathbf{I}}\right).
\tag{4.5.5}
$$

随后由 Bayes 法则得到逆向核；由于所有涉及的分布都是 Gaussian，可通过直接计算推导结果。更多细节见（[Kingma et al., 2021](../../references.md#ref-kingma2021variational)）的附录 A。

尽管 $p({\mathbf{x}}_{t + \Delta t}|{\mathbf{x}}_t)$ 与 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 在理论上等价，$p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 往往处于更核心的位置。[式 (4.5.5)](05-accumulated-perturbation-kernels.md#eq-zt-given-zs)中的逐步转移主要用于获得闭式逆向核。因此，近期工作（[Kingma et al., 2021](../../references.md#ref-kingma2021variational)）更倾向于直接指定 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$，因为这种方式清晰且具有可解释性。

**逆向过程的建模、训练与采样。** [式 (2.2.13)](../02-variational-perspective/02-ddpm.md#eq-ddpm-elbo)中的训练目标 ELBO，以及[第 2.2 节](../02-variational-perspective/02-ddpm.md#sec-ddpm)介绍的建模框架，仍适用于我们的推广设定。为清楚起见，遵循 [Kingma et al. (2021)](../../references.md#ref-kingma2021variational)，采用 ${\mathbf{x}}$ 预测形式，记为 ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s)$。不过，由于以下关系（如[式 (2.2.12)](../02-variational-perspective/02-ddpm.md#eq-ddpm-clean-noise)所示），等价的 ${\boldsymbol{\epsilon}}$ 预测视角也成立，其表示为 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s)$：

$$
{\mathbf{x}}_s = \alpha_s {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s) + \sigma_s \boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s),\quad \text{for any given } {\mathbf{x}}_s \sim q_s.
$$

**建模与扩散损失 $\mathcal{L}_{\text{diffusion}}$。** 与 DDPM 类似，[式 (4.5.4)](05-accumulated-perturbation-kernels.md#eq-p-backward-mean-var)中的条件分布 $p({\mathbf{x}}_t|{\mathbf{x}}_s, {\mathbf{x}})$，启发我们将干净信号 ${\mathbf{x}}$ 替换为可学习预测器 ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s)$，得到以下形式的参数化逆向模型：

<span id="eq-general-ddpm-sampling"></span>

$$
\begin{aligned}
p_{\boldsymbol{\phi}}({\mathbf{x}}_t | {\mathbf{x}}_s) := \mathcal{N}\left({\mathbf{x}}_t; \boldsymbol{\mu}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s, t), \sigma^2(s,t) \mathbf{I}\right),
\end{aligned}
\tag{4.5.6}
$$

其均值参数化为：

$$
\boldsymbol{\mu}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s, t) = \frac{\alpha_{s|t} \sigma_t^2}{\sigma_s^2} {\mathbf{x}}_s + \frac{\alpha_t \sigma_{s|t}^2}{\sigma_s^2} {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s).
$$

给定[式 (4.5.1)](05-accumulated-perturbation-kernels.md#eq-forward-kernel)中的前向核，$\mathcal{L}_{\text{diffusion}}({\mathbf{x}}; \boldsymbol{\phi})$ 中的 KL散度可化为加权回归损失：

<span id="eq-snr-kl"></span>

$$
\begin{aligned}
    \mathcal{D}_{\text{KL}}\big(p({\mathbf{x}}_t|{\mathbf{x}}_s, {\mathbf{x}}_0)  \| p_{\boldsymbol{\phi}}({\mathbf{x}}_t|{\mathbf{x}}_s)\big)
    &= \frac{1}{2\sigma^2(s,t)} \left\| \boldsymbol{\mu}({\mathbf{x}}_s, {\mathbf{x}}_0; s,t) - \boldsymbol{\mu}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s, t) \right\|_2^2 \\
    &= \frac{1}{2} \big(\text{SNR}(t) - \text{SNR}(s)\big) \left\| {\mathbf{x}}_0 - {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s) \right\|_2^2,
\end{aligned}
\tag{4.5.7}
$$

其中 ${\mathbf{x}}_s = \alpha_s{\mathbf{x}}_0 + \sigma_s\boldsymbol{\epsilon}$，且 ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$、$\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$，而 $\text{SNR}(s) := \alpha_s^2 / \sigma_s^2$ 表示时刻 $s$ 的信噪比。

**说明。**

在（[Kingma et al., 2021](../../references.md#ref-kingma2021variational)）中，作者研究[式 (4.5.7)](05-accumulated-perturbation-kernels.md#eq-snr-kl)在 $t \to s$ 时的连续时间极限，得到：

$$
\begin{aligned}
\mathcal{L}_{\text{VDM}}^\infty({\mathbf{x}}_0)
= -\frac{1}{2}  \mathbb{E}_{s, \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})}  \text{SNR}'(s)  \big\| {\mathbf{x}}_0 - {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_s, s) \big\|_2^2.
\end{aligned}
$$

这一设定还引入了可学习的噪声调度，并且可推广到连续数据之外，不过这些扩展超出了当前讨论范围。

**采样。** 使用[式 (4.5.6)](05-accumulated-perturbation-kernels.md#eq-general-ddpm-sampling)的参数化核，采样过程与 DDPM 类似：

<span id="eq-general-ddpm-sampling-discrete"></span>

$$
\begin{aligned}
{\mathbf{x}}_t = \underbrace{\frac{\alpha_{s|t} \sigma_t^2}{\sigma_s^2} {\mathbf{x}}_s + \frac{\alpha_t \sigma_{s|t}^2}{\sigma_s^2} {\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s, s)}_{\boldsymbol{\mu}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s, t, s)} + \sigma_{s|t} \frac{\sigma_t}{\sigma_s} \boldsymbol{\epsilon}_s,\quad  \boldsymbol{\epsilon}_s \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}).
\end{aligned}
\tag{4.5.8}
$$

[^1]: 将 $s$ 视为一般的较早时刻，这一恒等式自然扩展到连续时间。
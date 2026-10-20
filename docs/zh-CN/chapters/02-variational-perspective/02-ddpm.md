# 2.2 变分视角：DDPM {#sec-ddpm}

Denoising Diffusion Probabilistic Models（DDPM）([Sohl-Dickstein et al., 2015](../../references.md#ref-sohl2015deep); [Ho et al., 2020](../../references.md#ref-ho2020denoising)) 是扩散建模的基石之一。从概念上看，它们与 VAE 和 HVAE 一样，都建立在变分框架之内。不过，DDPM 引入了一项巧妙的变化，解决了此前模型面临的部分挑战。

DDPM 的核心涉及两个不同的随机过程：

-   **前向过程（固定编码器）：** 该过程通过转移核 $p(\mathbf{x}_{i}|\mathbf{x}_{i-1})$，在多个步骤中注入 Gaussian 噪声，逐渐扰动数据。数据演变为各向同性 Gaussian 分布，实际上变成了纯噪声。这意味着编码器是固定的，无须学习。

-   **逆向去噪过程（可学习解码器）：** 在这一过程中，神经网络通过参数化分布 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_{i})$ 学习逆转噪声扰动。该过程从纯噪声出发，通过迭代去噪生成逼真的样本。每一个去噪步骤，都比 VAE 通常尝试的从头生成一个完整样本更容易处理。

通过固定编码器，并将学习集中在逐步生成的轨迹上，DDPM 获得了出色的稳定性和表达能力。

<span id="fig-vdm-grap"></span>

![DDPM 固定的前向加噪链与学习得到的逆向去噪链](../../../en/assets/chapter-02/ddpm-graph.png)

**图 2.3：** **DDPM 示意图。** 它由一个固定的前向过程（灰色）和一个学习得到的逆向过程组成：前者逐渐向数据加入 Gaussian 噪声，后者逐步去噪以生成新样本。

来源：由本书作者绘制。

本节重点介绍 DDPM。更广泛的讨论留待[第 4.5 节](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#sec-variational-perspective)，届时我们将给出一个更一般、更灵活的框架。

## 2.2.1 前向过程（固定编码器） {#forward-process}

在 DDPM 中，前向过程是一项固定且不可训练的操作，起到编码器的作用。它通过多个步骤添加噪声，逐渐扰动原始数据，最终将其转化为一个简单的先验分布 $p_{\text{prior}} := \mathcal{N}(\mathbf{0}, \mathbf{I})$。[图 2.3](02-ddpm.md#fig-vdm-grap) 中的前向链和[图 2.4](02-ddpm.md#fig-vdm-forward) 展示了这一转化。

<span id="fig-vdm-forward"></span>

![DDPM 前向过程通过 Gaussian 噪声逐渐扰动干净图像](../../../en/assets/chapter-02/ddpm-forward.png)

**图 2.4：** DDPM 前向过程示意图：逐步加入 Gaussian 噪声，将数据样本扰动为纯噪声。

来源：由本书作者绘制。

下面对这一逐步退化过程进行形式化描述：

**固定的 Gaussian 转移。** 前向过程中的每一步都由一个固定的 Gaussian 转移核控制[^1]：

$$
p({\mathbf{x}}_i|{\mathbf{x}}_{i-1}) := \mathcal{N}({\mathbf{x}}_i; \sqrt{1 - \beta_i^2}   {\mathbf{x}}_{i-1}, \beta_i^2 {\mathbf{I}}).
$$

这里，过程从 ${\mathbf{x}}_{0}$ 开始，它表示从真实数据分布 $p_{\text{data}}$ 中抽取的一个样本。序列 $\{\beta_i\}_{i=1}^L$ 表示预先确定的、单调递增的噪声调度，其中每个 $\beta_i \in (0, 1)$ 控制第 $i$ 步注入的 Gaussian 噪声的方差。为方便起见，定义 $\alpha_i := \sqrt{1 - \beta_i^2}$。这一数学定义与下面的直观迭代更新完全等价：

$$
{\mathbf{x}}_i = \alpha_i {\mathbf{x}}_{i-1} + \beta_i \boldsymbol{\epsilon}_i,
$$

其中，$\boldsymbol{\epsilon}_i \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$ 独立同分布。这意味着，在每一步 $i$，我们将上一状态 ${\mathbf{x}}_{i-1}$ 乘以 $\alpha_i$ 进行缩小，再加入由 $\beta_i$ 缩放的、强度受控的 Gaussian 噪声。

**扰动核与先验分布。** 通过递归应用转移核，可以得到第 $i$ 步含噪样本在给定原始数据 ${\mathbf{x}}_0$ 时的分布的闭式表达式：

$$
p_i({\mathbf{x}}_i|{\mathbf{x}}_0) = \mathcal{N}\left({\mathbf{x}}_i; \bar{\alpha}_i {\mathbf{x}}_0, (1 - \bar{\alpha}_i^2) {\mathbf{I}} \right),
$$

其中

$$
\bar{\alpha}_i := \prod_{k=1}^i \sqrt{1-\beta_k^2} = \prod_{k=1}^i \alpha_k.
$$

这意味着，样本 ${\mathbf{x}}_i$ 可以直接由 ${\mathbf{x}}$ 按下式生成：

<span id="eq-ddpm-forward"></span>

$$
\begin{aligned}
{\mathbf{x}}_i = \bar{\alpha}_i {\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_i^2}   \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}).
\end{aligned}
\tag{2.2.1}
$$

令噪声调度 $\{\beta_i\}_{i=1}^L$ 为递增序列，则前向过程的边缘分布满足如下收敛关系：

$$
p_L(\mathbf{x}_L | {\mathbf{x}}_0) \longrightarrow \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \text{as } L \to \infty,
$$

因此可以将先验分布选为

$$
p_{\text{prior}} := \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

它不依赖于数据 ${\mathbf{x}}_0$。

## 2.2.2 逆向去噪过程（可学习解码器） {#reverse-denoising-process}

DDPM 的核心在于，能够*逆转*前向扩散过程施加的受控退化。从纯粹、无结构的噪声 $\mathbf{x}_L \sim p_{\text{prior}}$ 出发，目标是逐步对这种随机状态*去噪*，直到得到结构连贯且有意义的数据样本。这一逆向生成通过一条 Markov 链进行，如[图 2.5](02-ddpm.md#fig-ddpm-sampling-formal-rev) 所示。

<span id="fig-ddpm-sampling-formal-rev"></span>

![DDPM 从 Gaussian 噪声到生成图像的逆向去噪链](../../../en/assets/chapter-02/ddpm-reverse.png)

**图 2.5：** **DDPM 逆向（去噪）过程示意图。** 从噪声 $\mathbf{x}_L \sim p_{\mathrm{prior}}$ 出发，模型依次采样 $\mathbf{x}_{i-1} \sim p(\mathbf{x}_{i-1}|\mathbf{x}_i)$，其中 $i=L,\ldots,1$，从而得到新生成的数据 $\mathbf{x}_0$。oracle 转移 $p(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 未知，因此我们的目标是对它进行近似。

来源：由本书作者绘制。

由此，根本挑战以及引导 DDPM 发展的核心问题变成：

<span id="question-2-2-1"></span>

**问题 2.2.1**

我们能否精确计算这些逆向转移核 $p(\mathbf{x}_{i-1}|\mathbf{x}_i)$，或者至少对它们进行有效近似，尤其是在 $\mathbf{x}_i \sim p_i(\mathbf{x}_i)$ 的分布很复杂时？

DDPM 原始论文从证据下界（ELBO）的复杂数学推导切入，相关详细讨论见[第 2.2.5 节](02-ddpm.md#subsec-ddpm-elbo)。这里先从更直观的视角介绍训练目标：借助条件概率，得到可计算的表述。

**概述：建模与训练目标。** 为了实现生成过程，我们的目标是近似未知的真实逆向转移核 $p(\mathbf{x}_{i-1}|\mathbf{x}_i)$。为此，引入一个可学习的参数化模型 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i)$，并通过训练使下面的期望 KL散度最小：

<span id="eq-marginal-kl-matching"></span>

$$
\begin{aligned}
\mathbb{E}_{p_i(\mathbf{x}_i)} \left[
    \mathcal{D}_{\mathrm{KL}}\big(p(\mathbf{x}_{i-1}|\mathbf{x}_i) \| p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i)\big)
    \right].
\end{aligned}
\tag{2.2.2}
$$

然而，直接计算目标分布 $p(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 很困难。根据 Bayes 定理，需要计算：

$$
p(\mathbf{x}_{i-1}|\mathbf{x}_i) = p(\mathbf{x}_i|\mathbf{x}_{i-1}) \underbrace{\frac{p_{i-1}(\mathbf{x}_{i-1})}{p_i(\mathbf{x}_i)}}_{\text{intractable}}.
$$

边缘分布 $p_i({\mathbf{x}}_i)$ 和 $p_{i-1}({\mathbf{x}}_{i-1})$ 是关于未知数据分布 $p_{\mathrm{data}}$ 的期望，其中：

$$
p_i({\mathbf{x}}_i) = \int p_i({\mathbf{x}}_i | {\mathbf{x}}_0) p_{\mathrm{data}}({\mathbf{x}}_0) \mathrm{d}{\mathbf{x}}_0,
$$

$p_{i-1}({\mathbf{x}}_{i-1})$ 的表达式与之类似。由于 $p_{\mathrm{data}}$ 未知，这些积分无法以闭式求值，最多只能利用样本进行近似，因此实践中无法获得精确的密度值。

<span id="fig-conditional-trick"></span>

![DDPM 中的边缘逆向混合分布与可计算的条件 Gaussian 分布](../../../en/assets/chapter-02/ddpm-conditioning.png)

**图 2.6：** **DDPM 中的条件化技巧。** **（左）** 对于一个固定的含噪样本 ${\mathbf{x}}_i$，逆向分布 $p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)$ 对所有可能的干净样本 ${\mathbf{x}}_0$ 取平均，这些样本都可能产生 ${\mathbf{x}}_i$，因此它可能是一个复杂的混合分布。**（右）** 但在训练时，干净样本 ${\mathbf{x}}_0$ 是已知的，它被用来生成 ${\mathbf{x}}_i$。以这个样本为条件，就得到 $p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i,{\mathbf{x}}_0)$，这是一个均值和方差具有闭式表达式的 Gaussian 分布。因此，DDPM 可以将这一可计算的条件分布用作简单的回归目标。虽然每个训练样例只以一个干净样本为条件，但对所有样本的条件训练目标取平均，与直接学习边缘逆向分布具有相同的最优模型，见[定理 2.2.1](02-ddpm.md#thm-equiv-marginal-kl)。因此，DDPM 无须直接对边缘混合分布所产生的复杂、通常难以计算的目标进行回归，也能学到相同的最优逆向预测。

来源：由本书作者借助 AI 辅助编程制作。

**通过条件化解决计算困难。** DDPM 中的一个核心思路解决了这一计算困难：以干净数据样本 $\mathbf{x}_0$ 为条件，考虑逆向转移。这一巧妙而有效的步骤，将难以计算的转移核变为数学上可计算的形式：

$$
p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x}_0) = p(\mathbf{x}_i|\mathbf{x}_{i-1}) \frac{p(\mathbf{x}_{i-1}|\mathbf{x}_0)}{p_i(\mathbf{x}_i|\mathbf{x}_0)}.
$$

这种计算可行性来自前向过程的两个关键性质：一是 Markov 性，即 $p(\mathbf{x}_i|\mathbf{x}_{i-1}, \mathbf{x}_0) = p(\mathbf{x}_i|\mathbf{x}_{i-1})$；二是所涉及的分布均为 Gaussian 分布。因此，$p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x}_0)$ 本身也是 Gaussian 分布，并具有闭式表达式（见下文[式（2.2.4）](02-ddpm.md#eq-ddpm-reverse-mean)）。[图 2.6](02-ddpm.md#fig-conditional-trick) 展示了这一点。关键在于，这种条件化策略使我们能够推导出一个可计算的目标，在优化作用上与[式（2.2.2）](02-ddpm.md#eq-marginal-kl-matching) 中看似难以计算的边缘 KL散度等价。

<span id="thm-equiv-marginal-kl"></span>

**定理 2.2.1：边缘与条件 KL散度最小化的等价性**

以下等式成立：

<span id="eq-kl-matching"></span>

$$
\begin{aligned}
    &\mathbb{E}_{p_i({\mathbf{x}}_i)}\left[
        \mathcal{D}_{\mathrm{KL}}\big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i) \| p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right] \\
    ={}&
    \mathbb{E}_{p_{\mathrm{data}}({\mathbf{x}}_0)}
    \mathbb{E}_{p_i({\mathbf{x}}_i|{\mathbf{x}}_0)} \left[
        \mathcal{D}_{\mathrm{KL}}\big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0) \| p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right] + C.
\end{aligned}
\tag{2.2.3}
$$

其中，$C$ 是与 $\boldsymbol{\phi}$ 无关的常数。此外，[式（2.2.3）](02-ddpm.md#eq-kl-matching) 的最小化解满足

$$
p^*(\mathbf{x}_{i-1}|\mathbf{x}_i) = \mathbb{E}_{p(\mathbf{x}_0|\mathbf{x}_i)} \big[p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x}_0)\big] = p(\mathbf{x}_{i-1}|\mathbf{x}_i), \quad \mathbf{x}_i \sim p_i.
$$

**定理的证明。**

证明通过展开定义、应用概率链式法则，并使用对数恒等式，将 KL散度的期望改写为条件 KL散度的期望与边缘 KL散度之和。完整推导见[第 D.1.1 节](../../appendices/d-supplementary-proofs/01-variational-perspective.md#subsec-proof-equiv-marginal-kl)。

通过条件化获得可计算目标的这一视角，构成了 DDPM 的基础，也揭示了它与其他有影响力的扩散模型之间的深层共性。我们将在[第 3 章](../03-score-based-perspective/index.md#ch-score-based) 和[第 5 章](../05-flow-based-perspective/index.md#ch-flow-based) 中继续讨论。

它揭示了一种有力的等价关系：最小化边缘分布之间的 KL散度，在数学上等价于最小化特定条件分布之间的 KL散度。后一种表述尤其有用，因为其中关键的条件分布 $p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x})$ 具有便于计算的闭式表达式：

<span id="ddpm-reverse-kernel"></span>

**引理 2.2.2：逆向条件转移核**

$p(\mathbf{x}_{i-1} | \mathbf{x}_i, \mathbf{x})$ 为 Gaussian 分布，其闭式表达式为：

$$
\begin{aligned}
p({\mathbf{x}}_{i-1} | {\mathbf{x}}_i, {\mathbf{x}}) =\mathcal{N}\left({\mathbf{x}}_{i-1}; \boldsymbol{\mu}\left({\mathbf{x}}_i, {\mathbf{x}}, i\right), \sigma^2(i){\mathbf{I}} \right),
\end{aligned}
$$

其中

<span id="eq-ddpm-reverse-mean"></span>

$$
\begin{aligned}
            \boldsymbol{\mu}\left({\mathbf{x}}_i, {\mathbf{x}}, i\right):= \frac{\bar{\alpha}_{i-1}\beta_i^2}{1-\bar{\alpha}_i^2}{\mathbf{x}} + \frac{(1-\bar{\alpha}_{i-1}^2)\alpha_i }{1-\bar{\alpha}_i^2}{\mathbf{x}}_i,
\quad  \sigma^2(i) := \frac{1-\bar{\alpha}_{i-1}^2}{1-\bar{\alpha}_i^2}\beta_i^2.
\end{aligned}
\tag{2.2.4}
$$

后文的[引理 4.5.2](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#reverse-kernel) 将给出一个更一般的公式，其适用范围超出了[式（2.2.1）](02-ddpm.md#eq-ddpm-forward) 所描述的 DDPM 加噪过程。

## 2.2.3 逆向转移核 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 的建模 {#modeling-reverse-transition}

借助[定理 2.2.1](02-ddpm.md#thm-equiv-marginal-kl) 中梯度层面的等价性，以及[引理 2.2.2](02-ddpm.md#ddpm-reverse-kernel) 中逆向条件分布 $p(\mathbf{x}_{i-1}|\mathbf{x}_i, \mathbf{x})$ 的 Gaussian 形式，DDPM 假设每个逆向转移 $p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}| {\mathbf{x}}_i)$ 都是 Gaussian 分布，并将其参数化为

<span id="eq-def-reverse"></span>

$$
\begin{aligned}
p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i) := \mathcal{N}\big(\mathbf{x}_{i-1}; \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}_i, i), \sigma^2(i) \mathbf{I}\big),
\end{aligned}
\tag{2.2.5}
$$

其中，$\boldsymbol{\mu}_{\boldsymbol{\phi}}(\cdot, i) \colon \mathbb{R}^D \to \mathbb{R}^D$ 是可学习的均值函数，$\sigma^2(i) > 0$ 则按[式（2.2.4）](02-ddpm.md#eq-ddpm-reverse-mean) 的定义固定。

为匹配所有层的分布，将关于时间步 $i$ 取平均、以数据 ${\mathbf{x}}_0 \sim p_{\text{data}}$ 为条件的 KL散度记为：

<span id="eq-ddpm-diffusion"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{diffusion}}({\mathbf{x}}_0;\boldsymbol{\phi}) := \sum_{i=2}^L \mathbb{E}_{p_i({\mathbf{x}}_i|{\mathbf{x}}_0)}\left[\mathcal{D}_{\text{KL}}\big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)  \Vert  p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)\right].
\end{aligned}
\tag{2.2.6}
$$

由于这两个分布均为 Gaussian 分布，并采用了[式（2.2.5）](02-ddpm.md#eq-def-reverse) 中定义的参数化方式，该目标具有闭式表达式，可简化为：

<span id="eq-ddpm-diffusion-simp"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{diffusion}}({\mathbf{x}}_0;\boldsymbol{\phi}) = \sum_{i=2}^L \frac{1}{2\sigma^2(i)}  \mathbb{E}_{p_i({\mathbf{x}}_i|{\mathbf{x}}_0)}\left\| \boldsymbol{\mu}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) - \boldsymbol{\mu}({\mathbf{x}}_i, {\mathbf{x}}_0, i) \right\|_2^2 + C,
\end{aligned}
\tag{2.2.7}
$$

其中，$C$ 是与 $\boldsymbol{\phi}$ 无关的常数。对数据分布取平均，并略去不影响优化的常数 $C$，便得到最终的 DDPM 训练目标

<span id="eq-ddpm-mean-loss"></span>

$$
\begin{aligned}
    \mathcal{L}_{\text{DDPM}}(\boldsymbol{\phi}) := \sum_{i=2}^L \frac{1}{2\sigma^2(i)}  \mathbb{E}_{{\mathbf{x}}_0} \mathbb{E}_{p_i({\mathbf{x}}_i|{\mathbf{x}}_0)} \left[ \left\| \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}_i, i) - \boldsymbol{\mu}(\mathbf{x}_i, {\mathbf{x}}_0, i) \right\|_2^2 \right],
\end{aligned}
\tag{2.2.8}
$$

其中 ${\mathbf{x}}_0\sim p_{\mathrm{data}}$。

前向扰动核 $p_i({\mathbf{x}}_i | {\mathbf{x}}_0)$ 在训练目标的计算中发挥着重要作用。它给出了含噪状态 ${\mathbf{x}}_i$ 的分布，该状态由干净样本 ${\mathbf{x}}_0$ 经过 $i$ 个前向步骤得到。由于每个前向转移都是线性且 Gaussian 的，它们的复合也具有闭式的 Gaussian 分布，见[式（2.2.1）](02-ddpm.md#eq-ddpm-forward) 附近的讨论。因此，选定一个时间步 $i$ 后，样本 ${\mathbf{x}}_i \sim p_i({\mathbf{x}}_i | {\mathbf{x}}_0)$ 可以直接从 ${\mathbf{x}}_0$ 出发采样得到，无须先生成中间状态 ${\mathbf{x}}_1,\ldots,{\mathbf{x}}_{i-1}$。这样，每个损失项只需一个干净样本、一个选定的时间步以及一个直接扰动得到的样本即可估计，使随机训练具有较高的计算效率。

## 2.2.4 实践中预测方式与损失的选择 {#subsec-ddpm-prediction}

**${\boldsymbol{\epsilon}}$ 预测。** 在典型的 DDPM 实现中，训练通常不直接采用[式（2.2.8）](02-ddpm.md#eq-ddpm-mean-loss) 中基于*均值预测*参数化的原始损失，而是采用一种等价的重参数化方式，称为 *${\boldsymbol{\epsilon}}$ 预测*（噪声预测）。

回顾 DDPM 的前向过程，含噪样本 ${\mathbf{x}}_i \sim p({\mathbf{x}}_i |{\mathbf{x}}_0)$ 处于噪声水平 $i$，由下式生成：

<span id="eq-ddpm-perturbation"></span>

$$
{\mathbf{x}}_i = \bar{\alpha}_i {\mathbf{x}}_0 + \sqrt{1 - \bar{\alpha}_i^2}   \boldsymbol{\epsilon}, \quad {\mathbf{x}}_0 \sim p_{\text{data}}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
\tag{2.2.9}
$$

利用这一表达式，[式（2.2.4）](02-ddpm.md#eq-ddpm-reverse-mean) 中的逆向均值 $\boldsymbol{\mu}({\mathbf{x}}_i, {\mathbf{x}}_0, i)$ 可以改写为：

$$
\boldsymbol{\mu}({\mathbf{x}}_i, {\mathbf{x}}_0, i) = \frac{1}{\alpha_i} \left( {\mathbf{x}}_i - \frac{1 - \alpha_i^2}{\sqrt{1 - \bar{\alpha}_i^2}} \boldsymbol{\epsilon} \right).
$$

由此，模型均值 $\boldsymbol{\mu}_{\boldsymbol{\phi}}$ 可以用一个直接预测噪声的神经网络 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i)$ 来参数化：

$$
\boldsymbol{\mu}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) = \frac{1}{\alpha_i} \left( {\mathbf{x}}_i - \frac{1 - \alpha_i^2}{\sqrt{1 - \bar{\alpha}_i^2}} \underbrace{\boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i)}_{{\boldsymbol{\epsilon}}\text{-prediction}} \right).
$$

将其代入原始损失，得到预测噪声与真实噪声之间的 $\ell_2$ 误差平方：

$$
\left\| \boldsymbol{\mu}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) - \boldsymbol{\mu}({\mathbf{x}}_i, {\mathbf{x}}_0, i) \right\|_2^2 \propto \left\| \boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) - \boldsymbol{\epsilon} \right\|_2^2,
$$

两者相差一个依赖于 $i$ 的权重因子。直观地说，模型扮演“噪声侦探”的角色，估计前向过程中每一步加入的随机噪声。从受扰动的样本中减去这一估计，会使其更接近干净的原始样本；逐步重复这一操作，就能从纯噪声重建数据。

**${\boldsymbol{\epsilon}}$ 预测的简化损失。** 实践中，通常进一步略去权重项来简化这一表达式，从而得到广泛使用的 DDPM 训练损失：

<span id="eq-ddpm-simple-loss"></span>

$$
\mathcal{L}_{\text{simple}}(\boldsymbol{\phi}) := \mathbb{E}_{i} \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \left\| \boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) - \boldsymbol{\epsilon} \right\|_2^2 \right],
\tag{2.2.10}
$$

其中，${\mathbf{x}}_i = \bar{\alpha}_i {\mathbf{x}}_0 + \sqrt{1 - \bar{\alpha}_i^2} \boldsymbol{\epsilon}$，且 ${\mathbf{x}}_0 \sim p_{\text{data}}$。由于目标噪声在每个时间步 $t$ 都具有单位方差，[式（2.2.10）](02-ddpm.md#eq-ddpm-simple-loss) 中的 $\ell_2$ 损失在所有 $t$ 上保持一致的尺度。这避免了目标值消失或爆炸，也消除了显式损失加权的必要性。

$\mathcal{L}_{\text{DDPM}}$ 与 $\mathcal{L}_{\text{simple}}$ 具有相同的最优解 $\boldsymbol{\epsilon}^*$，因为[式（2.2.10）](02-ddpm.md#eq-ddpm-simple-loss) 本质上归结为一个最小二乘问题（[命题 4.3.1](../04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer) 或[命题 6.3.1](../06-unified-perspective/03-equivalent-parameterizations.md#equi-para) 给出了类似结论）[^2]：

$$
\boldsymbol{\epsilon}^*({\mathbf{x}}_i, i) = \mathbb{E}\left[ \boldsymbol{\epsilon}|{\mathbf{x}}_i \right], \quad {\mathbf{x}}_i \sim p_i.
$$

直观地说，${\boldsymbol{\epsilon}}$ 预测网络 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i)$ 估计的是前向过程中为生成 ${\mathbf{x}}_i$ 而加入的噪声。达到最优时，这一估计等于真实噪声的条件期望，尽管 ${\mathbf{x}}_i$ 并不能唯一确定原始干净样本。

**另一种等价参数化：${\mathbf{x}}$ 预测。** [式（2.2.4）](02-ddpm.md#eq-ddpm-reverse-mean) 提示了另一种等价的参数化方式，称为 *${\mathbf{x}}$ 预测*（干净数据预测）：训练一个神经网络 ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i)$，使其根据给定的含噪输入 ${\mathbf{x}}_i \sim p_i({\mathbf{x}}_i)$，在噪声水平 $i$ 下预测干净（去噪后）的样本。将逆向均值表达式中的真实干净样本 ${\mathbf{x}}_0$ 替换为 ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i)$，得到以下模型：

$$
\begin{aligned}
\boldsymbol{\mu}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) = \frac{\bar{\alpha}_{i-1} \beta_i^2}{1 - \bar{\alpha}_i^2} {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) + \frac{(1 - \bar{\alpha}_{i-1}^2)\alpha_i}{1 - \bar{\alpha}_i^2} {\mathbf{x}}_i.
\end{aligned}
$$

与 ${\boldsymbol{\epsilon}}$ 预测的表述类似，训练目标可写为

$$
\left\| \boldsymbol{\mu}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) - \boldsymbol{\mu}({\mathbf{x}}_i, {\mathbf{x}}_0, i) \right\|_2^2 \propto \left\| {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) - {\mathbf{x}}_0 \right\|_2^2, \quad {\mathbf{x}}_0 \sim p_{\text{data}},
$$

即训练模型预测原始数据样本 ${\mathbf{x}}_0$，输入则为其含噪版本 ${\mathbf{x}}_i$。这一等价关系将[式（2.2.8）](02-ddpm.md#eq-ddpm-mean-loss) 中的均值匹配损失化为

$$
\mathbb{E}_{i} \mathbb{E}_{{\mathbf{x}}_0 \sim p_{\text{data}}} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[\omega_i \left\| {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) - {\mathbf{x}}_0 \right\|_2^2 \right],
$$

其中 $\omega_i$ 是某个权重函数。由于这是一个最小二乘问题，其最优解为（见[命题 4.3.1](../04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer) 或[命题 6.3.1](../06-unified-perspective/03-equivalent-parameterizations.md#equi-para)）

<span id="eq-ddpm-clean-minimizer"></span>

$$
\begin{aligned}
{\mathbf{x}}^*({\mathbf{x}}_i, i) = \mathbb{E}\left[ {\mathbf{x}}_0|{\mathbf{x}}_i \right], \quad {\mathbf{x}}_i \sim p_i,
\end{aligned}
\tag{2.2.11}
$$

即给定含噪观测 ${\mathbf{x}}_i$ 时，模型应预测时间步 $i$ 对应的干净数据的期望。

${\mathbf{x}}$ 预测与 ${\boldsymbol{\epsilon}}$ 预测在数学上是等价的参数化方式，并通过前向过程相联系：

<span id="eq-ddpm-clean-noise"></span>

$$
{\mathbf{x}}_i = \bar{\alpha}_i {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) + \sqrt{1 - \bar{\alpha}_i^2}   \boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i).
\tag{2.2.12}
$$

也就是说，可以预测干净样本 ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i)$，也可以预测噪声 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i)$，使二者按前向加噪过程组合后重现 ${\mathbf{x}}_i$。

## 2.2.5 DDPM 的 ELBO {#subsec-ddpm-elbo}

按[式（2.2.5）](02-ddpm.md#eq-def-reverse) 定义逆向转移后，DDPM 的联合生成分布定义为：

$$
p_{\boldsymbol{\phi}}({\mathbf{x}}_0, {\mathbf{x}}_{1:L}) := p_{\boldsymbol{\phi}}({\mathbf{x}}_0|{\mathbf{x}}_1)  p_{\boldsymbol{\phi}}({\mathbf{x}}_1|{\mathbf{x}}_2)\cdots p_{\boldsymbol{\phi}}({\mathbf{x}}_{L-1}|{\mathbf{x}}_L)   p_{\text{prior}}({\mathbf{x}}_L),
$$

数据的边缘生成模型为：

$$
\begin{aligned}
p_{\boldsymbol{\phi}}({\mathbf{x}}_0) := \int p_{\boldsymbol{\phi}}({\mathbf{x}}_0, {\mathbf{x}}_{1:L}) \mathop{}\!\mathrm{d} {\mathbf{x}}_{1:L}.
\end{aligned}
$$

实际上，通过[式（2.2.6）](02-ddpm.md#eq-ddpm-diffusion) 进行的 DDPM 训练，可以严格建立在最大似然估计（[式（1.1.2）](../01-deep-generative-modeling/01-what-is-deep-generative-modeling.md#eq-mle)）的基础上。具体而言，其目标构成了一个 ELBO，与[第 2.1 节](01-variational-autoencoder.md#sec-vae) 中 VAE 和 HVAE 的 ELBO 类似，是对数密度的下界：

<span id="thm-ddpm-elbo"></span>

**定理 2.2.3：DDPM 的 ELBO**

<span id="eq-ddpm-elbo"></span>

$$
\begin{aligned}
        -\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0)  &\leq-\mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0; \boldsymbol{\phi})
        \\&:= \mathcal{L}_{\text{prior}}({\mathbf{x}}_0) + \mathcal{L}_{\text{recon.}}({\mathbf{x}}_0;\boldsymbol{\phi})  + \mathcal{L}_{\text{diffusion}}({\mathbf{x}}_0;\boldsymbol{\phi})
        \end{aligned}
\tag{2.2.13}
$$

这里，损失的各个组成部分定义为：

$$
\begin{aligned}
\mathcal{L}_{\text{prior}}({\mathbf{x}}_0) &:= \mathcal{D}_{\text{KL}}\Big( p({\mathbf{x}}_L|{\mathbf{x}}_0) \Vert p_{\text{prior}}({\mathbf{x}}_L)\Big)\\
   \mathcal{L}_{\text{recon.}}({\mathbf{x}}_0;\boldsymbol{\phi}) &:= \mathbb{E}_{p({\mathbf{x}}_1|{\mathbf{x}}_0)}\left[-\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0|{\mathbf{x}}_1)\right]\\
   \mathcal{L}_{\text{diffusion}}({\mathbf{x}}_0;\boldsymbol{\phi}) &= \sum_{i=2}^L  \mathbb{E}_{p_i({\mathbf{x}}_i|{\mathbf{x}}_0)}\left[\mathcal{D}_{\text{KL}}\Big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0) \Vert p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}|{\mathbf{x}}_i) \Big)\right].
\end{aligned}
$$

**定理的证明。**

推导与 HVAE/VAE 的 ELBO（[式（2.1.3）](01-variational-autoencoder.md#eq-hvae-elbo)）一样，应用 Jensen 不等式，再作进一步化简。详细证明见[第 D.1.2 节](../../appendices/d-supplementary-proofs/01-variational-perspective.md#app-vlb-proof)。

ELBO $\mathcal{L}_{\text{ELBO}}$ 的相反数由三项组成：

-   $\mathcal{L}_{\text{prior}}$ 可以小到可忽略不计，只需选择噪声调度 $\{\beta_i\}$，使得 $p(\cdot|{\mathbf{x}}_0) \approx p_{\text{prior}}(\cdot)$。

-   $\mathcal{L}_{\text{recon.}}$ 可以用 Monte Carlo 估计来近似并优化；实际实现参见 ([Ho et al., 2020](../../references.md#ref-ho2020denoising); [Kingma et al., 2021](../../references.md#ref-kingma2021variational))。

-   $\mathcal{L}_{\text{diffusion}}$（参见[式（2.2.6）](02-ddpm.md#eq-ddpm-diffusion)）将逆向条件分布 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 与 $p(\mathbf{x}_{i-1}|\mathbf{x}_i)$ 在所有步骤 $i$ 上匹配。

ELBO 目标 $\mathcal{L}_{\text{ELBO}}$ 也可以通过数据处理不等式来理解，其中潜变量为 ${\mathbf{z}} = {\mathbf{x}}_{1:L}$，如[式（2.1.2）](01-variational-autoencoder.md#eq-joint-bound) 所示：

$$
\begin{aligned}
\mathcal{D}_{\mathrm{KL}}(p_{\mathrm{data}}({\mathbf{x}}_0) \| p_{\boldsymbol{\phi}}({\mathbf{x}}_0)) \leq \mathcal{D}_{\mathrm{KL}}\left(p({\mathbf{x}}_0, {\mathbf{x}}_{1:L}) \| p_{\boldsymbol{\phi}}({\mathbf{x}}_0, {\mathbf{x}}_{1:L})\right),
\end{aligned}
$$

其中，<span class="book-inline-equation" tabindex="0">$p({\mathbf{x}}_0, {\mathbf{x}}_{1:L}) := p_{\mathrm{data}}({\mathbf{x}}_0) p({\mathbf{x}}_1|{\mathbf{x}}_0) p({\mathbf{x}}_2|{\mathbf{x}}_1) \cdots p({\mathbf{x}}_L|{\mathbf{x}}_{L-1})$</span> 表示沿前向过程的联合分布。

**注。**

扩散模型的变分视角以一种尤其便利的方式融入了层次潜变量框架：“编码器”就是固定的前向加噪链，因此无须学习推断网络。ELBO 仍然可以分解为每一步的 KL散度项，以及[式（2.2.13）](02-ddpm.md#eq-ddpm-elbo) 中的终端先验项和重建项。DDPM 训练之所以尤其容易计算，是因为在线性 Gaussian 前向过程下，这些逐步 KL散度项可以改写为简单的去噪回归问题，其条件目标具有闭式表达式。因此，关键的简化来自前向过程固定这一点：它将各层的 KL散度项转化为可高效采样的监督去噪目标，而非消除了各层的 KL散度项。

## 2.2.6 采样 {#sampling}

训练好 ${\boldsymbol{\epsilon}}$ 预测模型 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i, i)$[^3] 后，便按[图 2.5](02-ddpm.md#fig-ddpm-sampling-formal-rev) 所示依次采样，只是改为使用参数化转移 $p_{\boldsymbol{\phi}^\times}({\mathbf{x}}_{i-1} | {\mathbf{x}}_i)$。

更具体地说，从随机种子 ${\mathbf{x}}_L \sim p_{\text{prior}} = \mathcal{N}(\boldsymbol{0},{\mathbf{I}})$ 出发，根据 $p_{\boldsymbol{\phi}^\times}({\mathbf{x}}_{i-1} | {\mathbf{x}}_i)$ 递归采样，依次取 $i = L, L-1, \dots, 1$ 并采用下面的更新规则：

<span id="eq-ddpm-sampling"></span>

$$
\begin{aligned}
{\mathbf{x}}_{i-1} \leftarrow \underbrace{\frac{1}{\alpha_i}\left( {\mathbf{x}}_i  - \frac{1-\alpha_i^2}{\sqrt{1-\bar{\alpha}_i^2}} \boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i, i)\right)}_{\boldsymbol{\mu}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i, i)} +\sigma(i) \boldsymbol{\epsilon}_i,\quad \boldsymbol{\epsilon}_i\sim  \mathcal{N}(\boldsymbol{0},{\mathbf{I}}).
\end{aligned}
\tag{2.2.14}
$$

这一“去噪”过程持续进行，直到得到最终的干净生成样本 ${\mathbf{x}}_0$。

**DDPM 采样的另一种解释。**

理解 DDPM 采样的一种有用方式，是改写更新规则，使噪声水平的结构显式呈现出来。回顾[式（2.2.12）](02-ddpm.md#eq-ddpm-clean-noise)，预测噪声可以用当前样本 ${\mathbf{x}}_i$ 和预测的干净样本 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i,i)$ 表示：

$$
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i,i)
=
\frac{{\mathbf{x}}_i - \bar{\alpha}_i\,{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i,i)}
     {\sqrt{1-\bar{\alpha}_i^2}}.
$$

将其代入 DDPM 采样规则（[式（2.2.14）](02-ddpm.md#eq-ddpm-sampling)）并整理，得到

<span id="eq-ddpm-renoise"></span>

$$
{\mathbf{x}}_{i-1}
\;\leftarrow\;
\bar{\alpha}_{i-1}\,{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i,i)
\;+\;
\underbrace{
  \sqrt{1-\bar{\alpha}_{i-1}^2 - \tilde{\beta}_i^2}\;
  \boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i,i)
  \;+\;
  \tilde{\beta}_i\,\boldsymbol{\epsilon}_i
}_{\text{combined std. }=\;\sqrt{1-\bar{\alpha}_{i-1}^2}},
\tag{2.2.15}
$$

其中，$\tilde{\beta}_i^2 = \frac{(1-\bar{\alpha}_{i-1}^2)(1-\alpha_i^2)}{1-\bar{\alpha}_i^2}$ 是 DDPM 的后验方差[^4]。此外，残余项的系数满足

$$
\bigl(1-\bar{\alpha}_{i-1}^2-\tilde{\beta}_i^2\bigr)+\tilde{\beta}_i^2
=
1-\bar{\alpha}_{i-1}^2,
$$

因此，这一更新与水平 $i-1$ 下的前向边缘分布具有相同的噪声水平结构。这回到了[式（1.4.1）](../../foundations/operational-view.md#eq-ddpm-first-try) 中介绍的第一个可操作采样模板：预测干净部分，并在更低的噪声水平下重新构造样本。DDPM 以随机方式完成这一重构，将从 ${\mathbf{x}}_i$ 估计的噪声方向与一个新抽取的 Gaussian 噪声样本相结合。确定性采样器沿用相同的基本思路，但移除新引入的随机性，并相应调整剩余噪声项的系数。在[第 9.2 节](../09-numerical-solvers/02-ddim.md#sec-revisit-ddim) 更系统地研究扩散采样器时，我们将再次讨论这一构造。

将[式（2.2.15）](02-ddpm.md#eq-ddpm-renoise) 与前向边缘分布 ${\mathbf{x}}_{i-1} = \bar{\alpha}_{i-1}\,{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{i-1}^2}\,\bar{\boldsymbol{\epsilon}}$ 相比较。二者结构相同：信号系数为 $\bar{\alpha}_{i-1}$，噪声标准差为 $\sqrt{1-\bar{\alpha}_{i-1}^2}$，只是未知的 ${\mathbf{x}}_0$ 被网络预测 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i,i)$ 替代了。换言之，DDPM 每一步都*先预测干净数据，再重新加噪至恰好为 $i{-}1$ 的噪声水平*。因此，可以将 DDPM 采样理解为迭代执行两个操作：

1.  **去噪。** 估计干净数据 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_i, i)$，其依据是当前含噪输入 ${\mathbf{x}}_i$。

2.  **重新加噪。** 通过[式（2.2.15）](02-ddpm.md#eq-ddpm-renoise)，按降低后的水平 $i{-}1$ 加回噪声，得到样本 ${\mathbf{x}}_{i-1}$，其噪声水平与前向过程的第 $i{-}1$ 步一致。

然而，即使 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}$ 已训练为最优去噪器（即条件期望所给出的最小化解，见[式（2.2.11）](02-ddpm.md#eq-ddpm-clean-minimizer)），它也只能预测给定 ${\mathbf{x}}_i$ 时的*平均*干净样本。这一局限会导致预测模糊，尤其在高噪声水平下，从受到严重扰动的输入中恢复细节结构会变得困难。从这一视角看，扩散采样由高噪声向低噪声推进，逐渐细化对干净信号的估计。早期步骤确定整体结构，后期步骤补充精细细节，随着噪声被移除，样本也变得更加逼真。

<span id="fig-vdm-clean"></span>

![DDPM 采样：预测干净数据，再在更低噪声水平下重新加噪](../../../en/assets/chapter-02/ddpm-denoise-renoise.png)

**图 2.7：** **以先去噪、再重新加噪的视角理解 DDPM 采样。** 根据当前含噪样本 $\mathbf{x}_i$，模型首先估计其潜在的干净信号 $\mathbf{x}_{\boldsymbol{\phi}^\times}(\mathbf{x}_i,i)$，然后采样得到一个噪声更少的点 $\mathbf{x}_{i-1}$，方法是将这一估计重新加噪到第 $(i-1)$ 个噪声水平。

来源：由本书作者绘制。

**DDPM 的采样速度较慢。** DDPM（即扩散模型）的采样本身较慢。在原始表述与实现中，生成一个样本通常需要约 $1{,}000$ 个去噪步骤。这是因为采样要经过一长串小幅细化：在每一步，模型都将当前含噪样本向更干净的状态略微更新，而下一次更新必须以上一步的结果为起点。因此，标准 DDPM 采样按时间步推进，具有很强的顺序依赖性；通常需要很多这样的步骤，才能紧密跟踪逆向扩散路径，获得较好的样本质量。

[定理 2.2.1](02-ddpm.md#thm-equiv-marginal-kl) 表明，表达能力足够强的 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1} | \mathbf{x}_i)$ 在理论上可以匹配真实逆向分布 $p(\mathbf{x}_{i-1} | \mathbf{x}_i)$。然而，实践中通常将 $p_{\boldsymbol{\phi}}(\mathbf{x}_{i-1} | \mathbf{x}_i)$ 建模为 Gaussian 分布，以近似 $p(\mathbf{x}_{i-1} | \mathbf{x}_i)$，这限制了其表达能力。

当前向噪声尺度 $\beta_i$ 较小时，真实逆向分布近似为 Gaussian 分布，因此可以准确近似。相反，较大的 $\beta_i$ 会产生多模态或较强的非 Gaussian 性，单个 Gaussian 分布无法刻画。为保持准确性，DDPM 采用许多 $\beta_i$ 较小的步骤，形成一条依次推进的链；每一步都依赖于前一步，并需要计算一次神经网络 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\mathbf{x}_i, i)$。这需要 $\mathcal{O}(L)$ 次顺序执行的网络计算，无法并行，因此生成较慢。

后文[第 4 章](../04-score-sde-framework/index.md#ch-score-sde) 将从微分方程问题的角度，对这一固有的采样瓶颈给出更有理论依据的解释，由此引出加速生成的连续时间数值策略。

[^1]: 这一表述虽然看起来可能不同，但在数学上与原始 DDPM 的转移核等价。

[^2]: 从 $\mathcal{L}_{\text{DDPM}}$ 中移除由 ELBO 推导出的权重，会改变训练目标，但不会改变其总体最优预测：${\boldsymbol{\epsilon}}^*({\mathbf{x}}_i,i) = {\mathbb{E}}[{\boldsymbol{\epsilon}}| {\mathbf{x}}_i]$。因此，加权的 DDPM 损失与简化后的损失具有相同的理想预测器，但在训练时侧重于不同的噪声水平。实践中，这种加权仍然会影响优化和性能。

[^3]: 我们用符号“$\times$”表示模型已经训练完成，当前参数已冻结。

[^4]: 系数 $\sqrt{1-\bar{\alpha}_{i-1}^2 - \tilde{\beta}_i^2}$ 可化简为 $\frac{\alpha_i(1-\bar{\alpha}_{i-1}^2)}{\sqrt{1-\bar{\alpha}_i^2}}$，它始终非负，因此该平方根有明确定义。

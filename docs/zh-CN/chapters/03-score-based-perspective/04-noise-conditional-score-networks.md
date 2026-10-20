# 3.4 多噪声水平的去噪 score 匹配（NCSN） {#sec-smld}

## 3.4.1 动机 {#motivation}

向数据分布加入单一固定方差的 Gaussian 噪声，可以在一定程度上对其进行平滑，但只在一个噪声水平上训练基于 score 的模型会带来若干关键限制。当注入的噪声较小时，由于低密度区域的梯度消失，Langevin 动力学难以在多模态分布的各模式之间移动。噪声较大时，采样则更容易，但模型只能捕捉粗粒度结构，产生缺少精细细节的模糊样本。此外，在高维空间中，Langevin 动力学可能收敛缓慢，甚至失效。由于它依赖对数密度梯度来引导，较差的初始化，特别是在平台区域或鞍点附近，可能阻碍探索，或使采样器困在单一模式中。

<span id="fig-sm-acc-illustration"></span>

![低密度区域中的 score 估计可能不准确](../../../en/assets/chapter-03/score-matching-inaccuracy.png)

**图 3.6：** **SM 不准确性的示意图（回顾[图 3.4](02-score-based-models.md#fig-sm-illustration)）。** 红色区域表示低密度区域，由于样本覆盖有限，其中的 score 估计可能不准确；高密度区域通常能得到更准确的估计。

来源：作者绘制。

为应对这些挑战，[Song and Ermon (2019)](../../references.md#ref-song2019generative) 提出向数据分布注入多个水平的 Gaussian 噪声，并联合训练 Noise Conditional Score Network（NCSN），估计一系列噪声尺度上的 score 函数。生成时，以噪声退火的方式应用 Langevin 动力学：从高噪声水平开始进行粗粒度探索，再逐渐向低噪声水平细化，恢复精细细节。

<span id="fig-smld-ddpm-graphical-model"></span>

![Noise Conditional Score Networks 在多个噪声水平上训练和采样](../../../en/assets/chapter-03/ncsn.png)

**图 3.7：** **NCSN 示意图。** 前向过程用多个水平的加性 Gaussian 噪声 $p_{\sigma}({\mathbf{x}}_{\sigma} | {\mathbf{x}})$ 扰动数据。生成过程（虚线）在每个噪声水平上进行 Langevin 采样，并用当前水平的结果初始化下一个更小方差下的采样。

来源：作者绘制。

## 3.4.2 训练 {#training}

为克服仅在单一噪声水平上训练的 score 模型的限制，[Song and Ermon (2019)](../../references.md#ref-song2019generative) 提出向数据分布添加多个水平的 Gaussian 噪声。具体而言，选择由 $L$ 个噪声水平组成的序列 $\{\sigma_i\}_{i=1}^L$，满足

$$
0 < \sigma_1 < \sigma_2 < \cdots < \sigma_L,
$$

其中 $\sigma_1$ 足够小，能够保留数据的大部分精细细节，而 $\sigma_L$ 足够大，能够充分平滑分布，使训练更容易。

每个含噪样本都通过扰动干净数据点 ${\mathbf{x}} \sim p_{\text{data}}$ 构造为 ${\mathbf{x}}_{\sigma} = {\mathbf{x}} + \sigma \boldsymbol{\epsilon}$，其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$。这定义了

**扰动核：**

$$
p_{\sigma}({\mathbf{x}}_{\sigma} | {\mathbf{x}}) := \mathcal{N}({\mathbf{x}}_{\sigma}; {\mathbf{x}}, \sigma^2 {\mathbf{I}}),
$$

它在每个噪声水平下诱导出

**边缘分布：**

$$
p_\sigma({\mathbf{x}}_{\sigma}) = \int p_{\sigma} ({\mathbf{x}}_{\sigma} | {\mathbf{x}})    p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}},
$$

其中噪声水平为 $\sigma$。它表示经过 Gaussian 平滑的数据分布。

**NCSN 的训练目标。**

目标是训练 Noise Conditional Score Network ${\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}, \sigma)$，估计 score 函数 $\nabla_{{\mathbf{x}}} \log p_\sigma({\mathbf{x}})$，覆盖所有 $\sigma \in \{\sigma_i\}_{i=1}^L$。这通过最小化所有噪声水平上的 DSM 目标实现：

<span id="eq-nscn"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{NCSN}}(\boldsymbol{\phi}) := \sum_{i=1}^{L} \lambda(\sigma_i)   \mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma_i),
\end{aligned}
\tag{3.4.1}
$$

其中

$$
\begin{aligned}
\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma)
= \frac{1}{2} \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}}),   \tilde{{\mathbf{x}}} \sim p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})}
\left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}, \sigma) - \left( \frac{{\mathbf{x}} - \tilde{{\mathbf{x}}}}{\sigma^2} \right)\right\rVert_2^2 \right],
\end{aligned}
$$

而 $\lambda(\sigma_i)>0$ 是各尺度的权重函数。

最小化该目标得到 score 模型 ${\mathbf{s}}^*({\mathbf{x}}, \sigma)$，它恢复每个噪声水平的真实 score：

$$
{\mathbf{s}}^*(\cdot, \sigma) = \nabla_{{\mathbf{x}}} \log p_\sigma(\cdot),  \quad \text{for all } \sigma \in \{\sigma_i\}_{i=1}^L,
$$

因为这本质上就是 DSM 最小化（见[定理 3.3.1](03-denoising-score-matching.md#thm-sm-dsm)）。

**与 DDPM 损失的关系。** 令 ${\mathbf{x}}_\sigma={\mathbf{x}}+\sigma\boldsymbol{\epsilon}$，其中 $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$，并令 $p_\sigma$ 表示边缘分布。由 Tweedie 公式，

$$
\nabla_{{\mathbf{x}}_\sigma}\log p_\sigma({\mathbf{x}}_\sigma)
= -\frac{1}{\sigma} {\mathbb{E}} \left[\boldsymbol{\epsilon} \middle|  {\mathbf{x}}_\sigma\right].
$$

因此，NCSN 的最优解是真实 score ${\mathbf{s}}^*({\mathbf{x}}_\sigma,\sigma)=\nabla_{{\mathbf{x}}_\sigma}\log p_\sigma({\mathbf{x}}_\sigma)$，而 DDPM 损失[式 (2.2.10)](../02-variational-perspective/02-ddpm.md#eq-ddpm-simple-loss)下的 Bayes 最优噪声预测器为 $\boldsymbol{\epsilon}^*({\mathbf{x}}_\sigma,\sigma)={\mathbb{E}}[\boldsymbol{\epsilon}|{\mathbf{x}}_\sigma]$。二者通过下式严格等价：

$$
{\mathbf{s}}^*({\mathbf{x}}_\sigma,\sigma) = - \frac{1}{\sigma} \boldsymbol{\epsilon}^*({\mathbf{x}}_\sigma,\sigma),
\qquad
\boldsymbol{\epsilon}^*({\mathbf{x}}_\sigma,\sigma) = - \sigma {\mathbf{s}}^*({\mathbf{x}}_\sigma,\sigma).
$$

在 DDPM 设定下，第 $i$ 步的含噪样本可写为

$$
\sigma_i:=\sqrt{1-\bar{\alpha}_i^2},
\qquad
{\mathbf{x}}_i
=
\bar{\alpha}_i{\mathbf{x}}_0+\sigma_i{\boldsymbol{\epsilon}},
\qquad
{\boldsymbol{\epsilon}}\sim\mathcal{N}(\boldsymbol{0},{\mathbf{I}}).
$$

因此，score 与最优噪声预测包含相同的信息：

$$
{\mathbf{s}}^*({\mathbf{x}}_i,i)
=
-\frac{1}{\sigma_i}
{\mathbb{E}}\left[{\boldsymbol{\epsilon}}|{\mathbf{x}}_i\right].
$$

换言之，DDPM 中预测噪声，在相差一个已知尺度因子 $1/\sigma_i$ 的意义下，等价于预测噪声水平 $i$ 下的 score。

我们将在[第 6 章](../06-unified-perspective/index.md#ch-unified-perspective)系统比较并总结这些参数化之间的等价性。

## 3.4.3 采样 {#sampling}

<section class="book-algorithm" id="alg-smld" aria-labelledby="alg-smld-title" markdown="1">

<div class="book-algorithm__title" id="alg-smld-title" markdown="span">
算法 1：退火 Langevin 动力学
</div>

<div class="book-algorithm__meta" markdown="1">

**输入：** 训练好的 score ${\mathbf{s}}_{\boldsymbol{\phi}^\times}(\cdot,\sigma_\ell)$、步长 $\eta_\ell$ 和 Langevin 迭代次数 $N_\ell$，分别对应各噪声水平 $\ell=L,\ldots,1$；初始化分布 $p_{\mathrm{init}}$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="算法 1 步骤" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{x}} \sim p_{\mathrm{init}}$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">*在最大噪声水平下从无结构噪声初始化*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

**for** $\ell=L,L-1,\ldots,1$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{{\mathbf{x}}}_0 \gets {\mathbf{x}}$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">*使用上一个水平的输出初始化当前噪声水平*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

**for** $n=0$ **to** $N_\ell-1$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

$\boldsymbol{\epsilon}_n \sim \mathcal{N}(\boldsymbol{0},{\mathbf{I}})$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{{\mathbf{x}}}_{n+1} \gets \tilde{{\mathbf{x}}}_n +\eta_\ell {\mathbf{s}}_{\boldsymbol{\phi}^\times}(\tilde{{\mathbf{x}}}_n,\sigma_\ell) +\sqrt{2\eta_\ell}\,\boldsymbol{\epsilon}_n$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">7</div>
<div class="book-algorithm__statement" markdown="1">

**end for**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">8</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{x}} \gets \tilde{{\mathbf{x}}}_{N_\ell}$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">*若还有更小的噪声水平，将最终状态传给下一个水平*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">9</div>
<div class="book-algorithm__statement" markdown="1">

**end for**

</div>
</div>

</div>
</div>

<div class="book-algorithm__meta" markdown="1">

**输出：** ${\mathbf{x}}$

</div>

</section>

有了在多个噪声水平下训练好的 score 网络

$$
{\mathbf{s}}_{\boldsymbol{\phi}^\times}(\cdot, \sigma_1), \quad{\mathbf{s}}_{\boldsymbol{\phi}^\times}(\cdot, \sigma_2), \quad \cdots, \quad {\mathbf{s}}_{\boldsymbol{\phi}^\times}(\cdot, \sigma_{L-1}), \quad    {\mathbf{s}}_{\boldsymbol{\phi}^\times}(\cdot, \sigma_L),
$$

称为*退火 Langevin 动力学*（[Song and Ermon, 2019](../../references.md#ref-song2019generative)）的采样过程，通过从高噪声水平 $\sigma_L$ 逐步去噪到低噪声水平 $\sigma_1 \approx 0$ 来生成数据。

有了噪声水平 $0<\sigma_1<\cdots<\sigma_L$ 上训练好的 score，退火 Langevin 动力学就可以通过从高噪声逐渐移向低噪声来生成样本。从简单的初始化 ${\mathbf{x}}\sim p_{\mathrm{init}}$ 开始，它首先在最大噪声水平 $\sigma_L$ 上应用 Langevin 动力学，其 score ${\mathbf{s}}_{\boldsymbol{\phi}^\times}(\cdot,\sigma_L)$ 引导样本移向对应的平滑分布 $p_{\sigma_L}$。所得样本随后用于初始化下一个更低噪声水平上的 Langevin 动力学。反复执行这一过程会逐步细化样本，直到达到最小噪声水平 $\sigma_1$。

在每个噪声水平 $\sigma_\ell$ 上，Langevin 动力学执行更新

$$
\tilde{{\mathbf{x}}}_{n+1}
=
\tilde{{\mathbf{x}}}_n
+
\eta_\ell
{\mathbf{s}}_{\boldsymbol{\phi}^\times}(\tilde{{\mathbf{x}}}_n,\sigma_\ell)
+
\sqrt{2\eta_\ell}\,\boldsymbol{\epsilon}_n,
\qquad
\boldsymbol{\epsilon}_n\sim\mathcal{N}(\boldsymbol{0},{\mathbf{I}}),
$$

其中，上一个噪声水平的输出提供初值 $\tilde{{\mathbf{x}}}_0$。常见做法是让步长按噪声水平缩放为

$$
\eta_\ell
=
\delta\frac{\sigma_\ell^2}{\sigma_1^2},
\qquad
\delta>0.
$$

该算法在每个噪声水平上都执行 Langevin 更新，包括 $\sigma_1$，因此最终样本是在最小噪声水平上细化后得到的。[算法 1](04-noise-conditional-score-networks.md#alg-smld)总结了这一过程。

初始分布 $p_{\mathrm{init}}$ 仅为 Markov 链提供起点。在最大噪声水平上，Langevin 动力学使这一初始分布趋近 $p_{\sigma_L}$；随后，连续的噪声水平逐步引导样本接近数据分布的平滑程度越来越低的版本。

**NCSN 的采样速度缓慢。** NCSN 通过在各噪声尺度 $\{\sigma_i\}_{i=1}^L$ 上进行退火 MCMC（通常是 Langevin 动力学）生成样本。对于每个尺度 $\sigma_i$，它执行 $K$ 次迭代更新，形式为“更新 $\tilde{\mathbf{x}}_n$ 时，使用 score ${\mathbf{s}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_n,\sigma_i)$ 并加上小幅随机扰动”，每次都需要对 score 网络进行一次前向计算。两个因素使 $L \times K$ 必须较大：

-   **(i)** **局部准确性与稳定性：** 学得的 score 只对小扰动可靠，因此需要使用小步长，并在每个噪声水平上进行多次迭代，以避免偏差或不稳定；

-   **(ii)** **高维空间中混合缓慢：** 局部 MCMC 移动对多模态高维目标的探索效率较低，需要多次迭代才能到达典型数据区域。

由于更新严格串行（每次迭代依赖上一次），且每次都需要昂贵的网络求值，总成本为 $\mathcal{O}(L K)$ 次串行网络计算，使采样在计算上较慢。
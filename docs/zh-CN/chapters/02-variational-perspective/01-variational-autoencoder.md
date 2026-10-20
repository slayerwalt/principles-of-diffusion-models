# 2.1 变分自编码器 {#sec-vae}

神经网络如何学会生成逼真的数据？一个自然的起点是*自编码器*，它由两个网络组成：确定性的*编码器*将输入压缩成低维潜编码，确定性的*解码器*再从这一编码重建输入。训练过程最小化原始输入与重建结果之间的重建误差。这种设置虽然能够实现准确重建，但潜空间缺乏结构：随机采样潜编码通常会产生无意义的输出，从而限制了模型在生成任务中的应用。

*变分自编码器（Variational Autoencoder，VAE）*（[Kingma and Welling, 2013](../../references.md#ref-kingma2013auto)）通过在潜空间中引入概率结构来解决这一问题。这使模型从简单的重建工具转变为能够产生新颖、逼真数据的生成模型。

## 2.1.1 概率编码器与解码器 {#probabilistic-encoder-and-decoder}

<span id="fig-vae-graph"></span>

![包含随机编码器和解码器的变分自编码器](../../../en/assets/chapter-02/vae-graph.png)

**图 2.1：** **VAE 示意图。** 它包含一个随机编码器 $q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}})$，将数据 ${\mathbf{x}}$ 映射到潜变量 ${\mathbf{z}}$，以及一个解码器 $p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}})$，从潜变量重建数据。

来源：由原作者绘制。

**解码器（生成器）的构造。**

在 VAE 中，我们区分两类变量：*观测变量* ${\mathbf{x}}$ 对应于我们看到的数据（例如一幅图像），*潜变量* ${\mathbf{z}}$ 则刻画隐藏的变化因素（例如物体的形状、颜色或风格）。模型假设，每个观测 ${\mathbf{x}}$ 都由一个潜变量生成，而该潜变量采样自简单的*先验分布*，通常为标准 Gaussian 分布，即 ${\mathbf{z}} \sim p_{\text{prior}} := \mathcal{N}(\mathbf{0}, \mathbf{I})$。

为了将 ${\mathbf{z}}$ 映射回数据空间，我们定义*解码器（生成器）*分布 $p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}})$。在实践中，解码器通常保持简单，常采用因子化 Gaussian 分布（见[第 2.1.3 节](01-variational-autoencoder.md#subsec-optimization-vae)）或类似的分布，使学习过程着重提取有用的潜在特征，而非记忆数据。直观上，直接逐个生成像素极为困难；潜变量提供了紧凑的表示，从中解码出准确的像素排列就容易得多。生成新样本时，先采样 ${\mathbf{z}} \sim p_{\text{prior}}$，再通过 ${\mathbf{x}} \sim p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}})$ 进行解码。

由此，VAE 通过以下边缘似然定义了一个潜变量生成模型：

$$
p_{\boldsymbol{\phi}}({\mathbf{x}}) = \int p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}}) p({\mathbf{z}}) \mathop{}\!\mathrm{d} {\mathbf{z}}.
$$

理想情况下，通过最大化这一边缘似然来学习解码器参数 $\boldsymbol{\phi}$，即采用最大似然估计（见[公式 (1.1.2)](../01-deep-generative-modeling/01-what-is-deep-generative-modeling.md#eq-mle)）。然而，对于表达能力强的非线性解码器，关于 ${\mathbf{z}}$ 的积分难以计算，因此直接进行 MLE 在计算上不可行。这促使我们采用 VAE 中的变分方法。

**编码器（推断网络）的构造。**

为了将这个难以计算的生成器与真实数据联系起来，考虑反向的问题：给定一个观测 ${\mathbf{x}}$，哪些潜编码 ${\mathbf{z}}$ 可能生成它？根据 Bayes 公式，后验分布为

$$
p_{\boldsymbol{\phi}}({\mathbf{z}}|{\mathbf{x}}) \;=\; \frac{p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}}) p({\mathbf{z}})}{p_{\boldsymbol{\phi}}({\mathbf{x}})}.
$$

困难在于，分母涉及边缘似然 $p_{\boldsymbol{\phi}}({\mathbf{x}})$，它需要对所有潜变量积分，而对于非线性解码器，这一积分难以计算。因此，要精确推断 ${\mathbf{z}}$，就必须基于 ${\mathbf{x}}$ 进行计算，而其代价高得难以承受。

VAE 中的“变分”步骤用一个可计算的近似分布替代难以计算的后验分布，以解决这一问题。我们引入一个由神经网络参数化的*编码器*（或推断网络）$q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}})$，作为可学习的替代分布：

$$
q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}}) \;\approx\; p_{\boldsymbol{\phi}}({\mathbf{z}}|{\mathbf{x}}).
$$

在实践中，编码器将每个观测数据点 ${\mathbf{x}}$ 映射为潜编码上的一个分布，提供了一条从 ${\mathbf{x}}$ 返回 ${\mathbf{z}}$ 的可行且可训练的路径，从而使学习成为可能。

## 2.1.2 通过证据下界（ELBO）训练 {#training-via-elbo}

现在，我们定义一个可计算的训练目标。虽然无法直接优化 $\log p_{\boldsymbol{\phi}}(\mathbf{x})$，但可以最大化它的一个下界，即*证据下界（Evidence Lower Bound，ELBO）*：

<span id="thm-elbo"></span>

**定理 2.1.1：证据下界（ELBO）**

对于任意数据点 $\mathbf{x}$，对数似然满足：

$$
\log p_{\boldsymbol{\phi}}(\mathbf{x}) \geq \mathcal{L}_{\text{ELBO}}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}),
$$

其中，ELBO 为：

<span id="eq-vae-elbo"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})} \left[ \log p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z}) \right]}_{\text{Reconstruction Term}} - \underbrace{\mathcal{D}_{\mathrm{KL}}\left( q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}) \right)}_{\text{Latent Regularization}}.
\end{aligned}
\tag{2.1.1}
$$

**定理证明。**

ELBO 可由 Jensen 不等式得到：

$$
\begin{aligned}
\log p_{\boldsymbol{\phi}}(\mathbf{x}) &= \log \int p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z})   \mathrm{d}\mathbf{z} = \log \int q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}) \frac{p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})}   \mathrm{d}\mathbf{z} \\
&= \log \mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})} \left[ \frac{p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})} \right] \geq \mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})} \right].
\end{aligned}
$$

ELBO 目标自然地分解为两部分：

-   **重建项：** 使 $\mathbf{x}$ 能够从其潜编码 $\mathbf{z}$ 中被准确恢复。在编码器和解码器均为 Gaussian 分布的假设下，这一项恰好化为熟悉的自编码器重建损失（参见[第 2.1.3 节](01-variational-autoencoder.md#subsec-optimization-vae)）。不过，与自编码器一样，仅优化这一项有记忆训练数据的风险，因此需要额外的正则化。

-   **潜变量 KL 项：** 使编码器分布 $q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$ 接近简单的 Gaussian 先验 $p_{\mathrm{prior}}(\mathbf{z})$。这一正则化将潜空间塑造成平滑、连续的结构，确保从先验采样得到的样本能够被可靠地解码，从而实现有意义的生成。

这种权衡既保证忠实的重建，也保证采样结果的协调性。

**信息论视角：作为散度界的 ELBO。**

ELBO 目标具有自然的信息论解释。回顾最大似然训练，它相当于最小化以下 KL散度：

$$
\mathcal{D}_{\mathrm{KL}}(p_{\text{data}}({\mathbf{x}}) \| p_{\boldsymbol{\phi}}({\mathbf{x}})),
$$

该散度衡量模型分布对数据分布的逼近程度。由于这一项通常难以计算，变分框架转而引入联合分布之间的比较。

具体而言，考虑以下两个联合分布：

-   **生成联合分布** $p_{\boldsymbol{\phi}}({\mathbf{x}}, {\mathbf{z}}) = p({\mathbf{z}}) p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}})$，描述模型如何生成数据；

-   **推断联合分布** $q_{\boldsymbol{\theta}}({\mathbf{x}}, {\mathbf{z}}) = p_{\text{data}}({\mathbf{x}}) q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}})$，将真实数据与推断得到的潜变量结合起来。

比较这两个分布可得如下不等式：

<span id="eq-joint-bound"></span>

$$
\begin{aligned}
\mathcal{D}_{\mathrm{KL}}(p_{\text{data}}({\mathbf{x}}) \| p_{\boldsymbol{\phi}}({\mathbf{x}}))
    \leq \mathcal{D}_{\mathrm{KL}}(q_{\boldsymbol{\theta}}({\mathbf{x}},{\mathbf{z}}) \| p_{\boldsymbol{\phi}}({\mathbf{x}},{\mathbf{z}})),
\end{aligned}
\tag{2.1.2}
$$

它有时被称为 KL散度的链式法则。直观上，仅比较边缘分布（${\mathbf{x}}$）可能掩盖某些不匹配，而考虑完整的潜变量与数据的联合分布时，这些不匹配便会显现出来。

形式上，联合 KL散度可以展开为

$$
\begin{aligned}
&\underbrace{\mathcal{D}_{\mathrm{KL}}(q_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) \| p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}))}_{\text{Total Error Bound}}
    \\=& \mathbb{E}_{q_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})} \left[ \log \frac{p_{\text{data}}(\mathbf{x}) q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})}{p_{\boldsymbol{\phi}}(\mathbf{x}) p_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})} \right] \\=&\mathbb{E}_{p_{\text{data}}(\mathbf{x})} \left[ \log \frac{p_{\text{data}}(\mathbf{x})}{p_{\boldsymbol{\phi}}(\mathbf{x})} + \mathcal{D}_{\mathrm{KL}}\left( q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}) \| p_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \right) \right] \\
    \\=&  \underbrace{\mathcal{D}_{\mathrm{KL}}(p_{\text{data}} \| p_{\boldsymbol{\phi}})}_{\text{True Modeling Error}} + \underbrace{\mathbb{E}_{p_{\text{data}}(\mathbf{x})} \big[ \mathcal{D}_{\mathrm{KL}}( q_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}) \| p_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) ) \big]}_{\text{Inference Error}},
\end{aligned}
$$

其中，第一项是真实的建模误差，第二项是推断误差，即近似后验与真实后验之间的差距。后者始终非负，这就解释了[公式 (2.1.2)](01-variational-autoencoder.md#eq-joint-bound)。

最后，注意到

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}) - \mathcal{L}_{\text{ELBO}}(\boldsymbol{\theta},\boldsymbol{\phi};{\mathbf{x}})
= \mathcal{D}_{\mathrm{KL}} \left(q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}})\|p_{\boldsymbol{\phi}}({\mathbf{z}}|{\mathbf{x}})\right).
$$

因此，推断误差恰好是对数似然与 ELBO 之间的差距。最大化 ELBO 因而对应于直接减小推断误差，确保训练最小化整体界中一个有意义的部分。

## 2.1.3 Gaussian VAE {#subsec-optimization-vae}

VAE 的一种标准形式是在编码器和解码器中都采用 Gaussian 分布。

**编码器部分。** 编码器 $q_{\boldsymbol{\theta}}(\mathbf{z} | \mathbf{x})$ 通常建模为如下 Gaussian 分布：

$$
q_{\boldsymbol{\theta}}(\mathbf{z} | \mathbf{x}) := \mathcal{N}\left(\mathbf{z}; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}), \operatorname{diag}(\boldsymbol{\sigma}_{\boldsymbol{\theta}}^2(\mathbf{x}))\right),
$$

其中，$\boldsymbol{\mu}_{\boldsymbol{\theta}} : \mathbb{R}^D \to \mathbb{R}^d$ 和 $\boldsymbol{\sigma}_{\boldsymbol{\theta}} : \mathbb{R}^D \to \mathbb{R}_+^d$ 是编码器网络的确定性输出。

**解码器部分。** 解码器通常建模为具有固定方差的 Gaussian 分布：

$$
p_{\boldsymbol{\phi}}(\mathbf{x}|\mathbf{z}) := \mathcal{N}\bigl(\mathbf{x}; \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{z}), \sigma^2 {\mathbf{I}}\bigr),
$$

其中，$\boldsymbol{\mu}_{\boldsymbol{\phi}} : \mathbb{R}^d \to \mathbb{R}^D$ 是神经网络，$\sigma > 0$ 是控制方差的一个较小常数。

在这一假设下，ELBO 中的重建项简化为

$$
\mathbb{E}_{q_{\boldsymbol{\theta}}(\mathbf{z} | \mathbf{x})} \left[\log p_{\boldsymbol{\phi}}(\mathbf{x} | \mathbf{z})\right]
= - \frac{1}{2\sigma^2} \mathbb{E}_{q_{\boldsymbol{\theta}}(\mathbf{z} | \mathbf{x})} \left[\|\mathbf{x} - \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{z})\|^2\right] + C,
$$

其中，$C$ 是与 $\boldsymbol{\theta}$ 和 $\boldsymbol{\phi}$ 无关的常数。因此，ELBO 目标化为：

$$
\min_{\boldsymbol{\theta}, \boldsymbol{\phi}}  \mathbb{E}_{q_{\boldsymbol{\theta}}(\mathbf{z} | \mathbf{x})} \left[ \frac{1}{2\sigma^2} \|\mathbf{x} - \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{z})\|^2 \right]
+ \mathcal{D}_{\mathrm{KL}}\big(q_{\boldsymbol{\theta}}(\mathbf{z} | \mathbf{x}) \| p_{\mathrm{prior}}(\mathbf{z})\big),
$$

由于 Gaussian 分布的假设，其中的 KL 项具有闭式解。因此，训练 VAE 就化为最小化带有正则化的重建损失。

## 2.1.4 标准 VAE 的缺点 {#sec-drawback-standard-vae}

尽管 VAE 框架在理论上具有吸引力，但它存在一个关键缺点：生成的输出往往模糊不清[^1]。

**VAE 中模糊的生成结果。** 为了理解这一现象，考虑固定的 Gaussian 编码器 $q_{\text{enc}}(\mathbf{z}|\mathbf{x})$，以及具有以下形式的解码器：

$$
p_{\text{dec}}(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}({\mathbf{z}}), \sigma^2 {\mathbf{I}}),
$$

其中，$\boldsymbol{\mu}({\mathbf{z}})$ 表示解码器网络。对于任意给定的编码器，优化 ELBO 可化为最小化期望重建误差（忽略一个加性常数）：

$$
\operatorname*{arg\,min}_{\boldsymbol{\mu}} \mathbb{E}_{p_{\text{data}}(\mathbf{x})q_{\text{enc}}(\mathbf{z}|\mathbf{x})} \left[ \| \mathbf{x} - \boldsymbol{\mu}({\mathbf{z}}) \|^2 \right].
$$

这是关于 $\boldsymbol{\mu}({\mathbf{z}})$ 的标准最小二乘问题，其解由条件均值以闭式形式给出：

$$
\boldsymbol{\mu}^*(\mathbf{z}) = \mathbb{E}_{q_{\text{enc}}(\mathbf{x}|\mathbf{z})}[\mathbf{x}],
$$

其中，$q_{\text{enc}}(\mathbf{x}|\mathbf{z})$ 是由编码器诱导的、给定潜变量时输入的后验分布，由 Bayes 公式定义为：

$$
q_{\text{enc}}(\mathbf{x}|\mathbf{z})
=
\frac{q_{\text{enc}}(\mathbf{z}|\mathbf{x})\, p_{\text{data}}(\mathbf{x})}{q_{\text{enc}}(\mathbf{z})},
\qquad
q_{\text{enc}}(\mathbf{z})
:=
\int q_{\text{enc}}(\mathbf{z}|\mathbf{x})\, p_{\text{data}}(\mathbf{x})\, \mathop{}\!\mathrm{d} \mathbf{x}.
$$

利用 Bayes 公式，最优生成器也可以等价地写为：

$$
\boldsymbol{\mu}^*(\mathbf{z})
=
\frac{\mathbb{E}_{p_{\text{data}}(\mathbf{x})}\!\left[q_{\text{enc}}(\mathbf{z}|\mathbf{x}) \cdot \mathbf{x}\right]}
{q_{\text{enc}}(\mathbf{z})}
=
\frac{\mathbb{E}_{p_{\text{data}}(\mathbf{x})}\!\left[q_{\text{enc}}(\mathbf{z}|\mathbf{x}) \cdot \mathbf{x}\right]}
{\mathbb{E}_{p_{\text{data}}(\mathbf{x})}\!\left[q_{\text{enc}}(\mathbf{z}|\mathbf{x})\right]}.
$$

现在，假设两个不同的输入 $\mathbf{x} \neq \mathbf{x}'$ 被映射到潜空间中重叠的区域，即 $q_{\text{enc}}(\cdot|\mathbf{x})$ 与 $q_{\text{enc}}(\cdot|\mathbf{x}')$ 的支撑集相交。那么，$\boldsymbol{\mu}^*(\mathbf{z})$ 会对多个可能互不相关的输入取平均，从而产生模糊、缺乏鲜明特征的输出。对相互冲突的模式取平均，是 VAE 生成样本呈现典型模糊现象的一个根本原因。

## 2.1.5 （选读）从标准 VAE 到层次 VAE {#hierarchical-vaes}

为了建模复杂数据，层次变分自编码器（Hierarchical Variational Autoencoder，HVAE）（[Vahdat and Kautz, 2020](../../references.md#ref-vahdat2020nvae)）通过引入潜变量的层次结构来增强 VAE。这种深层的分层结构使模型能够在多个抽象层次上捕捉数据特征，显著增强表达能力，也体现了现实世界数据的组合性质。

<span id="fig-hvae-graph"></span>

![由多层可训练编码器和解码器堆叠而成的层次 VAE](../../../en/assets/chapter-02/hvae-graph.png)

**图 2.2：** **HVAE 的计算图。** 它具有层次结构，在多个潜变量层中堆叠了可训练的编码器和解码器。

来源：由原作者绘制。

**HVAE 的建模。** 标准 VAE 只使用单个潜编码 $\mathbf{z}$，层次 VAE（HVAE）则引入多层潜变量，形成自顶向下的层次结构。每一潜变量层都作为下一层的条件，由此构成一条条件先验链，捕捉逐渐精细的抽象层次上的结构。这就得到联合分布的如下自顶向下分解：

$$
p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}_{1:L}) = p_{\boldsymbol{\phi}}(\mathbf{x} | \mathbf{z}_1) \prod_{i=2}^{L} p_{\boldsymbol{\phi}}(\mathbf{z}_{i-1}|\mathbf{z}_i)   p(\mathbf{z}_L).
$$

这一结构定义了数据的边缘分布：

$$
p_{\text{HVAE}}(\mathbf{x}) := \int p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}_{1:L}) \mathop{}\!\mathrm{d} \mathbf{z}_{1:L}.
$$

生成过程逐步进行：从顶层潜变量 $\mathbf{z}_L$ 开始，依次向下解码各层潜变量，直至 $\mathbf{z}_1$，再生成最终观测 $\mathbf{x}$。

在编码部分，HVAE 使用具有结构且可学习的变分编码器 $q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L}|\mathbf{x})$，其结构与生成过程的层次结构相对应。一种常见选择是自底向上的 Markov 分解：

$$
q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L} | \mathbf{x}) = q_{\boldsymbol{\theta}}(\mathbf{z}_1 | \mathbf{x}) \prod_{i=2}^{L} q_{\boldsymbol{\theta}}(\mathbf{z}_i | \mathbf{z}_{i-1}).
$$

**HVAE 的 ELBO。** 与[公式 (2.1.1)](01-variational-autoencoder.md#eq-vae-elbo) 类似，可以利用 Jensen 不等式推导 ELBO：

<span id="eq-hvae-elbo"></span>

$$
\begin{aligned}
    \log p_{\text{HVAE}}(\mathbf{x})
&= \log \int p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}_{1:L}) \mathop{}\!\mathrm{d} \mathbf{z}_{1:L} \\
&= \log \int \frac{p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}_{1:L})}{q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L}|\mathbf{x})}   q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L}|\mathbf{x}) \mathop{}\!\mathrm{d} \mathbf{z}_{1:L} \\
&= \log \mathbb{E}_{q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L}|\mathbf{x})} \left[ \frac{p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}_{1:L})}{q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L}|\mathbf{x})} \right] \\
&\geq \mathbb{E}_{q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L}|\mathbf{x})} \left[ \log \frac{p_{\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}_{1:L})}{q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L}|\mathbf{x})} \right]\\
&=:\mathcal{L}_{\text{ELBO}}(\boldsymbol{\phi}).
\end{aligned}
\tag{2.1.3}
$$

代入分解后的形式，得到：

$$
\begin{aligned}
\mathcal{L}_{\text{ELBO}}
= \mathbb{E}_{q_{\boldsymbol{\theta}}(\mathbf{z}_{1:L}|\mathbf{x})} \left[
\log \frac{
p(\mathbf{z}_L) \prod_{i=2}^{L} p_{\boldsymbol{\phi}}(\mathbf{z}_{i-1} | \mathbf{z}_i)   p_{\boldsymbol{\phi}}(\mathbf{x} | \mathbf{z}_1)
}{
q_{\boldsymbol{\theta}}(\mathbf{z}_1|\mathbf{x}) \prod_{i=2}^{L} q_{\boldsymbol{\theta}}(\mathbf{z}_i|\mathbf{z}_{i-1})
}
\right].
\end{aligned}
$$

这一层次 ELBO 可分解为具有明确含义的若干项，包括重建项、相邻层匹配项，以及使顶层分布接近先验的 KL 正则项。

从浅层网络到深层网络的飞跃改变了机器学习，类似的思想也改变了生成模型。HVAE 展示了利用深层堆叠结构生成数据的能力。这种分层结构的概念是现代生成建模的基石，在基于 score 的方法（[第 3.4 节](../03-score-based-perspective/04-noise-conditional-score-networks.md#sec-smld)）和归一化流（[第 5.1 节](../05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#sec-flow-based-method)）中再次出现。其中的关键认识简单而有力：

<span id="observation-2-1-1"></span>

**观察 2.1.1**

堆叠多个层使模型能够逐步生成数据，从粗略结构开始，在每一步添加更精细的细节。这个过程使捕捉高维数据的复杂结构变得容易得多。

**为什么仅加深单层潜变量 VAE（flat VAE）中的网络还不够。** 标准的单层潜变量 VAE 存在两个根本限制，单纯增加编码器和解码器的深度无法解决它们。

第一个限制是变分分布族。在标准 VAE 中，

$$
q_{\boldsymbol{\theta}}({\mathbf{z}}| {\mathbf{x}})=\mathcal N \big({\mathbf{z}};\,\boldsymbol{\mu}_{{\boldsymbol{\theta}}}({\mathbf{x}}), \operatorname{diag}(\sigma_\theta^2({\mathbf{x}}))\big),
$$

因此，对于每个固定的 ${\mathbf{x}}$，编码器后验都是具有对角协方差的单个 Gaussian 分布。增加网络深度能提高 $\boldsymbol{\mu}_{{\boldsymbol{\theta}}}$ 和 $\sigma_{\boldsymbol{\theta}}$ 的准确性，却不会扩展这个分布族；即使采用完整协方差，它仍然只是一个单峰椭球。当 $p_{\boldsymbol{\phi}}({\mathbf{z}}| {\mathbf{x}})$ 具有多个峰时，这一分布族无法与之匹配，从而使 ELBO 变松，并削弱推断效果。要解决这一问题，需要更丰富的后验分布族，仅加深网络并不足够。

第二，如果解码器的表达能力过强，模型可能出现后验坍塌。为了理解其原因，回顾 VAE 的目标：

$$
\begin{aligned}
&\mathbb{E}_{p_{\text{data}}(\mathbf{x})}[\mathcal L_{\mathrm{ELBO}}({\mathbf{x}})]\\
&\quad\quad={\mathbb{E}}_{p_{\text{data}}(\mathbf{x})q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}})}[\log p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}})]
-\mathbb{E}_{p_{\text{data}}(\mathbf{x})}\big[\mathcal{D}_\mathrm{KL} \big(q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}}) \| p({\mathbf{z}})\big)\big]\\
&\quad\quad={\mathbb{E}}_{p_{\text{data}}(\mathbf{x})q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}})}[\log p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}})]-\mathcal I_q({\mathbf{x}};{\mathbf{z}}) - \mathcal{D}_\mathrm{KL}(q_{\boldsymbol{\theta}}({\mathbf{z}}) \| p({\mathbf{z}})),
\end{aligned}
$$

其中，$\mathcal I_q({\mathbf{x}};{\mathbf{z}})$ 是互信息，定义为

$$
\mathcal I_q({\mathbf{x}};{\mathbf{z}})
={\mathbb{E}}_{q({\mathbf{x}},{\mathbf{z}})} \Big[\log \tfrac{q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}})}{q({\mathbf{z}})}\Big]
={\mathbb{E}}_{p_{\mathrm{data}}({\mathbf{x}})} \big[\mathcal{D}_\mathrm{KL}(q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}}) \| q({\mathbf{z}}))\big],
$$

而聚合后验为 $q_{\boldsymbol{\theta}}({\mathbf{z}})= \int p_{\mathrm{data}}({\mathbf{x}}) q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}}$。

如果解码器类无需使用 ${\mathbf{z}}$ 就能很好地建模数据（即其中存在某个 $p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}})=r({\mathbf{x}})$，它与 $p_{\mathrm{data}}$ 接近），那么 ELBO 的一个最大化解会令 $q_{\boldsymbol{\theta}}({\mathbf{z}}|{\mathbf{x}})=p({\mathbf{z}})$，因而 $\mathcal I_q({\mathbf{x}};{\mathbf{z}})=0$ 且 $q_{\boldsymbol{\theta}}({\mathbf{z}})=p({\mathbf{z}})$。这种“忽略 ${\mathbf{z}}$”的解不会因网络加深而消失：（1）学到的编码与 ${\mathbf{x}}$ 独立，因此不包含对下游任务有用的、依赖于数据的结构；（2）以 ${\mathbf{z}}$ 为条件或在其空间中移动，都不会影响生成样本，因此无法实现可控生成。

**层次结构带来了什么改变？** HVAE 引入多个潜变量层级：

$$
p_{\boldsymbol{\phi}}({\mathbf{x}},{\mathbf{z}}_{1:L})=p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}}_1)\prod_{i=2}^L p_{\boldsymbol{\phi}}({\mathbf{z}}_{i-1}|{\mathbf{z}}_i) p({\mathbf{z}}_L),
$$

其 ELBO 为

$$
\begin{aligned}
\mathcal L_{\mathrm{ELBO}}({\mathbf{x}})
&= {\mathbb{E}}_{q_{\boldsymbol{\theta}}({\mathbf{z}}_1|{\mathbf{x}})}[\log p_{\boldsymbol{\phi}}({\mathbf{x}}|{\mathbf{z}}_1)]
\\
&\quad - {\mathbb{E}}_{q_{\boldsymbol{\theta}}({\mathbf{z}}_{1:2}|{\mathbf{x}})}
\big[\log q_{\boldsymbol{\theta}}({\mathbf{z}}_1|{\mathbf{x}})-\log p_{\boldsymbol{\phi}}({\mathbf{z}}_1|{\mathbf{z}}_2)\big]
\\
&\quad - \sum_{i=2}^{L-1}
{\mathbb{E}}_{q_{\boldsymbol{\theta}}({\mathbf{z}}_{1:i+1}|{\mathbf{x}})}
\big[\log q_{\boldsymbol{\theta}}({\mathbf{z}}_i|{\mathbf{z}}_{i-1})-\log p_{\boldsymbol{\phi}}({\mathbf{z}}_i|{\mathbf{z}}_{i+1})\big]
\\
&\quad - {\mathbb{E}}_{q_{\boldsymbol{\theta}}({\mathbf{z}}_{L-1}|{\mathbf{x}})}
\Big[\mathcal{D}_{\mathrm{KL}}(q_{\boldsymbol{\theta}}({\mathbf{z}}_L|{\mathbf{z}}_{L-1}) \| p({\mathbf{z}}_L))\Big].
\end{aligned}
$$

这里记 ${\mathbb{E}}_q := {\mathbb{E}}_{p_{\mathrm{data}}({\mathbf{x}}) q_{{\boldsymbol{\theta}}}({\mathbf{z}}_{1:L}|{\mathbf{x}})}$。每个潜变量层级仅与层次结构中相邻的层级相互作用：编码器通过 $q_{\boldsymbol{\theta}}({\mathbf{z}}_i|{\mathbf{z}}_{i-1})$ 向上传递信息，解码器则通过 $p_{\boldsymbol{\phi}}({\mathbf{z}}_{i-1}|{\mathbf{z}}_i)$ 向下传递信息。因此，ELBO 分解为重建项、相邻层匹配项，以及使顶层分布接近先验的 KL 正则项。这些性质源于潜变量的层次图结构，单纯加深单层潜变量 VAE 中的网络无法带来这些性质。

**接下来会遇到什么？** HVAE 通过多个潜变量层扩展 VAE 框架，以增强表达能力，但其训练也面临特有的挑战。由于编码器和解码器必须联合优化，学习过程会变得不稳定：较低层和解码器已经能够重建 ${\mathbf{x}}$，留给较高层潜变量的有效信号很少。此外，传递到更深层变量的梯度信息往往间接而微弱，使它们难以发挥实质作用。另一个困难在于平衡模型容量，因为表达能力过强的条件分布可能主导重建任务，抑制较高层潜变量的作用。

有趣的是，深层分层结构的思想在变分扩散模型中得到了更有力的体现，我们将在[第 2.2 节](02-ddpm.md#sec-ddpm) 探讨这一主题。扩散模型继承了 HVAE 的渐进结构，同时巧妙地避开其主要弱点。通过固定编码过程，仅学习用于生成的逆向过程，它们获得了更好的稳定性与更灵活的建模能力，使生成结果的质量显著提升。

为简化记号，我们将不再沿用 VAE 中以 $q$ 表示编码器、以 $p$ 表示生成器的常见约定。为避免歧义，我们统一用 $p$ 表示分布，并始终通过适当的下标或上标指定其作用，在上下文中加以说明。

[^1]: 这里研究的限制针对的是前文介绍的常见 VAE 设置：通过平方重建误差来训练简单的 Gaussian 解码器。在这一设置下，总体最优的解码器返回条件均值。如果几个视觉上不同的输入对应于相互重叠的潜空间区域，这种平均可能表现为模糊。这一结论不应被理解为适用于所有 VAE 架构或观测模型的普遍陈述。

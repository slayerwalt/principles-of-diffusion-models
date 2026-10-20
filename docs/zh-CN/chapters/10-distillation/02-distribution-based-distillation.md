# 10.2 基于分布的蒸馏 {#sec-vsd}

多项工作同期以不同名称探索了这种基于分布的蒸馏，包括 Distributional Matching Distillation（DMD）（[Yin et al., 2024](../../references.md#ref-yin2024one)；[Yin et al., 2024](../../references.md#ref-yin2024improved)）、Variational Score Distillation（VSD）（[Poole et al., 2023](../../references.md#ref-pooledreamfusion)；[Wang et al., 2023](../../references.md#ref-wang2023prolificdreamer)；[Luo et al., 2023](../../references.md#ref-luo2023diff)；[Lu and Song, 2024](../../references.md#ref-lu2024simplifying)）以及 Score Identity Distillation（SiD）（[Zhou et al., 2024](../../references.md#ref-zhou2024score)）。尽管技术细节有所不同，它们遵循同一原则：训练一个生成器，使其前向加噪后的边缘分布与教师的对应分布匹配。我们以 VSD 为代表进行介绍，因为其他方法遵循相似的原则。[^1]

## 10.2.1 作为代表方法的 VSD 表述 {#formulation-of-vsd}

**前向过程。** 令 $\{p_t\}_{t \in [0,T]}$ 表示由以下过程诱导的前向扩散边缘密度：

$$
{\mathbf{x}}_t = \alpha_t {\mathbf{x}}_0 + \sigma_t \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}),
$$

其初始分布为 $p_0 = p_{\mathrm{data}}$。相应地，令 $p_0^{\boldsymbol{\theta}}$ 表示确定性单步生成器 ${\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})$ 从潜变量 ${\mathbf{z}} \sim p_{\mathrm{prior}}({\mathbf{z}})$ 生成的合成样本的分布。定义 $\{p_t^{\boldsymbol{\theta}}\}_{t \in [0,T]}$ 为对 $p_0^{\boldsymbol{\theta}}$ 应用相同前向扩散过程得到的边缘密度，即

<span id="eq-vsd-forward"></span>

$$
\begin{aligned}
{\mathbf{x}}_t^{\boldsymbol{\theta}} := \alpha_t {\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}}) + \sigma_t \boldsymbol{\epsilon},
\end{aligned}
\tag{10.2.1}
$$

其中 ${\mathbf{z}} \sim p_{\mathrm{prior}}$，且 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$。因此，$p_t$ 和 $p_t^{\boldsymbol{\theta}}$ 使用相同的 Gaussian 扩散核 $p_t({\mathbf{x}}_t| {\mathbf{x}}_0)$，但起始分布不同（分别为 $p_{\mathrm{data}}$ 与单步合成样本的分布 $p_0^{\boldsymbol{\theta}}$）。

**训练目标与梯度。** 文献通常采用 KL散度匹配分布 $p_t$ 和 $p_t^{\boldsymbol{\theta}}$，一般通过最小化

$$
\begin{aligned}
\mathcal{L}_{\text{VSD}}(\boldsymbol{\theta}) :=\mathbb{E}_t \left[ \omega(t)  \mathcal{D}_{\text{KL}}(p_t^{\boldsymbol{\theta}} \, \| \, p_t) \right]
= \mathbb{E}_{t, {\mathbf{z}}, \boldsymbol{\epsilon}} \left[ \omega(t) \left( \log p_t^{\boldsymbol{\theta}}({\mathbf{x}}_t^{\boldsymbol{\theta}}) - \log p_t({\mathbf{x}}_t^{\boldsymbol{\theta}}) \right) \right],
\end{aligned}
$$

其中 $\omega(t)\geq 0$ 是依赖时间的权重函数。我们将在[第 10.2.3 节](02-distribution-based-distillation.md#subsec-vsd-discussion)讨论 KL散度为何在分布层面的蒸馏中具有特殊地位。

如（[Wang et al., 2023](../../references.md#ref-wang2023prolificdreamer)）所示，在总体层面，当 $p_0^{\boldsymbol{\theta}^*}=p_{\mathrm{data}}$ 时达到全局最优，这意味着生成器分布与数据分布匹配。因此，VSD 目标为学习 $p_{\mathrm{data}}$ 提供了一个具有理论依据的分布匹配损失。

然而，用密度表述的目标缺少高效的训练机制。所幸，对 $\boldsymbol{\theta}$ 求梯度后，可以得到[公式 (10.2.2)](02-distribution-based-distillation.md#eq-vsd-grad)中的表达式，以下命题对此作出概括。为简化记号，我们用 $\hat{\mathbf{x}}_t := {\mathbf{x}}_t^{\boldsymbol{\theta}}$ 表示[公式 (10.2.1)](02-distribution-based-distillation.md#eq-vsd-forward)中定义的量。

<span id="vsd-grad"></span>

**命题 10.2.1：关于 $\boldsymbol{\theta}$ 的梯度——$\mathcal{L}_{\text{VSD}}$**

我们有

<span id="eq-vsd-grad"></span>

$$
\begin{aligned}
        \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{VSD}}(\boldsymbol{\theta}) = \mathbb{E}_{t, {\mathbf{z}}, \boldsymbol{\epsilon}} \left[ \omega(t)\alpha_t  \left(
\nabla_{{\mathbf{x}}} \log p_t^{\boldsymbol{\theta}}(\hat{\mathbf{x}}_t) - \nabla_{{\mathbf{x}}} \log p_t(\hat{\mathbf{x}}_t)
\right) \cdot \partial_{\boldsymbol{\theta}} {\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}}) \right].
\end{aligned}
\tag{10.2.2}
$$

**命题的证明。**

推导使用链式法则：

$$
\begin{aligned}
&\nabla_{{\boldsymbol{\theta}}}\mathcal{L}_{\text{VSD}}({\boldsymbol{\theta}})
=
\nabla_{{\boldsymbol{\theta}}}{\mathbb{E}}_t
\left[
\omega(t)\mathcal{D}_{\mathrm{KL}}(p_t^{{\boldsymbol{\theta}}}\|p_t)
\right]
\\
&= {\mathbb{E}}_{t,{\mathbf{z}},{\boldsymbol{\epsilon}}}\!\left[
\omega(t)\,
\partial_{\boldsymbol{\theta}}
\big(\log p_t^{{\boldsymbol{\theta}}}(\hat{\mathbf{x}}_t)-\log p_t(\hat{\mathbf{x}}_t)\big)
\right] \\
&= {\mathbb{E}}_{t,{\mathbf{z}},{\boldsymbol{\epsilon}}} \left[
\omega(t)\left(
\underbrace{\partial_{{\boldsymbol{\theta}}}\log p_t^{{\boldsymbol{\theta}}}(\hat{\mathbf{x}}_t)}_{\text{first}}
+ (\nabla_{{\mathbf{x}}}\log p_t^{{\boldsymbol{\theta}}}(\hat{\mathbf{x}}_t))^\top \partial_{{\boldsymbol{\theta}}}\hat{\mathbf{x}}_t
- (\nabla_{{\mathbf{x}}}\log p_t(\hat{\mathbf{x}}_t))^\top \partial_{{\boldsymbol{\theta}}}\hat{\mathbf{x}}_t
\right)\right].
\end{aligned}
$$

根据 score 函数恒等式，第一项为零：

$$
{\mathbb{E}}_{\hat{\mathbf{x}}_t\sim p_t^{{\boldsymbol{\theta}}}}\!\left[\partial_{{\boldsymbol{\theta}}}\log p_t^{{\boldsymbol{\theta}}}(\hat{\mathbf{x}}_t)\right]
= \int \partial_{{\boldsymbol{\theta}}} p_t^{{\boldsymbol{\theta}}}({\mathbf{x}})  \mathop{}\!\mathrm{d}{\mathbf{x}}
= \partial_{{\boldsymbol{\theta}}}  \int p_t^{{\boldsymbol{\theta}}}({\mathbf{x}})  \mathop{}\!\mathrm{d}{\mathbf{x}}
= \partial_{{\boldsymbol{\theta}}}(1)=0.
$$

使用重参数化 $\hat{\mathbf{x}}_t=\alpha_t{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{z}})+\sigma_t{\boldsymbol{\epsilon}}$ 可得 $\partial_{{\boldsymbol{\theta}}}\hat{\mathbf{x}}_t=\alpha_t\partial_{{\boldsymbol{\theta}}}{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{z}})$，因此

$$
\nabla_{{\boldsymbol{\theta}}}\mathcal{L}_{\text{VSD}}({\boldsymbol{\theta}})
= {\mathbb{E}}_{t,{\mathbf{z}},{\boldsymbol{\epsilon}}} \left[\omega(t) \alpha_t
\big(\nabla_{{\mathbf{x}}}\log p_t^{{\boldsymbol{\theta}}}(\hat{\mathbf{x}}_t)-\nabla_{{\mathbf{x}}}\log p_t(\hat{\mathbf{x}}_t)\big)^\top
\partial_{{\boldsymbol{\theta}}}{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{z}})\right].
$$

这就证明了[公式 (10.2.2)](02-distribution-based-distillation.md#eq-vsd-grad)。详细推导见[第 D.5 节](../../appendices/d-supplementary-proofs/05-fast-generators.md#app-sec-flow-map)。

可以看到，在对 $\boldsymbol{\theta}$ 求梯度时，score 函数自然出现。因此，我们需要近似单步生成器的 score $\nabla_{{\mathbf{x}}}\log p_t^{\boldsymbol{\theta}}(\hat{\mathbf{x}}_t)$，以及数据分布的 score $\nabla_{{\mathbf{x}}}\log p_t(\hat{\mathbf{x}}_t)$，下一小节将详细介绍。

## 10.2.2 VSD 的训练流程 {#subsec-vsd-training}

现有工作（[Yin et al., 2024](../../references.md#ref-yin2024one)；[Yin et al., 2024](../../references.md#ref-yin2024improved)；[Poole et al., 2023](../../references.md#ref-pooledreamfusion)；[Wang et al., 2023](../../references.md#ref-wang2023prolificdreamer)；[Luo et al., 2023](../../references.md#ref-luo2023diff)；[Lu and Song, 2024](../../references.md#ref-lu2024simplifying)）通常采用双层优化方法解决这一问题：用来自 ${\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})$ 的样本训练新的扩散模型，以近似 $\nabla_{{\mathbf{x}}} \log p_t^{\boldsymbol{\theta}}(\hat{\mathbf{x}}_t)$；同时使用预训练扩散模型，为不可直接计算的 oracle score 函数 $\nabla_{{\mathbf{x}}} \log p_t(\hat{\mathbf{x}}_t)$ 在合成样本 $\hat{\mathbf{x}}_t$ 上提供替代值。更具体地，训练在以下两个阶段之间交替进行：

-   **Score 估计阶段。** 固定 $\boldsymbol{\theta}$。令 $\hat{{\mathbf{x}}}_0={\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})$，且 $\hat{{\mathbf{x}}}_t=\alpha_t \hat{{\mathbf{x}}}_0+\sigma_t{\boldsymbol{\epsilon}}$，其中 ${\mathbf{z}}\sim p_{\mathrm{prior}}$、${\boldsymbol{\epsilon}}\sim\mathcal{N}(\boldsymbol{0},{\mathbf{I}})$。利用已知的 Gaussian 扩散核，对 $s_{\boldsymbol{\zeta}}$ 进行 DSM 训练，该核为 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$：

    $$
    \mathcal{L}_{\text{DSM}}(\boldsymbol{\zeta};\boldsymbol{\theta})
        ={\mathbb{E}}_{t,{\mathbf{z}},{\boldsymbol{\epsilon}}}\Big\|\,{\mathbf{s}}_{\boldsymbol{\zeta}}(\hat{{\mathbf{x}}}_t,t)
        - \nabla_{{\mathbf{x}}_t}\log p_t(\hat{{\mathbf{x}}}_t|\hat{{\mathbf{x}}}_0)\,\Big\|^2,
    $$

    在最优处得到 ${\mathbf{s}}_{\boldsymbol{\zeta}}(\cdot,t)\approx \nabla_{{\mathbf{x}}}\log p_t^{\boldsymbol{\theta}}(\cdot)$（对于固定的 $\boldsymbol{\theta}$）。

-   **生成器更新阶段。** 冻结 $s_{\boldsymbol{\zeta}}$（停止梯度）后，使用[公式 (10.2.2)](02-distribution-based-distillation.md#eq-vsd-grad)中的梯度更新 $\boldsymbol{\theta}$，其中各 score 项用相应的替代值表示：

    $$
    {\mathbf{s}}_{\boldsymbol{\zeta}}(\hat{\mathbf{x}}_t,t)\approx \nabla_{{\mathbf{x}}}\log p_t^{\boldsymbol{\theta}}(\hat{\mathbf{x}}_t), \,\, \text{and} \,\, {\mathbf{s}}_{\boldsymbol{\phi}^\times}(\hat{{\mathbf{x}}}_t,t)\approx \nabla_{{\mathbf{x}}}\log p_t(\hat{\mathbf{x}}_t)\ \ \text{(teacher)}.
    $$

    于是[公式 (10.2.2)](02-distribution-based-distillation.md#eq-vsd-grad)近似变为：

    $$
    \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{VSD}}(\boldsymbol{\theta})
        \approx \mathbb{E}_{t,{\mathbf{z}},\boldsymbol{\epsilon}}\Big[\omega(t)\alpha_t
          \big({\mathbf{s}}_{\boldsymbol{\zeta}}(\hat{{\mathbf{x}}}_t,t) - {\mathbf{s}}_{\boldsymbol{\phi}^\times}(\hat{{\mathbf{x}}}_t,t)\big)^{\top}
          \partial_{\boldsymbol{\theta}}{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})
        \Big].
    $$

这两个阶段在整个训练过程中交替进行。理想情况下，辅助模型跟踪学生当前的 score ${\mathbf{s}}_{\boldsymbol{\zeta}}(\cdot,t)\approx\nabla_{{\mathbf{x}}}\log p_t^{\boldsymbol{\theta}}(\cdot)$，预训练教师则提供目标 score ${\mathbf{s}}_{\boldsymbol{\phi}^\times}(\cdot,t)\approx\nabla_{{\mathbf{x}}}\log p_t(\cdot)$ 的近似。在总体层面，如果这两个 score 函数在满足 $\sigma_t>0$ 和 $\alpha_t\neq 0$ 的某个加噪水平上一致，则相应的归一化密度也一致，即 $p_t^{\boldsymbol{\theta}}=p_t$。由于在 $\alpha_t\neq 0$ 时 Gaussian 加噪具有单射性，这进一步意味着 $p_0^{\boldsymbol{\theta}}=p_{\mathrm{data}}$。

实践中，这两个 score 都由有限的、学习得到的网络表示，因此这些等式刻画的是理想的总体解，并不构成交替优化的保证。

## 10.2.3 超越 KL：一般散度，以及 KL 的特殊性 {#subsec-vsd-discussion}

VSD 的分布匹配原则并不依赖于 KL散度。原则上，可以使用一般的 $f$ 散度（见[公式 (1.1.4)](../01-deep-generative-modeling/01-what-is-deep-generative-modeling.md#eq-f-div)）：

$$
\mathcal{D}_f(p_t^{{\boldsymbol{\theta}}}\|p_t)
=
\int p_t({\mathbf{x}})\,
f\!\left(
\frac{p_t^{{\boldsymbol{\theta}}}({\mathbf{x}})}{p_t({\mathbf{x}})}
\right)\mathop{}\!\mathrm{d} {\mathbf{x}}.
$$

反向 KL散度特别方便，原因在于其梯度结构。定义密度比

$$
r_t({\mathbf{x}})
:=
\frac{p_t^{{\boldsymbol{\theta}}}({\mathbf{x}})}{p_t({\mathbf{x}})}.
$$

对于二次可微的凸函数 $f$ 以及正密度，路径梯度可写为

$$
\nabla_{{\boldsymbol{\theta}}}\mathcal{D}_f(p_t^{{\boldsymbol{\theta}}}\|p_t)
=
{\mathbb{E}}_{\hat{\mathbf{x}}_t\sim p_t^{{\boldsymbol{\theta}}}}
\left[
r_t(\hat{\mathbf{x}}_t)f''\!\big(r_t(\hat{\mathbf{x}}_t)\big)
\big(
\nabla_{{\mathbf{x}}}\log p_t^{{\boldsymbol{\theta}}}(\hat{\mathbf{x}}_t)
-
\nabla_{{\mathbf{x}}}\log p_t(\hat{\mathbf{x}}_t)
\big)^\top
\partial_{{\boldsymbol{\theta}}}\hat{\mathbf{x}}_t
\right].
$$

从概念上看，该梯度由三部分组成：

$$
\begin{aligned}
\nabla_{{\boldsymbol{\theta}}}\mathcal{D}_f
=
{\mathbb{E}}\Big[
\underbrace{r_t f''(r_t)}_{\text{density-ratio weighting}}
\times
\underbrace{
\big(
\nabla_{{\mathbf{x}}}\log p_t^{{\boldsymbol{\theta}}}
-
\nabla_{{\mathbf{x}}}\log p_t
\big)^\top
}_{\text{score mismatch}}
\times
\underbrace{
\partial_{{\boldsymbol{\theta}}}\hat{\mathbf{x}}_t
}_{\text{parameter sensitivity}}
\Big].
\end{aligned}
$$

第一项决定在*何处、以多大强度*更新，第二项给出 score 不匹配的方向，最后一项将这一来自样本空间的信号映射回生成器参数。

这一表达式揭示了一个有用的共同结构：所有这类 $f$ 散度都使用相同的 *score 不匹配方向*，

$$
\nabla_{{\mathbf{x}}}\log p_t^{{\boldsymbol{\theta}}}({\mathbf{x}})
-
\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})
=
\nabla_{{\mathbf{x}}}\log r_t({\mathbf{x}}),
$$

而散度的选择则通过 $r_t f''(r_t)$ 改变不同区域的权重。因此，对于通常采用的严格凸 $f$，总体层面的分布匹配条件仍为 $p_t^{{\boldsymbol{\theta}}}=p_t$，但散度的选择会通过 $r_t f''(r_t)$ 改变优化过程中不同密度比范围的权重。

对于 VSD 使用的反向 KL散度，有 $f(r)=r\log r$，因此

$$
r f''(r)=1.
$$

密度比权重消失，从而得到[公式 (10.2.2)](02-distribution-based-distillation.md#eq-vsd-grad)中的 score 差梯度。因此，一旦学生和目标的 score 可用，反向 KL散度训练就不再需要额外的逐点密度比估计器。相比之下，一般的 $f$ 散度需要因子 $r_t f''(r_t)$；尽管 score 差给出了 $\nabla_{{\mathbf{x}}}\log r_t$，但对于隐式生成器，比值 $r_t$ 本身无法直接获得，通常需要额外估计。

例如，正向 KL散度对应于 $f(r)=-\log r$，此时 $r f''(r)=1/r$。因此，它会给模型密度相对于目标密度较小的区域赋予更大的权重。这说明，不同散度可以具有相同的底层 score 不匹配方向，同时侧重不同的密度比范围。

最后，这种 score 不匹配的视角与[第 B.3 节](../../appendices/b-density-evolution/03-wasserstein-gradient-flows.md#sec-wgf-distribution-level-training)中的 Wasserstein 梯度流视角直接相连。在分布层面，反向 KL散度的梯度流按目标 score 减去模型 score 的方向移动粒子。VSD 的参数梯度包含相同的 score 差，并通过生成器的 Jacobian 矩阵 $\partial_{{\boldsymbol{\theta}}}\hat{\mathbf{x}}_t$ 将其拉回参数空间。因此，Wasserstein 视角描述理想的分布空间运动，而 VSD 通过更新有限维的共享生成器来实现这一运动。

[^1]: 从历史上看，VSD（[Wang et al., 2023](../../references.md#ref-wang2023prolificdreamer)）最初用于文本到三维生成，其中预训练的二维扩散模型通过可微渲染器提供分布监督。将其特化为单粒子（Dirac）情形，可以恢复基本的 SDS（[Poole et al., 2023](../../references.md#ref-pooledreamfusion)）更新，但需考虑相应的时间权重与预测参数化约定。

# B.3 （选读）作为分布层面训练的 Wasserstein 梯度流 {#sec-wgf-distribution-level-training}

现在暂时离开扩散特有的动力学，从更广泛的分布层面看待深度生成建模。前文已说明，连续性方程描述粒子移动时概率分布如何移动。同样的语言也可以用于理解深度生成模型的训练：训练算法更新模型参数，更新后的生成器移动生成粒子，而粒子的运动改变其诱导的模型分布 $p_{{\boldsymbol{\phi}}}$。

从这一视角看，可以按照模型分布 $p_{{\boldsymbol{\phi}}}$ 如何被推向数据分布 $p_{\mathrm{data}}$ 来组织不同生成建模方法。Wasserstein 梯度流为描述这一运动提供了特别清晰的方式：它将训练视为分布空间中概率质量的理想化流动，再用有限维神经网络更新近似这一流动。

神经生成器 ${\mathbf{G}}_{{\boldsymbol{\phi}}}$ 同时定义函数和概率分布。如果

$$
{\mathbf{z}}\sim p_{\mathrm{prior}},
    \qquad
    {\mathbf{x}}={\mathbf{G}}_{{\boldsymbol{\phi}}}({\mathbf{z}}),
$$

那么生成分布就是推前

$$
p_{{\boldsymbol{\phi}}}:=({\mathbf{G}}_{{\boldsymbol{\phi}}})_\# p_{\mathrm{prior}}.
$$

从生成建模视角看，我们最终关心的是 $p_{{\boldsymbol{\phi}}}$ 如何接近数据分布 $p_{\mathrm{data}}$。

由此引出一个分布层面的问题：

<span id="question-B-3-1"></span>

**问题 B.3.1**

能否先描述将模型分布 $p_{{\boldsymbol{\phi}}}$ 移向 $p_{\mathrm{data}}$ 的理想方式，再更新神经网络参数，使生成器近似实现这一运动？

本选读节中，$\tau$ 表示*训练时间*，即离散训练迭代索引的理想化连续时间版本。它不应与扩散／加噪时间混淆。我们用 ${\boldsymbol{\phi}}_\tau$ 表示训练时刻 $\tau$ 的参数，用 $p_{{\boldsymbol{\phi}}_\tau}$ 表示相应模型分布。

**通过移动粒子移动模型分布。** 我们先描述理想的分布层面演化，再考虑神经生成器的有限维参数化。假设当前模型分布中的粒子按训练时间速度场 ${\mathbf{w}}_\tau$ 运动：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_\tau}{\mathop{}\!\mathrm{d} \tau}
    =
    {\mathbf{w}}_\tau({\mathbf{x}}_\tau),
    \qquad
    {\mathbf{x}}_\tau\sim p_{{\boldsymbol{\phi}}_\tau}.
$$

则模型分布按照连续性方程演化

$$
\partial_\tau p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}})
    +
    \nabla_{{\mathbf{x}}}\cdot
    \big(
        p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}}){\mathbf{w}}_\tau({\mathbf{x}})
    \big)
    =
    0.
$$

因此，${\mathbf{w}}_\tau$ 描述每个生成粒子如何运动，连续性方程描述整个生成分布如何运动[^1]。

**选择速度：分布空间中的梯度下降。** 现在需要选择速度场 ${\mathbf{w}}_\tau$，使 $p_{{\boldsymbol{\phi}}_\tau}$ 移向 $p_{\mathrm{data}}$。令

$$
\mathcal{D}(p_{{\boldsymbol{\phi}}_\tau},p_{\mathrm{data}})
$$

表示当前模型分布与数据分布之间的差异。*Wasserstein 梯度流*（[Jordan et al., 1998](../../references.md#ref-jordan1998variational)；[Ambrosio et al., 2005](../../references.md#ref-ambrosio2005gradient)）在移动概率质量的几何下，选择能够最直接减小这一差异的粒子速度。

为选择速度场，先考察当前分布受到无穷小扰动时，差异如何变化。将 $\mathcal{D}(p,p_{\mathrm{data}})$ 视为分布 $p$ 的全局能量。其一阶变分给出逐点能量势，说明对当前分布而言，哪些空间区域代价高，哪些区域更有利。

具体而言，对于小扰动 $p+\epsilon h$，其中 $\epsilon>0$ 很小且 $\int h({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}=0$，一阶变分定义为

$$
\mathcal{D}(p+\epsilon h,p_{\mathrm{data}})
    =
    \mathcal{D}(p,p_{\mathrm{data}})
    +
    \epsilon
    \int
    \frac{\delta\mathcal{D}}{\delta p}
    (p,p_{\mathrm{data}})({\mathbf{x}})\,
    h({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}
    +
    \mathop{\mathrm{o}}(\epsilon).
$$

这里，$p$ 是分布的哑变量。取变分后，在当前模型分布 $p=p_{{\boldsymbol{\phi}}_\tau}$ 处求值。

将所得逐点能量势定义为

$$
E_\tau({\mathbf{x}})
    :=
    \frac{\delta\mathcal{D}}{\delta p}
    (p_{{\boldsymbol{\phi}}_\tau},p_{\mathrm{data}})({\mathbf{x}}).
$$

直观上，若 $E_\tau({\mathbf{x}})$ 较大，在 ${\mathbf{x}}$ 附近放置更多概率质量会增大差异；若 $E_\tau({\mathbf{x}})$ 较小，将质量移向 ${\mathbf{x}}$ 更有利。因此，Wasserstein 梯度流速度使粒子沿这一势的下降方向运动：

$$
{\mathbf{w}}_\tau({\mathbf{x}})
    =
    -
    \nabla_{{\mathbf{x}}}E_\tau({\mathbf{x}}).
$$

采用这一选择，有

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} \tau}
    \mathcal{D}(p_{{\boldsymbol{\phi}}_\tau},p_{\mathrm{data}})
    =
    -
    \int
    p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}})
    \left\|
        \nabla_{{\mathbf{x}}}E_\tau({\mathbf{x}})
    \right\|_2^2
    \mathop{}\!\mathrm{d}{\mathbf{x}}
    \le 0.
$$

因此，Wasserstein 梯度流是概率分布的 $2$-Wasserstein 几何中的梯度下降[^2]。全局差异 $\mathcal{D}$ 定义整个分布的能量，一阶变分 $E_\tau$ 定义粒子看到的局部能量景观，而 $-\nabla_{{\mathbf{x}}}E_\tau$ 给出减小差异的理想粒子速度。

**用神经网络实现分布速度。** 上述速度是分布层面的对象，说明生成粒子应如何移动。但神经生成器不能独立移动每个粒子，只能通过对 ${\boldsymbol{\phi}}$ 的共享参数更新来移动粒子。

在训练时刻 $\tau$，当前生成器产生粒子

$$
{\mathbf{x}}_i
    =
    {\mathbf{G}}_{{\boldsymbol{\phi}}_\tau}({\mathbf{z}}_i),
    \qquad
    {\mathbf{z}}_i\sim p_{\mathrm{prior}},
    \qquad
    i=1,\ldots,N.
$$

一个小的 Wasserstein 步将这些粒子移到

$$
\widetilde{{\mathbf{x}}}_i
    =
    {\mathbf{x}}_i+\eta{\mathbf{w}}_\tau({\mathbf{x}}_i),
$$

其中 $\eta>0$ 是小步长。随后更新神经网络，使其新输出匹配这些移动后的粒子：

$$
{\boldsymbol{\phi}}_{\tau+\eta}
    \approx
    \arg\min_{{\boldsymbol{\phi}}}
    \frac{1}{N}
    \sum_{i=1}^N
    \left\|
        {\mathbf{G}}_{{\boldsymbol{\phi}}}({\mathbf{z}}_i)
        -
        \operatorname{sg}
        \big(
            {\mathbf{x}}_i+\eta{\mathbf{w}}_\tau({\mathbf{x}}_i)
        \big)
    \right\|_2^2.
$$

这里，$\operatorname{sg}(\cdot)$ 表示停止梯度，因此移动后的粒子被视为固定目标。

对于小参数更新 $\Delta{\boldsymbol{\phi}}$，有线性化

$$
{\mathbf{G}}_{{\boldsymbol{\phi}}_\tau+\Delta{\boldsymbol{\phi}}}({\mathbf{z}}_i)
    \approx
    {\mathbf{G}}_{{\boldsymbol{\phi}}_\tau}({\mathbf{z}}_i)
    +
    \big(\partial_{{\boldsymbol{\phi}}}{\mathbf{G}}_{{\boldsymbol{\phi}}_\tau}({\mathbf{z}}_i)\big)\Delta{\boldsymbol{\phi}}.
$$

因此，参数更新近似求解

$$
\min_{\Delta{\boldsymbol{\phi}}}
    \frac{1}{N}
    \sum_{i=1}^N
    \left\|
        \big(\partial_{{\boldsymbol{\phi}}}{\mathbf{G}}_{{\boldsymbol{\phi}}_\tau}({\mathbf{z}}_i)\big)\Delta{\boldsymbol{\phi}}
        -
        \eta{\mathbf{w}}_\tau({\mathbf{x}}_i)
    \right\|_2^2.
$$

换言之，Wasserstein 梯度流指定所需的分布层面粒子运动，神经网络训练则将这一运动投影到通过改变有限维参数 ${\boldsymbol{\phi}}$ 能够实现的运动集合上。

**正向 KL散度训练：数据侧的似然学习。** 一个经典的分布层面目标是数据侧的、即正向 KL散度：

$$
\mathcal{D}_{\mathrm{FKL}}(p_{{\boldsymbol{\phi}}},p_{\mathrm{data}})
    :=
    \mathcal D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{{\boldsymbol{\phi}}})
    =
    \int
    p_{\mathrm{data}}({\mathbf{x}})
    \log
    \frac{p_{\mathrm{data}}({\mathbf{x}})}{p_{{\boldsymbol{\phi}}}({\mathbf{x}})}
    \mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

由于 $p_{\mathrm{data}}$ 固定，最小化这一目标等价于最大似然（见[公式 (1.1.2)](../../chapters/01-deep-generative-modeling/01-what-is-deep-generative-modeling.md#eq-mle)）：

$$
\min_{{\boldsymbol{\phi}}}
    \mathcal D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{{\boldsymbol{\phi}}})
    \quad
    \Longleftrightarrow
    \quad
    \max_{{\boldsymbol{\phi}}}
    {\mathbb{E}}_{{\mathbf{x}}\sim p_{\mathrm{data}}}
    \big[
        \log p_{{\boldsymbol{\phi}}}({\mathbf{x}})
    \big].
$$

这就是许多基于似然的模型背后的分布层面视角。自回归模型和归一化流优化精确似然；基于似然的 VAE 优化变分下界；基于能量的模型通过正相和负相优化似然；扩散模型则可通过其变分和去噪目标，与似然或 score 匹配替代目标联系起来。

从 Wasserstein 梯度流视角看，正向 KL散度的一阶变分为

$$
\frac{\delta}{\delta p}
    \mathcal D_{\mathrm{KL}}(p_{\mathrm{data}}\|p)
    =
    -
    \frac{p_{\mathrm{data}}}{p}.
$$

在 $p=p_{{\boldsymbol{\phi}}_\tau}$ 处求值，得到速度

$$
{\mathbf{w}}_\tau^{\mathrm{FKL}}({\mathbf{x}})
    =
    \nabla_{{\mathbf{x}}}
    \left(
        \frac{p_{\mathrm{data}}({\mathbf{x}})}
             {p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}})}
    \right).
$$

这一表达式涉及密度比 $p_{\mathrm{data}}/p_{{\boldsymbol{\phi}}_\tau}$，对于隐式生成器而言通常难以估计。实践中，在似然、变分界或合适的似然梯度近似可用时，正向 KL散度或最大似然训练最为直接。Score 匹配通过匹配 score 函数提供另一条分布学习路线，并不直接实现正向 KL散度的 Wasserstein 流。

正向 KL散度训练的主要优势是覆盖数据，并且在似然或变分替代目标可用时具有良好的统计基础。其局限在于，通常需要可计算密度、变分界、score 匹配替代目标或昂贵的负采样，因而较难直接适用于任意隐式生成器。

**反向 KL散度训练：模型侧的 score 不匹配动力学。** 模型侧的、即反向 KL散度为

$$
\mathcal{D}_{\mathrm{RKL}}(p_{{\boldsymbol{\phi}}},p_{\mathrm{data}})
    :=
    \mathcal D_{\mathrm{KL}}(p_{{\boldsymbol{\phi}}}\|p_{\mathrm{data}})
    =
    \int
    p_{{\boldsymbol{\phi}}}({\mathbf{x}})
    \log
    \frac{p_{{\boldsymbol{\phi}}}({\mathbf{x}})}
         {p_{\mathrm{data}}({\mathbf{x}})}
    \mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

其一阶变分为

$$
\frac{\delta}{\delta p}
    \mathcal D_{\mathrm{KL}}(p\|p_{\mathrm{data}})
    =
    \log p-\log p_{\mathrm{data}}+1.
$$

在 $p=p_{{\boldsymbol{\phi}}_\tau}$ 处求值，得到 Wasserstein 速度

$$
{\mathbf{w}}_\tau^{\mathrm{RKL}}({\mathbf{x}})
    :=
    \nabla_{{\mathbf{x}}}\log p_{\mathrm{data}}({\mathbf{x}})
    -
    \nabla_{{\mathbf{x}}}\log p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}}).
$$

因此，反向 KL散度训练诱导一个由 score 不匹配产生的力：

$$
\text{target score}
    -
    \text{model score}.
$$

目标 score 将生成粒子吸引到数据分布的高密度区域，模型 score 则将粒子从当前模型分布已经过于集中的区域排斥出去。

在扩散蒸馏与分布匹配中（见[第 10.2 节](../../chapters/10-distillation/02-distribution-based-distillation.md#sec-vsd)），这一思想通常应用于噪声水平 $s$。令 $p_s$ 表示教师分布或加噪数据分布，$p_{{\boldsymbol{\phi}}_\tau,s}$ 表示当前生成器样本加噪后的分布。反向 KL散度目标

$$
{\mathbb{E}}_s
    \left[
        w(s)
        \mathcal D_{\mathrm{KL}}(p_{{\boldsymbol{\phi}}_\tau,s}\|p_s)
    \right]
$$

诱导 score 不匹配速度

$$
{\mathbf{w}}_{\tau,s}^{\mathrm{RKL}}({\mathbf{x}})
    :=
    \nabla_{{\mathbf{x}}}\log p_s({\mathbf{x}})
    -
    \nabla_{{\mathbf{x}}}\log p_{{\boldsymbol{\phi}}_\tau,s}({\mathbf{x}}).
$$

在参数层面，对于生成的含噪样本 $\hat{\mathbf{x}}_s\sim p_{{\boldsymbol{\phi}}_\tau,s}$，得到以下形式的梯度：

$$
\nabla_{{\boldsymbol{\phi}}}
    \mathcal D_{\mathrm{KL}}(p_{{\boldsymbol{\phi}},s}\|p_s)
    \big|_{{\boldsymbol{\phi}}={\boldsymbol{\phi}}_\tau}
    =
    {\mathbb{E}}_{\hat{\mathbf{x}}_s\sim p_{{\boldsymbol{\phi}}_\tau,s}}
    \left[
        \left(
            \nabla_{{\mathbf{x}}}\log p_{{\boldsymbol{\phi}}_\tau,s}(\hat{\mathbf{x}}_s)
            -
            \nabla_{{\mathbf{x}}}\log p_s(\hat{\mathbf{x}}_s)
        \right)^\top
        \partial_{{\boldsymbol{\phi}}}\hat{\mathbf{x}}_s
    \right].
$$

因此，score 差

$$
\nabla_{{\mathbf{x}}}\log p_s(\hat{\mathbf{x}}_s)
-
\nabla_{{\mathbf{x}}}\log p_{{\boldsymbol{\phi}}_\tau,s}(\hat{\mathbf{x}}_s)
$$

提供驱动参数梯度的分布层面下降方向。如上所述，单个生成样本的实际运动由共享神经网络参数化实现。这一 score 差梯度是 DMD 和 VSD 等基于反向 KL散度方法的基本分布层面机制：在生成样本上估计目标 score 与模型 score，再用它们之差更新生成器。

反向 KL散度训练的主要优势是：只要能估计相关 score，就天然适用于隐式生成器。它还具有模式寻求性，可提高单步或少步生成的视觉保真度。其局限在于可能遗漏模式，强烈依赖 score 估计质量，并且可能需要额外的模型 score 网络或强大的教师。

同样的吸引—排斥解释也出现在漂移式粒子更新中（[Weber, 2023](../../references.md#ref-weber2023the)；[Deng et al., 2026](../../references.md#ref-deng2026generative)）。按照 [Lai et al. (2026)](../../references.md#ref-lai2026unified) 的术语，DMD／VSD／SiD 提供这一 score 不匹配动力学的参数化、基于 score 的实例，而漂移式方法（[Weber, 2023](../../references.md#ref-weber2023the)；[Deng et al., 2026](../../references.md#ref-deng2026generative)）提供非参数、基于核的实例。

**结语。** 连续性方程为扩散动力学与分布层面训练提供共同语言。在扩散模型中，它描述中间分布 $p_t$ 如何沿给定噪声动力学或概率流动力学演化。在 Wasserstein 梯度流中，同一方程描述模型分布 $p_{{\boldsymbol{\phi}}_\tau}$ 如何在训练期间演化，此时生成粒子向数据分布 $p_{\mathrm{data}}$ 移动。

这一视角将连续性方程的作用扩展到扩散模型之外，使我们可以将生成建模理解为分布动力学的设计：不同方法选择不同差异度量、粒子速度和参数更新，使 $p_{{\boldsymbol{\phi}}}$ 趋近 $p_{\mathrm{data}}$。从这一视角看，扩散模型、基于似然的模型、分布匹配方法和基于 score 不匹配的方法，可以纳入共同的分布层面分类。

[^1]: 这一速度不应与扩散时间下的 PF-ODE 速度 ${\mathbf{v}}^*({\mathbf{x}},t)$ 混淆。变量 $t$ 或 $s$ 描述加噪／去噪时间，$\tau$ 则描述模型分布在训练时间中的演化。

[^2]: 这一理想的分布空间动力学不应与神经网络参数 ${\boldsymbol{\phi}}$ 上的普通梯度下降混淆：后者受有限维生成器族约束，通常只能近似实现所需的粒子运动。

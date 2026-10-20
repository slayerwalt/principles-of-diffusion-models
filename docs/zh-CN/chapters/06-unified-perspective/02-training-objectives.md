# 6.2 理解扩散模型训练损失的路线图 {#sec-elucidating}

本节系统考察扩散模型的训练损失。[第 6.2.1 节](02-training-objectives.md#subsec-four-predictions)将标准的三个目标扩展到更广泛的四种参数化，说明它们如何从不同建模视角产生。[第 6.2.2 节](02-training-objectives.md#subsec-summary-disentagle)进一步将这些结果提炼为一般框架，厘清扩散目标的结构，为[第 6.3 节](03-equivalent-parameterizations.md#sec-equivalent-parametrizations)的等价性结果奠定基础。

## 6.2.1 扩散模型的四种常见参数化 {#subsec-four-predictions}

除非另行说明，本节始终考虑前向扰动核

$$
p_t(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I} \right),
$$

其中 $\mathbf{x}_0 \sim p_{\mathrm{data}}$，定义见[式 (4.5.1)](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#eq-forward-kernel)。

令 $\omega: [0, T] \to \mathbb{R}_{>0}$ 表示正的时间权重函数。为清楚起见，并便于后续讨论，下面总结四种标准参数化（噪声 $\boldsymbol{\epsilon}_{\boldsymbol{\phi}}$、干净数据 ${\mathbf{x}}_{\boldsymbol{\phi}}$、score ${\mathbf{s}}_{\boldsymbol{\phi}}$ 与速度 ${\mathbf{v}}_{\boldsymbol{\phi}}$），以及各自的最优解 $\boldsymbol{\epsilon}^*$、${\mathbf{x}}^*$、${\mathbf{s}}^*$ 与 ${\mathbf{v}}^*$。

**变分视角。** 基于 DDPM 中的 KL散度（见[第 2.2.4 节](../02-variational-perspective/02-ddpm.md#subsec-ddpm-prediction)与[第 4.5.2 节](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#subsec-connection-vdm)），这一方法归结为预测产生 $\mathbf{x}_t$ 的噪声期望，或预测受到扰动后形成 $\mathbf{x}_t$ 的干净信号期望。

1.  **${\boldsymbol{\epsilon}}$ 预测（噪声预测）**（[Ho et al., 2020](../../references.md#ref-ho2020denoising)）：

    <span id="eq-noise-def"></span>

    $$
    \boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) \approx \mathbb{E}[\boldsymbol{\epsilon}|\mathbf{x}_t]=\boldsymbol{\epsilon}^*({\mathbf{x}}_t, t)
    \tag{6.2.1}
    $$

    其训练目标为

    $$
    \mathcal{L}_{\text{noise}}(\boldsymbol{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) - \boldsymbol{\epsilon} \right\|_2^2 \right].
    $$

    这里，${\boldsymbol{\epsilon}}^*$ 表示为得到给定 ${\mathbf{x}}_t$ 而注入的噪声的平均值。

2.  **${\mathbf{x}}$ 预测（干净数据预测）**（[Kingma et al., 2021](../../references.md#ref-kingma2021variational)；[Karras et al., 2022](../../references.md#ref-karras2022elucidating)；[Song et al., 2023](../../references.md#ref-song2023consistency)）：

    <span id="eq-clean-def"></span>

    $$
    \mathbf{x}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) \approx \mathbb{E}[\mathbf{x}_0|\mathbf{x}_t]={\mathbf{x}}^*({\mathbf{x}}_t, t)
    \tag{6.2.2}
    $$

    其训练目标为

    $$
    \mathcal{L}_{\text{clean}}(\boldsymbol{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \mathbf{x}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) - \mathbf{x}_0 \right\|_2^2 \right].
    $$

    这里，${\mathbf{x}}^*$ 表示给定含噪观测 ${\mathbf{x}}_t$ 时，所有可能干净信号猜测的平均值。

**基于 score 的视角。** 预测噪声水平 $t$ 下的 score 函数，其方向是将 $\mathbf{x}_t$ 去噪、移回所有可能产生它的干净样本的平均方向：

3.  **score 预测**（[Song and Ermon, 2019](../../references.md#ref-song2019generative)；[Song et al., 2020](../../references.md#ref-song2020score)）：

    <span id="eq-score-def"></span>

    $$
    \mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \mathbb{E}\left[\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0) |\mathbf{x}_t\right] = {\mathbf{s}}^*({\mathbf{x}}_t, t)
    \tag{6.2.3}
    $$

    其训练目标为

    $$
    \mathcal{L}_{\text{score}}(\boldsymbol{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0) \right\|_2^2 \right],
    $$

    其中条件 score 满足 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0) = -\frac{1}{\sigma_t} \boldsymbol{\epsilon}$。

**基于流的视角。** 预测数据演化经过 $\mathbf{x}_t$ 时的瞬时平均速度：

4.  **${\mathbf{v}}$ 预测（速度预测）**（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)；[Liu, 2022](../../references.md#ref-liu2022rectified)；[Salimans and Ho, 2021](../../references.md#ref-salimans2021progressive)；[Albergo et al., 2023](../../references.md#ref-albergo2023stochastic)）：

    <span id="eq-velocity-def"></span>

    $$
    \mathbf{v}_{\boldsymbol{\phi}}(\mathbf{x}_t, t) \approx \mathbb{E}\left[\left. \frac{\mathop{}\!\mathrm{d}  \mathbf{x}_t}{\mathop{}\!\mathrm{d}  t} \right| {\mathbf{x}_t}\right] ={\mathbf{v}}^*({\mathbf{x}}_t, t)
    \tag{6.2.4}
    $$

    其训练目标为

    $$
    \mathcal{L}_{\text{velocity}}(\boldsymbol{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \mathbf{v}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) - {\mathbf{v}}_t({\mathbf{x}}_t | {\mathbf{x}}_0,\boldsymbol{\epsilon}) \right\|_2^2 \right],
    $$

    其中条件速度为 ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{x}}_0,\boldsymbol{\epsilon}) = \alpha_t' \mathbf{x}_0 + \sigma_t' \boldsymbol{\epsilon}$。

    这里，${\mathbf{v}}^*$ 表示通过观测点 ${\mathbf{x}}_t$ 的平均速度向量。

基于[式 (6.1.1)](01-conditioning-tricks.md#eq-three-minimizer-oracle)的认识，四种预测类型最终都旨在近似一个条件期望，即给定观测 $\mathbf{x}_t$ 时，噪声、干净数据、score 或速度的平均值。

## 6.2.2 厘清扩散模型的训练目标 {#subsec-summary-disentagle}

如[第 6.2.1 节](02-training-objectives.md#subsec-four-predictions)所示，四种预测类型的目标函数，通常共享以下扩散模型训练模板：

<span id="eq-general-dm-loss-d"></span>

$$
\begin{aligned}
   \mathcal{L}(\boldsymbol{\phi}):=\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}} \underbrace{\bbox[2px,#d5dcf2]{\color{black}{\mathbb{E}_{p_{\text{time}}(t)}}}}_{\substack{\text{time} \\  \text{distribution}}}\Big[ \underbrace{\bbox[2px,#d6e5c9]{\color{black}{\omega(t)}}}_{\substack{\text{time} \\ \text{weighting}}}\underbrace{\left\lVert \bbox[2px,#e7cfea]{\color{black}{\mathrm{NN}_{\boldsymbol{\phi}}}}(\bbox[2px,#ffe7e4]{\color{black}{{\mathbf{x}}_t}}, t) - \bbox[2px,#e7cfea]{\color{black}{(A_t {\mathbf{x}}_0 + B_t \boldsymbol{\epsilon}) }}\right\rVert_2^2}_{\text{MSE part}}\Big].
\end{aligned}
\tag{6.2.5}
$$

为了提高训练效率、优化扩散模型学习流程，若干关键设计选择至关重要（[Karras et al., 2022](../../references.md#ref-karras2022elucidating)；[Lu and Song, 2024](../../references.md#ref-lu2024simplifying)）：

-   **(A)** $\bbox[2px,#ffe7e4]{\color{black}{{\mathbf{x}}_t}}$ 前向过程中的噪声调度，通过 $\alpha_t$ 与 $\sigma_t$ 指定；

-   **(B)** $\bbox[2px,#e7cfea]{\color{black}{\mathrm{NN}_{\boldsymbol{\phi}}}}$ 的预测类型及对应回归目标 $\bbox[2px,#e7cfea]{\color{black}{(A_t {\mathbf{x}}_0 + B_t \boldsymbol{\epsilon}) }}$；

-   **(C)** 时间分布 $\bbox[2px,#d5dcf2]{\color{black}{p_{\text{time}}}}$；

-   **(D)** 时间权重函数 $\bbox[2px,#d6e5c9]{\color{black}{\omega(\cdot)}} \colon [0,T] \to \mathbb{R}_{\geq 0}$。

下面详细说明这四个组成部分，作为后续各节讨论的路线图。

**(A) 噪声调度 $\alpha_t$ 与 $\sigma_t$。** 使用者可以灵活选择适合应用的调度，常见示例见[表 5.2](../05-flow-based-perspective/03-probability-paths-and-velocities.md#tb-interpolant-instances)。正如[式 (6.3.3)](03-equivalent-parameterizations.md#eq-fm-general-affine-formula)与[式 (6.3.5)](03-equivalent-parameterizations.md#eq-any-to-trig)将说明的，所有形如 $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$ 的仿射流在数学上等价。具体而言，任何此类插值，都可通过适当的时间重参数化和空间缩放，转化为标准线性调度（$\alpha_t = 1 - t$、$\sigma_t = t$），或三角函数调度（$\alpha_t = \cos t$、$\sigma_t = \sin t$）。

**(B) 参数化 $\mathrm{NN}_{\boldsymbol{\phi}}$ 与训练目标 $A_t {\mathbf{x}}_0 + B_t \boldsymbol{\epsilon}$。**

<span id="tb-a-t-b-t-parametrizations"></span>

**表 6.1：** 不同参数化之间关系的总结。四种参数化在数学上等价，可通过简单的代数变换相互转换。

<div style="overflow-x: auto;" markdown="1">
<table markdown="1">
<tr markdown="1">
<th markdown="block">

回归目标 =

</th>
<th colspan="2" markdown="block">

$A_t{\mathbf{x}}_0 + B_t \boldsymbol{\epsilon}$

</th>
</tr>
<tr markdown="1">
<td markdown="block">



</td>
<td markdown="block">

$A_t$

</td>
<td markdown="block">

$B_t$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

干净数据

</td>
<td markdown="block">

$1$

</td>
<td markdown="block">

$0$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

噪声

</td>
<td markdown="block">

$0$

</td>
<td markdown="block">

$1$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

条件 score

</td>
<td markdown="block">

$0$

</td>
<td markdown="block">

-$\frac{1}{\sigma_t}$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

条件速度

</td>
<td markdown="block">

$\alpha_t'$

</td>
<td markdown="block">

$\sigma_t'$

</td>
</tr>
</table>
</div>

扩散训练中有两种经常联系在一起的选择：*网络直接预测什么量*，以及*训练损失比较什么量*。干净数据、噪声、score 与速度都可以用于其中任一种用途。

**预测与损失匹配。** [式 (6.2.5)](02-training-objectives.md#eq-general-dm-loss-d)采用的常见选择，是直接以网络输出的同一种预测类型训练网络。每种预测类型都有对应的条件训练目标，形式为

$$
\text{Regression Target}
=
A_t{\mathbf{x}}_0+B_t{\boldsymbol{\epsilon}},
$$

其中 $A_t$ 与 $B_t$ 取决于预测类型及前向调度 $(\alpha_t,\sigma_t)$。[表 6.1](02-training-objectives.md#tb-a-t-b-t-parametrizations)总结了这些目标。

例如，${\mathbf{x}}$ 预测网络与干净目标 ${\mathbf{x}}_0$ 比较，而 ${\mathbf{v}}$ 预测网络与条件速度比较：

$$
{\mathbf{v}}_t
=
\alpha_t'{\mathbf{x}}_0+\sigma_t'{\boldsymbol{\epsilon}}.
$$

如后面的[式 (6.3.1)](03-equivalent-parameterizations.md#eq-predictions-equivalence)所示，四种预测类型编码相同的 oracle 信息，可以相互转换。

**预测一个量，在另一个量上计算损失。** 网络输出与损失使用的量也可以分别选择。先利用[式 (6.3.1)](03-equivalent-parameterizations.md#eq-predictions-equivalence)中的关系转换网络预测，再在转换后的预测类型中，与相应目标计算损失。

例如，在标准调度

$$
{\mathbf{x}}_t=(1-t){\mathbf{x}}_0+t{\boldsymbol{\epsilon}},
$$

下，假设网络直接预测干净数据 ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)$。将这一预测转换为速度，得到

$$
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
=
\frac{{\mathbf{x}}_t-{\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)}{t},
$$

而条件速度目标满足

$$
{\mathbf{v}}_t
=
\frac{{\mathbf{x}}_t-{\mathbf{x}}_0}{t}.
$$

因此，

$$
\left\|
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)-{\mathbf{v}}_t
\right\|_2^2
=
\frac{1}{t^2}
\left\|
{\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)-{\mathbf{x}}_0
\right\|_2^2.
$$

所以，当网络输出固定为干净数据预测时，计算速度损失，严格等价于为干净数据预测损失再施加一个随时间变化的权重。一般仿射调度与其他预测类型的对应关系，将在[式 (6.3.6)](03-equivalent-parameterizations.md#eq-s-e-x-v-equi)中总结。

改变网络直接预测的量，还会产生额外影响。虽然 oracle 预测可通过代数转换，但此时有限容量神经网络要表示的是另一个函数，输出尺度与参数梯度也不同。因此，网络预测类型的选择可能与模型架构和优化相互影响，而损失的选择还控制不同噪声水平上预测误差的相对权重。

为清楚起见，除非另行说明，[式 (6.2.5)](02-training-objectives.md#eq-general-dm-loss-d)使用预测与损失匹配的情形。

**(C) 时间分布 $p_{\text{time}}(t)$。** 接下来考察训练期间如何采样时间 $t$，从而采样噪声水平。为方便起见，定义固定时间回归损失

$$
\mathcal{L}_t(\boldsymbol{\phi})
:=
\mathbb{E}_{{\mathbf{x}}_0,{\boldsymbol{\epsilon}}}
\left[
\left\|
\mathrm{NN}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
-
\bigl(A_t{\mathbf{x}}_0+B_t{\boldsymbol{\epsilon}}\bigr)
\right\|_2^2
\right],
\qquad
{\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}.
$$

则[式 (6.2.5)](02-training-objectives.md#eq-general-dm-loss-d)可简洁写为

$$
\mathcal{L}(\boldsymbol{\phi})
=
\mathbb{E}_{t\sim p_{\text{time}}}
\left[
\omega(t)\mathcal{L}_t(\boldsymbol{\phi})
\right].
$$

因此，若不作修正，更频繁地采样某些时刻，也会赋予它们更大的目标权重。在精确期望损失层面，只有乘积 $p_{\text{time}}(t)\omega(t)$ 起作用。

非均匀采样的另一用途是*重要性采样*。假设上述目标固定，但改从提议分布 $q_{\text{time}}(t)$ 抽取时间，则

$$
\mathcal{L}(\boldsymbol{\phi})
=
\mathbb{E}_{t\sim q_{\text{time}}}
\left[
\frac{p_{\text{time}}(t)}{q_{\text{time}}(t)}
\omega(t)\mathcal{L}_t(\boldsymbol{\phi})
\right],
$$

这里要求 $q_{\text{time}}(t)>0$ 在所有满足 $p_{\text{time}}(t)\omega(t)>0$ 的位置成立。有了这一修正，改变 $q_{\text{time}}$ 不会改变目标，只会改变其 Monte Carlo 估计器，例如估计器的方差。

因此，关键区别是：

<div style="overflow-x: auto;" markdown="1">
<table markdown="1">
<tr markdown="1">
<td markdown="block">

改变目标中的 $p_{\text{time}}$

</td>
<td markdown="block">

$\Longrightarrow$

</td>
<td markdown="block">

不同权重，不同目标

</td>
</tr>
<tr markdown="1">
<td markdown="block">

改变 $q_{\text{time}}$ $+$ 重要性修正

</td>
<td markdown="block">

$\Longrightarrow$

</td>
<td markdown="block">

相同目标，不同估计器

</td>
</tr>
</table>
</div>

因此，我们在概念上区分时间采样与损失加权。均匀时间采样使这一区别不太显眼，因为其密度只贡献一个常数因子；非均匀采样则必须说明，采样分布是在定义目标，还是仅作为重要性采样的提议分布。

扩散模型训练中两种用法都存在。ScoreFlow 显式使用重要性采样，降低固定似然加权目标的方差（[Song et al., 2021](../../references.md#ref-song2021maximum)）。VDM 及后续分析同样表明，当目标在噪声水平坐标中固定时，噪声调度可以充当重要性采样分布（[Kingma et al., 2021](../../references.md#ref-kingma2021variational)；[Kingma and Gao, 2023](../../references.md#ref-kingma2023understanding)）。EDM（[Karras et al., 2022](../../references.md#ref-karras2022elucidating)）则同时指定噪声水平上的非均匀训练分布和单独的损失权重，二者共同决定有效训练权重。

**(D) 时间权重函数 $\omega(t)$。** 对固定 $p_{\text{time}}$，权重函数 $\omega(t)$ 决定各时刻对训练目标的相对贡献。结合前文可知，时间上的有效权重为 $p_{\text{time}}(t)\omega(t)$。

一个简单基准是均匀时间采样并取 $\omega(t)\equiv1$，标准的简化 ${\boldsymbol{\epsilon}}$ 预测目标与 FM/RF 目标采用这种方式（[Ho et al., 2020](../../references.md#ref-ho2020denoising)；[Lipman et al., 2022](../../references.md#ref-lipman2022flow)；[Liu, 2022](../../references.md#ref-liu2022rectified)）。其他权重强调不同噪声范围，也可能具有似然或变分解释。例如，对于 score 匹配目标，在适当正则性条件下，似然权重 $\omega(t)=g^2(t)$ 给出负对数似然的上界（[Song et al., 2021](../../references.md#ref-song2021maximum)）。更一般地，依赖 SNR 的权重是重新分配各噪声水平训练重点的一种常见方式（[Kingma et al., 2021](../../references.md#ref-kingma2021variational)；[Kingma and Gao, 2023](../../references.md#ref-kingma2023understanding)）。

**单调权重与带数据增强的 ELBO 视角。** 描述扰动程度的一种有用方式是*对数信噪比*（log-SNR）

$$
\Lambda_t
:=
\log\frac{\alpha_t^2}{\sigma_t^2}.
$$

我们使用 $\Lambda$ 表示完整的 log-SNR，以区别于后面[第 9 章](../09-numerical-solvers/index.md#ch-numerical-solvers)使用的半对数信噪比坐标 $\lambda_t=\log(\alpha_t/\sigma_t)=\tfrac12\Lambda_t$。由于前向过程噪声增大时 $\Lambda_t$ 减小，将损失表示为共同的预测参数化后，令 $w(\Lambda)$ 表示分配给每个 log-SNR 水平的有效权重。

[Kingma and Gao (2023)](../../references.md#ref-kingma2023understanding) 的一个有用结果是：若 $w(\Lambda)$ 关于 $\Lambda$ 单调不增（等价地，其权重不随着前向过程噪声增加而减小），则加权扩散目标可改写为 Gaussian 扰动数据上的平均负 ELBO，至多相差与模型参数无关的项[^1]。

因此，单调加权具有具体的概率解释：它不仅强调某些噪声水平，还可理解为对经过不同强度 Gaussian 扰动的数据版本的变分训练取平均。

**结论。** 对于这里的讨论，主要启示是：扩散训练中许多看似不同的设计选择，可以通过它们如何改变目标中各噪声水平的有效权重来理解。这一权重既影响实践中的优化景观，也在单调加权等特殊情形下影响损失的变分解释。

[^1]: 为理解单调性的作用，将从噪声水平 $t$ 开始的剩余变分代价定义为

    $$
    L(t;{\mathbf{x}}_0)
    :=
    \mathcal{D}_{\mathrm{KL}}
    \left(
    p({\mathbf{x}}_{[t,T]}|{\mathbf{x}}_0)
    \,\Vert\,
    p_{\boldsymbol{\phi}}({\mathbf{x}}_{[t,T]})
    \right),
    $$

    其中 ${\mathbf{x}}_{[t,T]}:=\{{\mathbf{x}}_s\}_{s\in[t,T]}$ 表示剩余的连续扩散路径。这是[式 (2.2.13)](../02-variational-perspective/02-ddpm.md#eq-ddpm-elbo)中负 ELBO 的连续时间对应形式，从扰动水平 $t$ 开始；更准确地说，它等于扰动样本 ${\mathbf{x}}_t$ 的期望负 ELBO，至多相差与 $\boldsymbol{\phi}$ 无关的项。

    对于 ${\boldsymbol{\epsilon}}$ 预测，[Kingma and Gao (2023)](../../references.md#ref-kingma2023understanding) 说明

    $$
    \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}L(t;{\mathbf{x}}_0)
    =
    \frac{1}{2}\frac{\mathop{}\!\mathrm{d}\Lambda_t}{\mathop{}\!\mathrm{d} t}
    {\mathbb{E}}_{{\boldsymbol{\epsilon}}}
    \left[
    \left\|
    {\boldsymbol{\epsilon}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)-{\boldsymbol{\epsilon}}
    \right\|_2^2
    \right].
    $$

    因此，加权扩散损失为

    $$
    -\int_0^T
    w(\Lambda_t)
    \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}L(t;{\mathbf{x}}_0)\mathop{}\!\mathrm{d} t
    =
    w(\Lambda_0)L(0;{\mathbf{x}}_0)
    +
    \int_0^T
    \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}w(\Lambda_t)
    L(t;{\mathbf{x}}_0)\mathop{}\!\mathrm{d} t
    +
    C,
    $$

    其中 $C$ 与 $\boldsymbol{\phi}$ 无关。若 $w(\Lambda)$ 关于 $\Lambda$ 单调不增，则 $\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}w(\Lambda_t)\geq0$，因为 $\Lambda_t$ 随 $t$ 减小。因此，归一化之后，上述各项构成对不同 Gaussian 扰动水平的 ELBO 的非负加权，从而得到带 Gaussian 数据增强的 ELBO 解释。

# 5.4 （选读）标准仿射流的性质 {#sec-optimal-canonical}

给定两个端点分布 $p_0=p_{\mathrm{src}}$ 与 $p_1=p_{\mathrm{tgt}}$，在流匹配（FM）（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)）与 Rectified Flow（RF）（[Liu et al., 2022](../../references.md#ref-liu2022flow)）中，定义条件路径的一种自然且广泛使用的选择是线性插值

$$
a_t = 1 - t, \qquad b_t = t,
$$

得到插值形式

$$
\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t \mathbf{x}_1,
\qquad \mathbf{x}_0 \sim p_{\mathrm{src}},\ \mathbf{x}_1 \sim p_{\mathrm{tgt}}.
$$

在这一选择下，训练目标简化为

$$
\mathbb{E}_{t \sim \mathcal{U}[0,1]}
\mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1}
\left[ \bigl\|\mathbf{v}_{\boldsymbol{\phi}}(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\bigr\|_2^2 \right].
$$

这种线性流具有若干有吸引力的性质。特别地，它支持称为 *`Reflow`* 的迭代改进方案，在保留边缘分布的同时，逐渐拉直分布之间的路径。

## 5.4.1 矫正流：从含噪猜测到有结构的配对 {#subsec-rectify}

**为什么改变耦合？** 在独立耦合

$$
\pi({\mathbf{x}}_0,{\mathbf{x}}_1)
=
p_{\mathrm{src}}({\mathbf{x}}_0)
p_{\mathrm{tgt}}({\mathbf{x}}_1),
$$

下，每条条件插值 $(1-t){\mathbf{x}}_0+t{\mathbf{x}}_1$ 都是直线。然而，正如[第 5.3.5 节](03-probability-paths-and-velocities.md#conditional-vs-marginal-straightness)所强调的，对应边缘速度会对许多不相关端点对的方向取平均，因此诱导的边缘 ODE 轨迹未必是直线。这样的弯曲边缘动力学，可能更难仅用少量数值步骤准确积分。

矫正通过改变端点耦合来处理这一问题。学得的 ODE 将每个源样本输运到模型生成的端点，产生相关耦合，随后用它定义新的条件插值。

**通过相关耦合矫正流。** 我们使用预训练扩散模型 ${\mathbf{v}}_{{\boldsymbol{\phi}}^\times}(\cdot,t)$ 作为 PF-ODE 的漂移，*确定性地*输运每个源点，无须依赖任意配对。从 ${\mathbf{z}}(0)={\mathbf{z}}_0 \sim p_{\text{src}}$ 出发，积分

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{z}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{v}}_{{\boldsymbol{\phi}}^\times}({\mathbf{z}}(t),t), \qquad t \in [0,1],
$$

得到位于预训练模型所学数据空间附近的 $\hat{\mathbf{z}}_1 := {\mathbf{z}}(1)$。所得样本对 $({\mathbf{z}}_0,\hat{\mathbf{z}}_1)$ 构成*相关耦合*：它遵循模型引导的有结构路径，而非任意插值。这一思想自然扩展到形如 ${\mathbf{x}}_t = a_t {\mathbf{x}}_0 + b_t {\mathbf{x}}_1$ 的仿射参考路径，其中 ${\mathbf{x}}_0 \sim p_{\text{src}}$ 与 ${\mathbf{x}}_1 \sim p_{\text{tgt}}$。

<section class="book-algorithm" id="alg-rectify" aria-labelledby="alg-rectify-title" markdown="1">

<div class="book-algorithm__title" id="alg-rectify-title" markdown="span">
算法 2：Rectify 操作
</div>

<div class="book-algorithm__meta" markdown="1">

**输入：** 参考路径 $\{{\mathbf{x}}_t\}_{t\in[0,1]}$（例如 ${\mathbf{x}}_t=a_t{\mathbf{x}}_0+b_t{\mathbf{x}}_1$）

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="算法 2 步骤" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line book-algorithm__line--block" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

**预训练扩散模型。** 在所选路径上，通过最小化下式拟合 ${\mathbf{v}}_{{\boldsymbol{\phi}}^\times}$：

$$
{\boldsymbol{\phi}}^\times \in \arg\min_{{\boldsymbol{\phi}}}
\mathbb{E}_{t,{\mathbf{x}}_0,{\mathbf{x}}_1} \left[\Big\|{\mathbf{v}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)-\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}\Big\|_2^2\right].
$$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--block" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

**矫正。** 采样 ${\mathbf{z}}_0 \sim p_{\text{src}}$，并积分

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{z}}(t)}{\mathop{}\!\mathrm{d} t}={\mathbf{v}}_{{\boldsymbol{\phi}}^\times}({\mathbf{z}}(t),t),\qquad {\mathbf{z}}(0)={\mathbf{z}}_0,\quad t\in[0,1],
$$

得到 $\hat{\mathbf{z}}_1={\mathbf{z}}(1)$ 及轨迹 $\{{\mathbf{z}}(t)\}_{t\in[0,1]}$。

</div>
</div>

</div>
</div>

<div class="book-algorithm__meta" markdown="1">

**输出：** 相关且协调的样本对 $({\mathbf{z}}_0,\hat{\mathbf{z}}_1)$，或完整轨迹。

</div>

</section>

**为什么有效：保持边缘分布的结构。** 令 ${\boldsymbol{\Phi}}_{0\to t}$ 表示上述 ODE 生成的流映射，该 ODE 由预训练扩散模型 ${\mathbf{v}}_{{\boldsymbol{\phi}}^\times}$ 定义；则 ${\mathbf{z}}(t)={\boldsymbol{\Phi}}_{0\to t}({\mathbf{z}}_0)$ 且 $\hat{\mathbf{z}}_1={\boldsymbol{\Phi}}_{0\to 1}({\mathbf{z}}_0)$。`Rectify` 过程将每个源点与其流的终点配对，得到确定性联合分布

$$
\pi_{\text{Rectify}}({\mathbf{z}}_0,{\mathbf{z}}_1)
= p_{\text{src}}({\mathbf{z}}_0) \delta \big({\mathbf{z}}_1-{\boldsymbol{\Phi}}_{0\to 1}({\mathbf{z}}_0)\big).
$$

立即得到两个结果：

-   **源边缘分布得到保留：** $\displaystyle \int \pi_{\text{Rectify}}({\mathbf{z}}_0,{\mathbf{z}}_1) \mathop{}\!\mathrm{d} {\mathbf{z}}_1 = p_{\text{src}}({\mathbf{z}}_0)$。

-   **沿流推前：** $\displaystyle ({\boldsymbol{\Phi}}_{0\to t})_{\#}p_{\text{src}}=\mathrm{Law}({\mathbf{z}}(t))$，即时刻 $t$ 的分布是 $p_{\text{src}}$ 经 ${\boldsymbol{\Phi}}_{0\to t}$ 推前的结果。

若 ${\mathbf{v}}_{{\boldsymbol{\phi}}^\times}$ 匹配给定参考路径 ${\mathbf{x}}_t$ 的 oracle 漂移，则所有中间边缘分布都一致：

$$
\mathrm{Law}({\mathbf{z}}(t))=\mathrm{Law}({\mathbf{x}}_t),\quad \text{for all }t\in[0,1],
\quad\text{and}\quad
({\boldsymbol{\Phi}}_{0\to 1})_{\#}p_{\mathrm{src}}=p_{\mathrm{tgt}}.
$$

**小结。** 矫正将独立端点耦合替换为学得的流所诱导的耦合。反复执行 `Reflow`，可以得到越来越易于用少量数值步骤积分的边缘输运。

对于标准路径，反复应用 `Rectify`（即 *`Reflow`*）会进一步拉直轨迹，同时不增加输运代价，使训练也更容易。

## 5.4.2 `Reflow`：迭代拉直流 {#subsec-reflow-def}

**为什么使用 `Reflow`？** 独立配对通常在 $p_{\text{src}}$ 与 $p_{\text{tgt}}$ 之间诱导不规则、曲折的 ODE 轨迹，增大模拟中的离散化误差与方差。这自然引出以下问题：

<span id="ques-straight"></span>

**问题 5.4.1**

能否学习一种耦合，使其诱导的输运路径在两个分布之间更接近直线，同时仍保留正确的边缘分布？

这启发了 *`Reflow`*：反复应用 `Rectify` 更新耦合，使后续的流更易积分。

**核心思想：通过 `Rectify` 递归拉直。**

从乘积耦合 $\pi^{(0)}:=p_{\text{src}}({\mathbf{x}}_0)p_{\text{tgt}}({\mathbf{x}}_1)$ 上的标准插值开始，

$$
{\mathbf{x}}_t=(1-t){\mathbf{x}}_0+t{\mathbf{x}}_1.
$$

应用 `Rectify`，用学得的 ODE 所诱导的相关配对 $({\mathbf{z}}_0,\hat{\mathbf{z}}_1)$ 替换独立配对。反复执行这一过程，即 *`Reflow`*，通常会产生更直的轨迹，使其更容易通过少量数值步骤准确积分。[第 5.4.3 节](04-canonical-affine-flow.md#subsec-reflow-properties)将讨论精确的拉直保证。

**`Reflow` 过程。** 每次迭代执行两步：

-   **重新拟合流：** 用当前耦合的样本训练新速度场：

    <span id="eq-reflow-loss"></span>

    $$
    \begin{aligned}
    \boldsymbol{\phi}_{k+1} &= \arg\min_{\boldsymbol{\phi}} \mathcal{L}\left(\boldsymbol{\phi}\Big\vert \pi^{(k)}\right), \quad \text{where}  \\ \mathcal{L}\left(\boldsymbol{\phi}\Big\vert \pi^{(k)}\right) &:= \mathbb{E}_{t, ({\mathbf{z}}_0^{(k)}, \hat{\mathbf{z}}_1^{(k)})\sim \pi^{(k)}} \left[\left\|{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{z}}_t, t) - (\hat{\mathbf{z}}_1^{(k)} - {\mathbf{z}}_0^{(k)})\right\|^2\right]
    \end{aligned}
    \tag{5.4.1}
    $$

    其中 ${\mathbf{z}}_t = (1-t){\mathbf{z}}_0^{(k)} + t\hat{\mathbf{z}}_1^{(k)}$。

-   **生成新耦合：** 从新的源样本 ${\mathbf{z}}_0^{(k+1)} \sim p_{\text{src}}$ 出发，求解学得的 ODE：

    $$
    \hat{\mathbf{z}}_1^{(k+1)} \gets {\mathbf{z}}_0^{(k+1)} + \int_0^1 {\mathbf{v}}_{\boldsymbol{\phi}_{k+1}}({\mathbf{z}}(t), t) \mathrm{d}t,
    $$

    并定义更新后的耦合：

    $$
    \pi^{(k+1)}({\mathbf{z}}_0, {\mathbf{z}}_1) := p_{\text{src}}({\mathbf{z}}_0) \delta\left({\mathbf{z}}_1 - \hat{\mathbf{z}}_1^{(k+1)}\right).
    $$

换言之，`Reflow` 可以视为反复应用 `Rectify` 算子，产生逐步改进的耦合序列：

<span id="eq-rectify-op"></span>

$$
\begin{aligned}
\pi^{(k+1)} = \texttt{Rectify} \left(\pi^{(k)}\right)
\end{aligned}
\tag{5.4.2}
$$

从而使流与耦合共同演化，产生逐步更稳定的输运路径。

## 5.4.3 `Reflow` 的性质 {#subsec-reflow-properties}

`Reflow` 的实用性来自两个关键理论性质：降低输运代价，以及拉直轨迹。

**I. `Reflow` 不会增加输运代价。** 令 $c({\mathbf{y}})$ 为凸代价函数（例如 $\|{\mathbf{y}}\|_2^p$，其中 $p \geq 1$）。每一步 `Rectify` 形成新耦合 $({\mathbf{z}}_0, \hat{\mathbf{z}}_1)$，其代价不劣于原耦合：

<span id="rec-transport-cost"></span>

**命题 5.4.1：`Rectify` 可以降低输运代价**

假设使用理想速度场 ${\mathbf{v}}^* = {\mathbf{v}}_{\boldsymbol{\phi}^\times}$，则有：

$$
\mathbb{E}\left[c\left(\hat{\mathbf{z}}_1 - {\mathbf{z}}_0\right)\right] \leq \mathbb{E}\left[c\left({\mathbf{x}}_1 - {\mathbf{x}}_0\right)\right].
$$

**命题证明。**

由 Jensen 不等式得到。完整推导见 [Liu et al. (2022)](../../references.md#ref-liu2022flow)。

递归应用这一结果可知，`Reflow` 过程不会增加输运代价。

**II. `Reflow` 拉直路径。** 递归应用 `Reflow` 会使轨迹越来越接近直线，但直线性度量并不保证每次迭代都单调改善。为衡量这一性质，将路径 $\mathbf{Y} = \{{\mathbf{y}}_t\}_{t\in[0,1]}$ 的*直线性泛函*定义为

$$
\mathcal{S}(\mathbf{Y}) := \int_0^1 \mathbb{E}\left[\left\|{\mathbf{y}}_1 - {\mathbf{y}}_0 - \frac{\mathrm{d}{\mathbf{y}}_t}{\mathrm{d}t} \right\|_2^2 \right]  \mathrm{d}t.
$$

若 $\mathcal{S}(\mathbf{Y}) = 0$，则 $\mathbf{Y}$ 严格为直线。

<span id="reflow-straight"></span>

**命题 5.4.2：`Reflow` 拉直随机路径**

对于矫正后的路径 ${\mathbf{Z}}^{(k)}$，有：

$$
\min_{k \in \{0,\dots,K\}} \mathcal{S}({\mathbf{Z}}^{(k)}) \leq \frac{\mathbb{E}\left[\|{\mathbf{x}}_1 - {\mathbf{x}}_0\|^2\right]}{K}.
$$

**命题证明。**

见 [Liu et al. (2022)](../../references.md#ref-liu2022flow) 的定理 3.7。

<span id="fig-ill-reflow"></span>

![Reflow 逐渐拉直输运路径](../../../en/assets/chapter-05/ill-reflow.png)

**图 5.11：** **`Reflow` 示意图。** 通过 `Rectify` 过程，路径逐渐变得更直。

来源：改编自 [Liu et al. (2022)](../../references.md#ref-liu2022flow)。

采用线性插值核的 FM 或 RF 形式，结合 `Reflow` 过程，提供了更简单的训练目标，以及改进随机耦合的实用方法。理论细节请参阅（[Liu et al., 2022](../../references.md#ref-liu2022flow)；[Liu, 2022](../../references.md#ref-liu2022rectified)）。

**III. 与最优输运的联系。** 最后指出，直线耦合在最优输运（OT）的意义下不一定最优。这涉及一些将在[第 7.2 节](../07-optimal-transport/02-transport-problem-setups.md#sec-ot-intro)介绍的术语；不熟悉 OT 的读者可先参阅该节。

二次代价最优输运的一个特征是粒子沿直线运动：位于 ${\mathbf{x}}_0$ 的粒子移动到 ${\mathbf{T}}({\mathbf{x}}_0)$，经过路径 ${\mathbf{x}}_t = (1 - t){\mathbf{x}}_0 + t {\mathbf{T}}({\mathbf{x}}_0)$，其中 ${\mathbf{T}}$ 是最优输运映射。然而，并非每个产生这种直线路径的映射 ${\mathbf{S}}$，即 ${\mathbf{x}}_t = (1 - t){\mathbf{x}}_0 + t {\mathbf{S}}({\mathbf{x}}_0)$，都是最优的。映射 ${\mathbf{S}}$ 只有最小化 Monge 代价 $\mathbb{E}[\|{\mathbf{x}}_0 - {\mathbf{S}}({\mathbf{x}}_0)\|^2]$ 时才产生最优流。因此，直线路径虽然必要，却不充分；最优性还取决于正确的端点映射 ${\mathbf{T}}$。

**示例：直线耦合未必最优**

令 $p_{\text{src}} = p_{\text{tgt}} = \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$。对于代价 $c({\mathbf{x}}, {\mathbf{y}}) = \|{\mathbf{x}} - {\mathbf{y}}\|^p$，其中 $p > 0$，$c$ 最优耦合是恒等耦合 $\pi^{*}$，其中 $\pi^{*}$ 为 $({\mathbf{x}}, {\mathbf{x}})$ 的分布，且 ${\mathbf{x}} \sim p_{\text{src}}$。

现在考虑耦合 $\pi_{{\mathbf{A}}}$，定义为 $({\mathbf{x}}, {\mathbf{A}} {\mathbf{x}})$ 的分布，其中 ${\mathbf{x}} \sim p_{\text{src}}$，${\mathbf{A}}$ 是旋转矩阵，满足 ${\mathbf{A}}^\top {\mathbf{A}} = {\mathbf{I}}$、$\det({\mathbf{A}}) = 1$、${\mathbf{A}} \neq {\mathbf{I}}$，且 $-1$ 不是其特征值。于是，$\pi_{{\mathbf{A}}}$ 是 $p_{\text{src}}$ 与 $p_{\text{tgt}}$ 的有效耦合，对应 ${\mathbf{x}}$ 与 ${\mathbf{A}} {\mathbf{x}}$ 之间的直线路径，但它不是 $c$ 最优的，对任意二次可微、严格凸且 Hessian 矩阵可逆的代价 $c$ 都如此。这种次优性源自旋转变换。如[式 (7.5.2)](../07-optimal-transport/05-probability-flow-odes-and-optimal-transport.md#eq-reflow-loss-gradient)所述，即使去除旋转，也未必得到最优耦合。

我们将在[第 7.5.2 节](../07-optimal-transport/05-probability-flow-odes-and-optimal-transport.md#subsec-ot-linear-reflow)继续探索与 OT 的联系。

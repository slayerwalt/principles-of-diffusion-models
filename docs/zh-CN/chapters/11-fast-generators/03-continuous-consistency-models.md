# 11.3 特殊流映射：连续时间 Consistency Model {#sec-conti-cm}

现在，我们离开一致性模型的离散时间设定，转而考虑连续时间视角。离散方法固定时间网格，仅在采样到的网格点上训练；连续表述则将流映射视为在所有时刻均有定义。这一转变消除了对任意离散化选择的依赖，使其更有依据地与底层动力学对齐。它还有助于减少离散积分自然产生的近似误差，并确保在全局施加一致性，而不只在选定步骤上施加。

## 11.3.1 连续时间 Consistency Model {#subsec-conti-ct}

为引出连续时间表述，我们先回顾[公式 (11.2.3)](02-discrete-consistency-models.md#eq-cm-diff)，它描述了可对时间求导时的一致性条件。使用链式法则，可得

<span id="eq-conti-ct-motivation"></span>

$$
\begin{aligned}
        &\frac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s} {\mathbf{f}}^*({\mathbf{x}}(s), s) = 0 \\
    \Longleftrightarrow\,\,
    &\left(\nabla_{{\mathbf{x}}} {\mathbf{f}}^*\right)({\mathbf{x}}(s), s) \cdot
    \underbrace{\frac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s} {\mathbf{x}}(s)}_{\text{ODE velocity}} +
    \left(\frac{\partial }{\partial s} {\mathbf{f}}^*\right)({\mathbf{x}}(s), s) = 0,
\end{aligned}
\tag{11.3.1}
$$

其中轨迹 ${\mathbf{x}}(s)$ 遵循 PF-ODE

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s}{\mathbf{x}}(s) =  {\mathbf{v}}^*({\mathbf{x}}(s), s).
\end{aligned}
$$

这一关系表明，一致性函数 ${\mathbf{f}}^*$ 沿 ODE 的任意解轨迹保持不变。在实践中，速度场 ${\mathbf{v}}^*$ 可以通过预训练扩散模型估计（若有可用模型），也可以用直接的单点近似，例如 $\alpha_s' {\mathbf{x}}_0 + \sigma_s' {\boldsymbol{\epsilon}}$，如[第 11.2 节](02-discrete-consistency-models.md#sec-discrete-cm)所述。

[公式 (11.3.1)](03-continuous-consistency-models.md#eq-conti-ct-motivation)为沿 PF-ODE 轨迹的一致性提供了自然的连续时间刻画。一种可能的方法是直接最小化残差来施加这一微分条件，其方式类似于物理信息神经网络（PINN）（[Raissi, 2018](../../references.md#ref-raissi2018deep)；[Boffi et al., 2025](../../references.md#ref-boffi2024flow)）：

$$
\begin{aligned}
\min_{\boldsymbol{\theta}}
\mathbb{E}_{s,{\mathbf{x}}_s\sim p_s}
\left[
\left\|
\partial_s{\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s)
+
\big(\partial_{{\mathbf{x}}}{\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s)\big)
{\mathbf{v}}^*({\mathbf{x}}_s,s)
\right\|_2^2
\right].
\end{aligned}
$$

然而，在实践中，[Song et al. (2023)](../../references.md#ref-song2023consistency) 和 [Lu and Song (2024)](../../references.md#ref-lu2024simplifying) 使用基于教师的一致性目标，而不直接优化这一残差。因此，一个问题是：当 $\Delta s \to 0$ 时，离散一致性损失是否存在有意义的无穷小对应形式？以下命题表明，在停止梯度目标与在线网络重合的比较点，缩放后的离散时间梯度收敛于一个连续时间目标的梯度：

<span id="eq-cm-discrete"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CM}}^{\Delta s}({\boldsymbol{\theta}},{\boldsymbol{\theta}}^-)
:= {\mathbb{E}} \left[\omega(s) \big\|
{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s) -
{\mathbf{f}}_{{\boldsymbol{\theta}}^-} \big({\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s), s-\Delta s\big)
\big\|_2^2\right].
\end{aligned}
\tag{11.3.2}
$$

对于[公式 (11.2.4)](02-discrete-consistency-models.md#eq-cd-discrete-approx-n)和[公式 (11.2.6)](02-discrete-consistency-models.md#eq-ct-discrete-approx-n)所用的均匀划分，令

$$
\Delta s_N:=\frac{T}{N-1}.
$$

则有 $\Delta s_N\to0$，当 $N\to\infty$ 时成立。因此，[公式 (11.3.2)](03-continuous-consistency-models.md#eq-cm-discrete)中的无穷小极限对应于不断细化这些离散目标。

我们将这一关键思想概括为以下命题。

<span id="continuous-time-ct"></span>

**命题 11.3.1：连续时间一致性训练**

在 $\boldsymbol{\theta}=\boldsymbol{\theta}^-$ 处，以下收敛结果成立：

$$
\begin{aligned}
\left.\lim_{\Delta s\to 0}\frac{1}{\Delta s}\nabla_{{\boldsymbol{\theta}}}\mathcal{L}_{\mathrm{CM}}^{\Delta s}(\boldsymbol{\theta}, \boldsymbol{\theta}^-)\right|_{\boldsymbol{\theta}=\boldsymbol{\theta}^-}
    =
    \left.\nabla_{\boldsymbol{\theta}} \mathcal{L}^\infty_{\mathrm{CM}}(\boldsymbol{\theta}, \boldsymbol{\theta}^-)\right|_{\boldsymbol{\theta}=\boldsymbol{\theta}^-}.
\end{aligned}
$$

这里，

$$
\begin{aligned}
\mathcal{L}^\infty_{\mathrm{CM}}(\boldsymbol{\theta},\boldsymbol{\theta}^-)
    :=
    \mathbb{E}_{s,{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[
    2\omega(s)\,
    {\mathbf{f}}_{\boldsymbol{\theta}}^\top({\mathbf{x}}_s, s)\cdot
    \frac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{\boldsymbol{\theta}^-}({\mathbf{x}}_s, s)
    \right],
\end{aligned}
$$

而全微分恒等式为

<span id="eq-total-diff"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
    =
    \partial_s {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
    + \big(\partial_{{\mathbf{x}}}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)\big){\mathbf{v}}^*({\mathbf{x}}_s,s).
\end{aligned}
\tag{11.3.3}
$$

**命题的证明。**

在 $({\mathbf{x}}_s, s)$ 附近对停止梯度目标作一阶 Taylor 展开可知，损失 $\mathcal{L}_{\mathrm{CM}}^{\Delta s}$ 在忽略 $\mathcal{O}(\Delta s^2)$ 阶项后，行为类似于学生更新 $\nabla_{{\boldsymbol{\theta}}}{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s)$ 与切向变化 $\tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)$ 的内积。在 $\boldsymbol{\theta}=\boldsymbol{\theta}^-$ 处求值会消除零阶差异项，因此缩放后的梯度满足

$$
\left.\lim_{\Delta s\to 0}\frac{1}{\Delta s}\nabla_{{\boldsymbol{\theta}}}\mathcal{L}_{\mathrm{CM}}^{\Delta s}\right|_{\boldsymbol{\theta}=\boldsymbol{\theta}^-}
=
\left.\nabla_{{\boldsymbol{\theta}}}{\mathbb{E}}\left[
2\omega(s)\,
{\mathbf{f}}_{{\boldsymbol{\theta}}}^\top({\mathbf{x}}_s,s)\cdot
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
\right]\right|_{\boldsymbol{\theta}=\boldsymbol{\theta}^-},
$$

这就是所要证明的恒等式。证明细节留待[第 D.5 节](../../appendices/d-supplementary-proofs/05-fast-generators.md#app-sec-flow-map)。

上述结果写在梯度算子 $\nabla_{\boldsymbol{\theta}}$ 下，使涉及 $\boldsymbol{\theta}^-$ 的项消失，因为在停止梯度下，$\boldsymbol{\theta}^-$ 被视为常量。注意，$\tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)$ 表示沿 oracle 轨迹的全导数，而非简单的时间偏导数。

因此，连续时间一致性模型可使用以下伪目标的梯度进行训练（忽略因子 $2$）：

<span id="eq-conti-ct-loss"></span>

$$
\begin{aligned}
\mathcal{L}^\infty_{\mathrm{CM}}(\boldsymbol{\theta},\boldsymbol{\theta}^-) := \mathbb{E}_{s,{\mathbf{x}}_0, \boldsymbol{\epsilon}}\left[\omega(s){\mathbf{f}}_{\boldsymbol{\theta}}^\top({\mathbf{x}}_s, s) \cdot \frac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s} {\mathbf{f}}_{\boldsymbol{\theta}^-}({\mathbf{x}}_s, s) \right].
\end{aligned}
\tag{11.3.4}
$$

## 11.3.2 训练连续时间 Consistency Model {#training-continuous-time-consistency-model}

与[第 11.2.1 节](02-discrete-consistency-models.md#subsec-approx-cm)讨论的离散时间情形类似，我们现在说明如何在实践中近似[公式 (11.3.4)](03-continuous-consistency-models.md#eq-conti-ct-loss)中的切向项，其中涉及无法获得的 oracle 速度 ${\mathbf{v}}^*$：

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
    = \partial_s {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
    + \big(\partial_{{\mathbf{x}}}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)\big)\,{\mathbf{v}}^*({\mathbf{x}}_s,s).
\end{aligned}
$$

训练好连续时间 CM 后，采样流程与离散时间情形相同（[第 11.2.2 节](02-discrete-consistency-models.md#subsec-cm-sampling)）。

**连续时间 Consistency Distillation。** 若有满足 ${\mathbf{v}}_{\boldsymbol{\phi}^\times} \approx {\mathbf{v}}^*$ 的预训练扩散模型，则[公式 (11.3.3)](03-continuous-consistency-models.md#eq-total-diff)中的切向量 $\tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)$ 可用以下替代量近似：

<span id="eq-cd-continuous-time"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
     \approx
    \partial_s {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
    + \big(\partial_{{\mathbf{x}}}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)\big)\,
      {\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s).
\end{aligned}
\tag{11.3.5}
$$

将所得目标记为 $\mathcal{L}^\infty_{\mathrm{CM}}(\boldsymbol{\theta},\boldsymbol{\theta}^-;\boldsymbol{\phi}^\times)$。相应地，[命题 11.3.1](03-continuous-consistency-models.md#continuous-time-ct)可重述为

$$
\lim_{N\to\infty}
\frac{1}{\Delta s_N}\,\nabla_{\boldsymbol{\theta}}\,
\mathcal{L}_{\mathrm{CD}}^{N}(\boldsymbol{\theta},\boldsymbol{\theta}^-;\boldsymbol{\phi}^\times)
=
\nabla_{\boldsymbol{\theta}}\,
\mathcal{L}_{\mathrm{CD}}^{\infty}(\boldsymbol{\theta},\boldsymbol{\theta}^-;\boldsymbol{\phi}^\times).
$$

**连续时间 Consistency Training（从头训练）。** 另一方面，若没有预训练扩散模型，oracle 速度 ${\mathbf{v}}^*$ 可以用单点条件估计 $\alpha_s'{\mathbf{x}}_0 + \sigma_s'{\boldsymbol{\epsilon}}$ 近似。此时，[公式 (11.3.3)](03-continuous-consistency-models.md#eq-total-diff)中的切向量 $\tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)$ 被替换为

<span id="eq-ct-continuous-time"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
     \approx
    \partial_s {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
    + \big(\partial_{{\mathbf{x}}}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)\big)
      \left(\alpha_s'{\mathbf{x}}_0 + \sigma_s'{\boldsymbol{\epsilon}}\right).
\end{aligned}
\tag{11.3.6}
$$

将所得目标记为 $\mathcal{L}^\infty_{\mathrm{CT}}(\boldsymbol{\theta},\boldsymbol{\theta}^-)$，对应于从头训练的设定。相应地，[命题 11.3.1](03-continuous-consistency-models.md#continuous-time-ct)可重述为

$$
\lim_{N\to\infty}
\frac{1}{\Delta s_N}\,\nabla_{\boldsymbol{\theta}}
\mathcal{L}_{\mathrm{CT}}^{N}(\boldsymbol{\theta},\boldsymbol{\theta}^-)
=
\nabla_{\boldsymbol{\theta}}
\mathcal{L}_{\mathrm{CT}}^{\infty}(\boldsymbol{\theta},\boldsymbol{\theta}^-).
$$

至此，我们已介绍[表 11.1](02-discrete-consistency-models.md#tb-consistency-methods)列出的全部基本方法，用于学习一致性函数 ${\boldsymbol{\Psi}}_{s\to 0}$。为便于整体理解，[图 11.4](03-continuous-consistency-models.md#fig-cd-ct-loss-distill)概括了训练一致性函数的不同损失之间的关系。图中也标明了各方法是否依赖预训练扩散模型，并区分连续时间与离散时间目标。

<span id="fig-cd-ct-loss-distill"></span>

![连续时间一致性蒸馏和训练的损失](../../../en/assets/chapter-11/cd-ct-loss.png)

**图 11.4：** **在 $\ell_2$ 距离度量下，离散／连续时间 CD 与 CT 之间的关系图：$d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_2^2$。** 图中定理的标号沿用（[Song et al., 2023](../../references.md#ref-song2023consistency)）。凡定理涉及 CT，均假设 score 完美，即 ${\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}, t) \equiv \nabla_{\mathbf{x}} \log p_t({\mathbf{x}})$。$\mathcal{L}^\infty_{\text{CT}}$ 定义于[公式 (11.3.4)](03-continuous-consistency-models.md#eq-conti-ct-loss)，$\mathcal{L}^\infty_{\text{CD}}$ 则定义于[公式 (11.3.5)](03-continuous-consistency-models.md#eq-cd-continuous-time)。

来源：作者绘制。

图中链接：[定理 11.2.1](02-discrete-consistency-models.md#thm-ct-cm)。

不过，切向量 $\tfrac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{\boldsymbol{\theta}^-}$ 经常导致训练不稳定。在接下来的选读部分中，我们介绍 *Simplifying, Stabilizing and Scaling Continuous Time Consistency Models*（sCM）（[Lu and Song, 2024](../../references.md#ref-lu2024simplifying)）中用于缓解这些问题的技术。

## 11.3.3 （选读）连续时间 CT 的实践考虑 {#practical-considerations}

我们关注从头训练的情形，因为它能得到不依赖外部预训练扩散模型的独立生成模型。因此，我们重点讨论连续时间情形。

但在实践中，直接使用[公式 (11.3.4)](03-continuous-consistency-models.md#eq-conti-ct-loss)训练往往不稳定，因为 $\tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{\boldsymbol{\theta}^-}$ 项可能具有很大或无界的时间导数，导致优化过程出现梯度爆炸。为克服这一问题，通常需要合适的参数化和稳定化策略（[Geng et al., 2025](../../references.md#ref-geng2025consistency)；[Lu and Song, 2024](../../references.md#ref-lu2024simplifying)）。如[第 6.2.2 节](../06-unified-perspective/02-training-objectives.md#subsec-summary-disentagle)所概括，影响训练稳定性的主要因素包括*扩散过程*、*参数化选择*、*时间权重函数*和*时间采样分布*，在连续时间 CM 中同样应仔细设计并区分这些因素。

**扩散过程。** 对于标准扩散参数化 ${\mathbf{x}}_s = \alpha_s {\mathbf{x}}_0 + \sigma_s \boldsymbol{\epsilon}$，其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$，[Lu and Song (2024)](../../references.md#ref-lu2024simplifying) 改用三角函数调度。尽管这一调度在数学上与原始形式等价（如[公式 (6.3.4)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-trig-para)所示），它能带来更简洁的结构，并使训练目标中的各部分更好地分离，从而提升训练稳定性[^1]。此外，遵循[第 D.6.1 节](../../appendices/d-supplementary-proofs/06-edm.md#subsec-d-design-edm)中 EDM 的设计，他们纳入数据标准差 $\sigma_{\mathrm{d}}$，其对应的数据分布为 $p_{\mathrm{data}}$：

<span id="eq-trig-scm"></span>

$$
\begin{aligned}
{\mathbf{x}}_s := \cos(s) {\mathbf{x}}_0 + \sin(s) {\mathbf{z}}, \quad \text{where } {\mathbf{z}} \sim \mathcal{N}(\boldsymbol{0}, \sigma_{\mathrm{d}}^2 {\mathbf{I}}).
\end{aligned}
\tag{11.3.7}
$$

这一表述具有完全的一般性。对于任意具有形式 ${\mathbf{x}}_s = \alpha_s {\mathbf{x}}_0 + \sigma_s \boldsymbol{\epsilon}$、且满足 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$ 的扩散过程，都可以等价地写为：

$$
{\mathbf{x}}_s = \alpha_s {\mathbf{x}}_0 + \frac{\sigma_s}{\sigma_\mathrm{d}} \cdot (\sigma_{\mathrm{d}} \boldsymbol{\epsilon}),
$$

其中定义 ${\mathbf{z}} := \sigma_{\mathrm{d}} \boldsymbol{\epsilon}$、$\alpha'_s := \alpha_s$ 和 $\sigma'_s := \frac{\sigma_s}{\sigma_{\mathrm{d}}}$。随后，利用[公式 (6.3.5)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-any-to-trig)所述的归一化，可将变换后的系数对 $(\alpha'_s, \sigma'_s)$ 映射为三角函数形式 $(\cos(s), \sin(s))$。

**参数化。** 考虑[第 D.6.1 节](../../appendices/d-supplementary-proofs/06-edm.md#subsec-d-design-edm)中 EDM 的类似原则，[Lu and Song (2024)](../../references.md#ref-lu2024simplifying) 提出以下神经网络参数化，与[公式 (D.6.1)](../../appendices/d-supplementary-proofs/06-edm.md#eq-d-parametrization)类似：

$$
\begin{aligned}
{\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}, s) := c_{\mathrm{skip}}(s){\mathbf{x}} + c_{\mathrm{out}}(s){\mathbf{F}}_{\boldsymbol{\theta}}\left(c_{\mathrm{in}}(s){\mathbf{x}}, c_{\mathrm{noise}}(s)\right).
\end{aligned}
$$

这里，$c_{\mathrm{skip}}(s)$、$c_{\mathrm{out}}(s)$ 和 $c_{\mathrm{in}}(s)$ 可用[第 D.6.1 节](../../appendices/d-supplementary-proofs/06-edm.md#subsec-d-design-edm)中相同的准则推导（详细推导见 [Lu and Song (2024)](../../references.md#ref-lu2024simplifying) 的附录 B），其表达式为

$$
c_{\mathrm{skip}}(s) = \cos(s), \quad c_{\mathrm{out}}(s) = -\sigma_{\mathrm{d}} \sin(s), \quad c_{\mathrm{in}}(s) \equiv \frac{1}{\sigma_{\mathrm{d}}}.
$$

同时采用默认选择 $c_{\mathrm{noise}}(s) = s$，其中 $\partial_s c_{\mathrm{noise}}(s)$ 有界，以确保训练稳定性，相关讨论见[公式 (11.3.10)](03-continuous-consistency-models.md#eq-ds-decomposition)附近。由此得到三角函数调度下的参数化：

<span id="eq-trig-nn"></span>

$$
\begin{aligned}
{\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}, s) = \cos(s){\mathbf{x}} - \sin(s)\sigma_{\mathrm{d}}{\mathbf{F}}_{\boldsymbol{\theta}}\left(\frac{{\mathbf{x}}}{\sigma_{\mathrm{d}}}, c_{\mathrm{noise}}(s)\right).
\end{aligned}
\tag{11.3.8}
$$

注意，这一参数化还确保神经网络始终满足边界条件 ${\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}, 0) \equiv {\mathbf{x}}$，对所有 ${\mathbf{x}}$ 均成立，这是一致性函数的必要性质。

**稳定切向训练的技术。** 在三角函数调度及[公式 (11.3.8)](03-continuous-consistency-models.md#eq-trig-nn)所述的网络参数化下，[公式 (11.3.4)](03-continuous-consistency-models.md#eq-conti-ct-loss)中损失的梯度变为

<span id="eq-conti-ct-loss-trig"></span>

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}}\mathcal{L}^\infty_{\text{CT}}(\boldsymbol{\theta},\boldsymbol{\theta}^-)  =\nabla_{\boldsymbol{\theta}} \mathbb{E}_{s, {\mathbf{x}}_0, \boldsymbol{\epsilon}} \left[ -\omega(s) \sigma_{\mathrm{d}} \sin(s) {\mathbf{F}}_{\boldsymbol{\theta}}^\top\left( \frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, s \right) \cdot \frac{\mathop{}\!\mathrm{d} {\mathbf{f}}_{\boldsymbol{\theta}^-}}{\mathop{}\!\mathrm{d} s} ({\mathbf{x}}_s, s) \right].
\end{aligned}
\tag{11.3.9}
$$

理论上，使用[公式 (11.3.9)](03-continuous-consistency-models.md#eq-conti-ct-loss-trig)中的梯度更新，可能已足以学习一致性函数。然而，[Lu and Song (2024)](../../references.md#ref-lu2024simplifying) 在实验中观察到，实际训练会因切向函数的行为而变得不稳定，该函数为

$$
\begin{aligned}
&\underbrace{\frac{\mathop{}\!\mathrm{d} {\mathbf{f}}_{\boldsymbol{\theta}^-}({\mathbf{x}}_s, s)}{\mathop{}\!\mathrm{d} s}}_{\textbf{A.}}
= \\&-\cos(s) \left( \sigma_{\mathrm{d}} {\mathbf{F}}_{\boldsymbol{\theta}^-}\left( \frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, s \right) - \frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_s}{\mathop{}\!\mathrm{d} s} \right) - \sin(s) \left( {\mathbf{x}}_s + \sigma_{\mathrm{d}} \frac{\mathop{}\!\mathrm{d} {\mathbf{F}}_{\boldsymbol{\theta}^-}}{\mathop{}\!\mathrm{d} s}\left( \frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, c_{\mathrm{noise}}(s) \right) \right).
\end{aligned}
$$

具体而言，在以下项中观察到了不稳定：

$$
\begin{aligned}
\underbrace{\sin(s) \frac{\mathop{}\!\mathrm{d} {\mathbf{F}}_{\boldsymbol{\theta}^-}}{\mathop{}\!\mathrm{d} s}\left( \frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, c_{\mathrm{noise}}(s) \right)}_{\textbf{B.}}
= \sin(s) \nabla_{{\mathbf{x}}_s} {\mathbf{F}}_{\boldsymbol{\theta}^-} \frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_s}{\mathop{}\!\mathrm{d} s}
+ \sin(s) \partial_s {\mathbf{F}}_{\boldsymbol{\theta}^-}.
\end{aligned}
$$

更准确地说，不稳定来自分量

<span id="eq-ds-decomposition"></span>

$$
\begin{aligned}
\sin(s) \partial_s {\mathbf{F}}_{\boldsymbol{\theta}^-} = \sin(s) \underbrace{\frac{\partial c_{\mathrm{noise}}(s)}{\partial s} \cdot \frac{\partial \text{emb}(c_{\mathrm{noise}})}{\partial c_{\mathrm{noise}}}}_{\textbf{C.}} \cdot \frac{\partial {\mathbf{F}}_{\boldsymbol{\theta}^-}}{\partial \text{emb}(c_{\mathrm{noise}})}.
\end{aligned}
\tag{11.3.10}
$$

这里沿用 DM 和 CM 文献中的常见做法，对时间变量应用位置嵌入或 Fourier 嵌入，记为 $\text{emb}(\cdot)$，时间变量记为 $c_{\mathrm{noise}}(s)$：

$$
s \mapsto c_{\mathrm{noise}}(s) \mapsto \text{emb}(c_{\mathrm{noise}}(s)) \mapsto {\mathbf{F}}_{\boldsymbol{\theta}^-}\left( \frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, \text{emb}(c_{\mathrm{noise}}(s)) \right).
$$

因此，还引入了一些经验技术来缓解不稳定：

-   **A. 切向量归一化。** 将 $\frac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{\theta^-}$ 替换为 $\frac{\frac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s} {\mathbf{f}}_{\theta^-}}{\left\| \frac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s} {\mathbf{f}}_{\theta^-} \right\|_2 + c}$，显式归一化切向函数，其中 $c>0$ 是按经验设置的常数。另一种方式是将切向量裁剪到 $[-1, 1]$ 内，也能有效限制其方差。

-   **B. 切向量预热。** 由于 $\sin(s)({\mathbf{x}}_s + \sigma_{\mathrm{d}} \frac{\mathop{}\!\mathrm{d} }{\mathop{}\!\mathrm{d} s}{\mathbf{F}}_{\boldsymbol{\theta}^-})$ 项可能引发不稳定，可以选用一项技术：将系数 $\sin(s)$ 替换为 $r \cdot \sin(s)$，其中 $r$ 在训练初期的若干次迭代中从 0 线性增至 1。

-   **C. 时间嵌入。** 根据[公式 (11.3.10)](03-continuous-consistency-models.md#eq-ds-decomposition)中的导数链，[Lu and Song (2024)](../../references.md#ref-lu2024simplifying) 选择较小的幅度参数，以控制导数 $\frac{\partial \text{emb}(c_{\mathrm{noise}})}{\partial c_{\mathrm{noise}}}$。出于类似原因，选择 $c_{\mathrm{noise}}(s) = s$，其中 $\partial_s c_{\mathrm{noise}}(s)=1$ 是有界常数。

除此之外，通常还需要调整网络结构以改进归一化（提高稳定性），并采用基于 JVP 的方式高效计算 $\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{\theta^-}$，但这些内容超出了本书范围。

**时间权重函数。** 手工设计时间权重函数 $\omega(s)$ 可能导致次优表现。为解决这一问题，参照 EDM-2（[Karras et al., 2024](../../references.md#ref-karras2024analyzing)）的类似方法，[Lu and Song (2024)](../../references.md#ref-lu2024simplifying) 学习自适应权重函数 $\omega_{\boldsymbol{\varphi}}(s)$，平衡不同时间 $s$ 的训练损失方差（期望达到的结果见[公式 (11.3.11)](03-continuous-consistency-models.md#eq-effect-adaptive-wgt)）。

进一步而言，可以观察到[公式 (11.3.9)](03-continuous-consistency-models.md#eq-conti-ct-loss-trig)中的目标函数具有形式

$$
\mathbb{E}_{s, {\mathbf{x}}_0, \boldsymbol{\epsilon}} \left[{\mathbf{F}}_{\boldsymbol{\theta}}^\top {\mathbf{y}} \right],\quad\text{with}\quad {\mathbf{y}}=-\omega(s) \sigma_{\mathrm{d}} \sin(s)\frac{\mathop{}\!\mathrm{d} {\mathbf{f}}_{\boldsymbol{\theta}^-}}{\mathop{}\!\mathrm{d} s}.
$$

由于 ${\mathbf{y}}$ 是与 $\boldsymbol{\theta}$ 无关的向量，[公式 (11.3.9)](03-continuous-consistency-models.md#eq-conti-ct-loss-trig)等价于

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} \mathbb{E}_{s, {\mathbf{x}}_0, \boldsymbol{\epsilon}} \left[{\mathbf{F}}_{\boldsymbol{\theta}}^\top {\mathbf{y}} \right]
    = \frac{1}{2} \nabla_{\boldsymbol{\theta}} \mathbb{E}_{s, {\mathbf{x}}_0, \boldsymbol{\epsilon}} \left\| {\mathbf{F}}_{\boldsymbol{\theta}} - {\mathbf{F}}_{\boldsymbol{\theta}^-} + {\mathbf{y}} \right\|_2^2.
\end{aligned}
$$

基于这一观察，[Lu and Song (2024)](../../references.md#ref-lu2024simplifying) 提出额外训练一个自适应权重网络 $\omega_{\boldsymbol{\varphi}}(s)$ 来估计损失范数，表述为以下最小化问题：

$$
\begin{aligned}
\min_{\boldsymbol{\varphi}} \mathbb{E}_{s, {\mathbf{x}}_0, \boldsymbol{\epsilon}} \left[ \frac{e^{\omega_{\boldsymbol{\varphi}}(s)}}{D} \left\| {\mathbf{F}}_{\boldsymbol{\theta}} - {\mathbf{F}}_{\boldsymbol{\theta}^-} + {\mathbf{y}} \right\|_2^2 - \omega_{\boldsymbol{\varphi}}(s) \right].
\end{aligned}
$$

为理解自适应权重的作用，注意最优解 $\omega^*(s)$（通过将上述目标对 $\omega_{\boldsymbol{\varphi}}$ 求偏导得到）满足

<span id="eq-effect-adaptive-wgt"></span>

$$
\begin{aligned}
\mathbb{E}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}
    \left[
    \frac{e^{\omega^*(s)}}{D}
    \left\|
    {\mathbf{F}}_{\boldsymbol{\theta}}
    - {\mathbf{F}}_{\boldsymbol{\theta}^-}
    + {\mathbf{y}}
    \right\|_2^2
    \right]
    = 1,
    \qquad \text{for almost every } s.
\end{aligned}
\tag{11.3.11}
$$

也就是说，对于不受约束的逐点权重函数，缩放后损失的期望在每个时间 $s$ 分别归一化。因此，自适应权重有效减小了不同时间步之间训练损失的方差，使训练更均衡、更稳定。

**时间采样分布。** [Lu and Song (2024)](../../references.md#ref-lu2024simplifying) 选择从对数正态提议分布（[Karras et al., 2022](../../references.md#ref-karras2022elucidating)）采样 $\tan(s)$，即

<span id="eq-s-dist"></span>

$$
\begin{aligned}
\log\big(\sigma_{\mathrm{d}} \tan(s)\big) \sim \mathcal{N}(\cdot;P_{\text{mean}}, P_{\text{std}}^2).
\end{aligned}
\tag{11.3.12}
$$

这里，$P_{\text{mean}}$ 和 $P_{\text{std}}$ 是两个超参数。遵循 [Lu and Song (2024)](../../references.md#ref-lu2024simplifying)，可使用先验权重

$$
\omega(s)=\frac{1}{\sigma_{\mathrm{d}}\tan(s)},
$$

从而

$$
-\omega(s)\sigma_{\mathrm{d}}\sin(s)
\frac{\mathop{}\!\mathrm{d}{\mathbf{f}}_{\boldsymbol{\theta}^-}}{\mathop{}\!\mathrm{d} s}
=
-\cos(s)\frac{\mathop{}\!\mathrm{d}{\mathbf{f}}_{\boldsymbol{\theta}^-}}{\mathop{}\!\mathrm{d} s}.
$$

**训练目标总结。** 综合以上讨论，最终训练损失表示为：

$$
\begin{aligned}
&\mathcal{L}_{\text{sCM}}(\boldsymbol{\theta}, \boldsymbol{\varphi})
    := \\&\mathbb{E}_{s, {\mathbf{x}}_0, \boldsymbol{\epsilon}} \left[
    \frac{e^{\omega_{\boldsymbol{\varphi}}(s)}}{D}
    \left\|
    {\mathbf{F}}_{\boldsymbol{\theta}}\left(\frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, s\right)
    - {\mathbf{F}}_{\boldsymbol{\theta}^-}\left(\frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, s\right)
    - \cos(s) \frac{\mathop{}\!\mathrm{d} {\mathbf{f}}_{\boldsymbol{\theta}^-}}{\mathop{}\!\mathrm{d} s}\left({\mathbf{x}}_s, s\right)
    \right\|_2^2
    - \omega_{\boldsymbol{\varphi}}(s)
    \right].
\end{aligned}
$$

这里，$s$ 按[公式 (11.3.12)](03-continuous-consistency-models.md#eq-s-dist)采样，${\mathbf{x}}_s$ 通过[公式 (11.3.7)](03-continuous-consistency-models.md#eq-trig-scm)计算。用这一损失训练的模型称为 *sCM*，其训练流程见[算法 10](03-continuous-consistency-models.md#alg-scm)。

<section class="book-algorithm" id="alg-scm" aria-labelledby="alg-scm-title" markdown="1">

<div class="book-algorithm__title" id="alg-scm-title" markdown="span">
算法 10：连续时间 Consistency Models（sCM）的训练
</div>

<div class="book-algorithm__meta" markdown="1">

**输入：** 数据集 $\mathcal{D}$，其标准差为 $\sigma_{\mathrm{d}}$；预训练 DM $\mathbf{F}_{\text{pretrain}}$，参数为 $\boldsymbol{\theta}_{\text{pretrain}}$；模型 $\mathbf{F}_{\boldsymbol{\theta}}$；权重 $\omega_{\boldsymbol{\varphi}}$；学习率 $\eta$；提议分布 $(P_{\text{mean}}, P_{\text{std}})$；常数 $c$；预热迭代数 $H$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="Algorithm 10 steps" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

**初始化：** $\boldsymbol{\theta} \gets \boldsymbol{\theta}_{\text{pretrain}}$，Iters $\gets 0$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

**repeat**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{x}}_0 \sim \mathcal{D}$, $\mathbf{z} \sim \mathcal{N}(\boldsymbol{0}, \sigma_{\mathrm{d}}^2 \mathbf{I})$, $\tau \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$, $s \gets \arctan\left(\frac{e^\tau}{\sigma_{\mathrm{d}}}\right)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{x}}_s \gets \cos(s) {\mathbf{x}}_0 + \sin(s) \mathbf{z}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

**if** 一致性训练 **then**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

$\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_s}{\mathop{}\!\mathrm{d} s} \gets \cos(s) \mathbf{z} - \sin(s) {\mathbf{x}}_0$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">7</div>
<div class="book-algorithm__statement" markdown="1">

**else**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">8</div>
<div class="book-algorithm__statement" markdown="1">

$\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_s}{\mathop{}\!\mathrm{d} s} \gets \sigma_{\mathrm{d}} \mathbf{F}_{\text{pretrain}}\left(\frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, s\right)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">9</div>
<div class="book-algorithm__statement" markdown="1">

**end if**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">10</div>
<div class="book-algorithm__statement" markdown="1">

$r \gets \min\left(1, \frac{\text{Iters}}{H}\right)$  <span class="book-algorithm__comment">▷ *切向量预热*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">11</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{w}} \gets - \cos^2(s) (\sigma_{\mathrm{d}} \mathbf{F}_{\boldsymbol{\theta}}^{-} - \frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_s}{\mathop{}\!\mathrm{d} s}) - r \cos(s) \sin(s) \left({\mathbf{x}}_s + \sigma_{\mathrm{d}} \frac{\mathop{}\!\mathrm{d}\mathbf{F}_{\boldsymbol{\theta}^-}}{\mathop{}\!\mathrm{d} s}\right)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">12</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{w}} \gets \frac{{\mathbf{w}}}{\Vert {\mathbf{w}}\Vert  + c}$  <span class="book-algorithm__comment">▷ *切向量归一化*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">13</div>
<div class="book-algorithm__statement" markdown="1">

$\mathcal{L}_{\text{sCM}}(\boldsymbol{\theta}, \boldsymbol{\varphi}) \gets\frac{e^{\omega_{\boldsymbol{\varphi}}(s)}}{D} \left\lVert \mathbf{F}_{\boldsymbol{\theta}}\left( \frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, s\right) - \mathbf{F}_{\boldsymbol{\theta}^{-}}\left( \frac{{\mathbf{x}}_s}{\sigma_{\mathrm{d}}}, s\right) - {\mathbf{w}} \right\rVert^2_2 - \omega_{\boldsymbol{\varphi}}(s)$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">▷ *自适应权重*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">14</div>
<div class="book-algorithm__statement" markdown="1">

$(\boldsymbol{\theta}, \boldsymbol{\varphi}) \gets (\boldsymbol{\theta}, \boldsymbol{\varphi}) - \eta \nabla_{\boldsymbol{\theta}, \boldsymbol{\varphi}} \mathcal{L}_{\text{sCM}}(\boldsymbol{\theta}, \boldsymbol{\varphi})$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">15</div>
<div class="book-algorithm__statement" markdown="1">

Iters $\gets$ Iters + 1

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">16</div>
<div class="book-algorithm__statement" markdown="1">

**until** 收敛

</div>
</div>

</div>
</div>

</section>

[^1]: 直观地说，三角函数及其导数都有界，这有助于防止 $\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s} {\mathbf{f}}_{\boldsymbol{\theta}^-}$ 等项的尺度爆炸。后文将详细讨论。

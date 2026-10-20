# 11.2 特殊流映射：离散时间 Consistency Model {#sec-discrete-cm}

**流映射的一条重要原则：半群性质。**

<span id="fig-semigroup"></span>

![流映射的复合与沿同一轨迹的直接转移一致](../../../en/assets/chapter-11/semigroup.png)

**图 11.3：** **流映射半群性质示意图。** 这一性质说明，先从 $s$ 转移到 $u$，再从 $u$ 转移到 $t$，等价于直接从 $s$ 转移到 $t$。

来源：作者绘制。

Consistency Models（在[第 11.2 节](02-discrete-consistency-models.md#sec-discrete-cm)和[第 11.3 节](03-continuous-consistency-models.md#sec-conti-cm)中介绍）及其推广 Consistency Trajectory Model（[第 11.4 节](04-consistency-trajectory-models.md#sec-ctm)）利用流映射的一种关键数学结构定义回归目标。这一结构就是基本的*半群性质*：

<span id="eq-semi-group"></span>

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{u\to t}\circ {\boldsymbol{\Psi}}_{s\to u}={\boldsymbol{\Psi}}_{s\to t},
       \,\,\,
       {\boldsymbol{\Psi}}_{s\to s}={\mathbf{I}}, \quad \text{for all }\, s, u, t \in [0,T].
\end{aligned}
\tag{11.2.1}
$$

直观地说，如果先将状态从 $s$ 演化到 $u$（通过 ${\boldsymbol{\Psi}}_{s\rightarrow u}$），再从 $u$ 演化到 $t$（通过 ${\boldsymbol{\Psi}}_{u\rightarrow t}$），最终会到达与直接从 $s$ 演化到 $t$ 完全相同的位置。这正是 ODE 求解的基本原则[^1]：一旦指定流的起点，其未来演化就完全确定，并沿着唯一、明确的路径进行。无论用一个长步沿这条路径推进，还是将其分成较小的区间，都会沿同一轨迹到达相同的最终状态。

为进一步直观理解半群性质，考虑 PF-ODE 的解轨迹 $\{{\mathbf{x}}(s)\}_{s \in [0,T]}$：

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(\tau)}{\mathop{}\!\mathrm{d} \tau} = {\mathbf{v}}^*({\mathbf{x}}(\tau), \tau),
\end{aligned}
$$

固定初始条件为 ${\mathbf{x}}(T)$，对应时刻 $T$，并逆时间求解。如果将终止时刻固定为 $t=0$，相应的流映射可更简洁地写为

$$
{\mathbf{f}}^*(\cdot, s) := {\boldsymbol{\Psi}}_{s\to 0}(\cdot),
$$

这称为*一致性函数*。按照构造，在 $t=0$ 的情形下，该函数直接从[公式 (11.2.1)](02-discrete-consistency-models.md#eq-semi-group)的半群恒等式继承若干基本性质：

1.  **全局一致性：** 轨迹上的每一点都映射到相同的干净终点，

    $$
    {\mathbf{f}}^*({\mathbf{x}}(s), s) = {\mathbf{x}}(0), \quad \text{for all }\, s \in [0,T].
    $$

    这是因为

    $$
    \begin{aligned}
    {\mathbf{f}}^*({\mathbf{x}}(s),s)
    &= {\boldsymbol{\Psi}}_{s\to 0}\big({\boldsymbol{\Psi}}_{0\to s}({\mathbf{x}}(0))\big)
    = \big({\boldsymbol{\Psi}}_{s\to 0}\circ {\boldsymbol{\Psi}}_{0\to s}\big)({\mathbf{x}}(0))
    \\&= {\boldsymbol{\Psi}}_{0\to 0}({\mathbf{x}}(0))
    = {\mathbf{x}}(0).
    \end{aligned}
    $$

2.  **自一致性：** 同一轨迹上的任意两点都必须给出相同的输出，

    <span id="eq-cm-self-consistency"></span>

    $$
    \begin{aligned}
    {\mathbf{f}}^*({\mathbf{x}}(s), s) = {\mathbf{f}}^*({\mathbf{x}}(u), u), \quad \text{for all }\, s, u \in [0,T].
    \end{aligned}
    \tag{11.2.2}
    $$

    这是对半群恒等式的直接重述：$\Psi_{s\rightarrow 0}\circ\Psi_{0\rightarrow s}=\Psi_{u\rightarrow 0}\circ\Psi_{0\rightarrow u}$。

3.  **局部一致性：** 沿轨迹计算时，一致性函数不随 $s$ 改变，

    <span id="eq-cm-diff"></span>

    $$
    \begin{aligned}
    \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}\,{\mathbf{f}}^*({\mathbf{x}}(s),s) = 0,
            \qquad {\mathbf{f}}^*({\mathbf{x}}(0),0) = {\mathbf{x}}(0).
    \end{aligned}
    \tag{11.2.3}
    $$

    这由全局一致性推出，因为全局一致性说明，沿轨迹的 ${\mathbf{f}}^*({\mathbf{x}}(s), s)$ 不随 $s$ 改变。

这三条性质相互等价。它们都说明：沿任意解轨迹 $s\mapsto {\mathbf{x}}(s)$，流向时间原点的映射／一致性映射 ${\mathbf{f}}^*({\mathbf{x}}(s),s)={\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}(s))$ 都得到相同的终点 ${\mathbf{x}}(0)$，与起始时刻无关。

**Consistency Models 的目标。** CM 旨在训练神经网络 ${\mathbf{f}}_{\boldsymbol{\theta}} \colon \mathbb{R}^D \times [0,T] \to \mathbb{R}^D$，近似特殊流映射 ${\boldsymbol{\Psi}}_{s\to 0}$，即一致性函数[^2]。其关键思想是在 PF-ODE 轨迹上施加一致性：位于同一条 ODE 轨迹上的状态，应映射到相同的时刻 $0$ 的终点（更准确地说，这对应[公式 (11.2.1)](02-discrete-consistency-models.md#eq-semi-group)中 $t=0$ 且 $u=s-\Delta s$ 的特殊情形）。

不过，实现这一目标有多种方式。具体选择取决于是否有可用的预训练扩散模型，以及训练采用离散时间还是连续时间形式。我们先在[表 11.1](02-discrete-consistency-models.md#tb-consistency-methods)中概括这些变体，并在[图 11.4](03-continuous-consistency-models.md#fig-cd-ct-loss-distill)中展示其目标。后续各节（[第 11.2 节](02-discrete-consistency-models.md#sec-discrete-cm)和[第 11.3 节](03-continuous-consistency-models.md#sec-conti-cm)）将逐步展开各方法的细节。

<span id="tb-consistency-methods"></span>

**表 11.1：Consistency Models 的训练目标**

<table markdown="1" style="display: table; overflow: visible;">
<tr markdown="1">
<th markdown="block">



</th>
<th markdown="block">

**蒸馏**

</th>
<th markdown="block">

**从头训练**

</th>
</tr>
<tr markdown="1">
<td markdown="block">

**离散时间**

</td>
<td markdown="block">

[公式 (11.2.4)](02-discrete-consistency-models.md#eq-cd-discrete-approx-n)

</td>
<td markdown="block">

[公式 (11.2.6)](02-discrete-consistency-models.md#eq-ct-discrete-approx-n)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**连续时间**

</td>
<td markdown="block">

[公式 (11.3.5)](03-continuous-consistency-models.md#eq-cd-continuous-time)

</td>
<td markdown="block">

[公式 (11.3.6)](03-continuous-consistency-models.md#eq-ct-continuous-time)

</td>
</tr>
</table>

## 11.2.1 学习一致性函数的离散时间近似 {#subsec-approx-cm}

原则上，可以通过最小化[公式 (11.1.0)](01-prologue.md#eq-oracle-cm)中的 oracle 损失来学习一致性函数：

$$
\mathcal{L}_{\mathrm{oracle}\text{-}\mathrm{CM}}({\boldsymbol{\theta}})
    := {\mathbb{E}}_{s} {\mathbb{E}}_{{\mathbf{x}}_s\sim p_s} \left[ w(s)  d \left({\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s), {\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)\right)\right].
$$

这一目标要求每个含噪样本 ${\mathbf{x}}_s$ 都映射回其干净终点 ${\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)$。

困难在于，实际中无法获得 oracle 映射 ${\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)$。为解决这一问题，[Song et al. (2023)](../../references.md#ref-song2023consistency) 利用了*半群性质*：任意含噪状态及其沿同一 PF-ODE 轨迹推进一步所得的状态，都必须映射到相同的干净终点。具体而言，用取自轨迹上略早一点的*停止梯度*目标替代 oracle 目标：

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)
    &= {\boldsymbol{\Psi}}_{s-\Delta s \to 0} \left({\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)\right) \\
    &\approx {\mathbf{f}}_{{\boldsymbol{\theta}}^-} \left({\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s),\,s-\Delta s\right),
    \qquad \Delta s > 0,
\end{aligned}
$$

其中 $\boldsymbol{\theta}^-$ 是停止梯度算子下的参数。进一步的困难在于，中间状态 ${\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)$ 同样没有闭式表达式，本身也必须近似。为此提出了两种实际训练方式：

**使用预训练扩散模型（Consistency Distillation）。** 假设已有预训练扩散模型。Consistency Distillation（CD）利用教师模型，仅模拟一个逆向 ODE 步骤，就近似中间状态 ${\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)$：

$$
{\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)  \approx  \mathtt{Solver}_{s \to s-\Delta s}({\mathbf{x}}_s).
$$

更具体地，预训练扩散模型提供 score 函数 ${\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s, s) \approx \nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s)$ 的估计。利用它，可以从 ${\mathbf{x}}_s$ 出发执行一次 DDIM 更新，近似 $s' = s-\Delta s$ 时的状态：

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)
    &\approx \frac{\alpha_{s'}}{\alpha_s}\,{\mathbf{x}}_s
    + \sigma_s^2 \left( \frac{\alpha_{s'}}{\alpha_s} - \frac{\sigma_{s'}}{\sigma_s} \right)\nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s) \\
    &\approx \frac{\alpha_{s'}}{\alpha_s}\,{\mathbf{x}}_s
    + \sigma_s^2 \left( \frac{\alpha_{s'}}{\alpha_s} - \frac{\sigma_{s'}}{\sigma_s} \right){\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s, s) \\
    &:= \tilde{{\mathbf{x}}}_{s'}^{\boldsymbol{\phi}^\times}.
\end{aligned}
$$

将这一构造与停止梯度目标结合，便得到 oracle 损失 $\mathcal{L}_{\mathrm{oracle}\text{-}\mathrm{CM}}({\boldsymbol{\theta}})$ 的实用*离散时间*替代。形式上，在划分 $0 = s_1 < s_2 < \cdots < s_N = T$ 上，CD 的训练目标为

<span id="eq-cd-discrete-approx-n"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{CD}}^N(\boldsymbol{\theta}, \boldsymbol{\theta}^-; \boldsymbol{\phi}^\times)
    := {\mathbb{E}}_{{\mathbf{x}}_0, {\boldsymbol{\epsilon}}, i} \Big[\, \omega(s_i)\,
    d\big({\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}_{s_{i+1}}, s_{i+1}),\,
          {\mathbf{f}}_{\boldsymbol{\theta}^-}(\tilde{{\mathbf{x}}}_{s_i}^{\boldsymbol{\phi}^\times}, s_i)\big)\Big].
\end{aligned}
\tag{11.2.4}
$$

这里，$\omega(\cdot)$ 是依赖时间的权重，$d(\cdot,\cdot)$ 是距离度量，$\boldsymbol{\theta}^-$ 表示停止梯度的参数，它可防止模型坍塌到平凡解（例如恒定预测）。

**不使用预训练扩散模型（Consistency Training）。** 没有预训练扩散模型时，仍可以用简单的单点近似直接估计 oracle score $\nabla_{{\mathbf{x}}}\log p_s({\mathbf{x}}_s)$，不过方差较高。回顾其条件期望形式：

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_s} \log p_s({\mathbf{x}}_s)
    &= {\mathbb{E}}_{{\mathbf{x}}_0 \sim p({\mathbf{x}}_0|{\mathbf{x}}_s)} \left[ \nabla_{{\mathbf{x}}_s}\log p({\mathbf{x}}_s|{\mathbf{x}}_0) \right] \\
    &= {\mathbb{E}}_{{\mathbf{x}}_0 \sim p({\mathbf{x}}_0|{\mathbf{x}}_s)} \left[ -\frac{{\mathbf{x}}_s - \alpha_s {\mathbf{x}}_0}{\sigma_s^2} \right].
\end{aligned}
$$

上述恒等式给出了一个简单的单样本估计量。如果 ${\mathbf{x}}_s$ 由配对样本 $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$ 通过 ${\mathbf{x}}_s = \alpha_s {\mathbf{x}}_0 + \sigma_s {\boldsymbol{\epsilon}}$ 得到，那么

$$
\widehat{\nabla_{{\mathbf{x}}}\log p_s}({\mathbf{x}}_s)
:=-\frac{\boldsymbol\epsilon}{\sigma_s}
=-\frac{{\mathbf{x}}_s-\alpha_s {\mathbf{x}}_0}{\sigma_s^2}
$$

就是 ${\mathbf{x}}_s$ 处 score 的*无偏*估计量（相对于 $p({\mathbf{x}}_0|{\mathbf{x}}_s)$ 条件无偏）。它恰好对应于去噪 score 匹配中作为回归目标的*条件 score*。

将这一估计代入从 $s$ 到 $s' = s-\Delta s$ 的 DDIM 单步更新（见[公式 (9.2.2)](../09-numerical-solvers/02-ddim.md#eq-ddim-update-all)），得到

<span id="eq-one-pt-score"></span>

$$
\begin{aligned}
    {\boldsymbol{\Psi}}_{s\to s'}({\mathbf{x}}_s)
&\approx
\frac{\alpha_{s'}}{\alpha_s}\,{\mathbf{x}}_s
+ \sigma_s^2 \left(\frac{\alpha_{s'}}{\alpha_s}-\frac{\sigma_{s'}}{\sigma_s}\right)\nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s) && \text{(DDIM)}
\\
&\approx
\frac{\alpha_{s'}}{\alpha_s}\,{\mathbf{x}}_s
+ \sigma_s^2 \left(\frac{\alpha_{s'}}{\alpha_s}-\frac{\sigma_{s'}}{\sigma_s}\right)\widehat{\nabla_{{\mathbf{x}}_s}\log p_s}({\mathbf{x}}_s) && \text{(1-pt score)}
\\
&=
\frac{\alpha_{s'}}{\alpha_s}\,{\mathbf{x}}_s
-\left(\frac{\alpha_{s'}}{\alpha_s}-\frac{\sigma_{s'}}{\sigma_s}\right)({\mathbf{x}}_s-\alpha_s{\mathbf{x}}_0)
\\
&= \alpha_{s'}{\mathbf{x}}_0+\sigma_{s'}{\boldsymbol{\epsilon}},
\end{aligned}
\tag{11.2.5}
$$

其中[^3]，${\mathbf{x}}_0$ 是同一个数据样本，${\boldsymbol{\epsilon}}$ 是同一个 Gaussian 噪声，二者均与构造 ${\mathbf{x}}_s$ 时所用的相同。

由此得到不依赖教师的离散时间替代目标，用以近似 oracle 目标 $\mathcal{L}_{\mathrm{oracle}\text{-}\mathrm{CM}}$，可写为

<span id="eq-ct-discrete-approx-n"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{CT}}^N(\boldsymbol{\theta}, \boldsymbol{\theta}^-)
    := {\mathbb{E}}_{{\mathbf{x}}_0, {\boldsymbol{\epsilon}}, i} \left[
       \omega(s_i)\, d \left({\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}_{s_{i+1}}, s_{i+1}),\,
       {\mathbf{f}}_{\boldsymbol{\theta}^-}({\mathbf{x}}_{s_i}, s_i)\right)
    \right],
\end{aligned}
\tag{11.2.6}
$$

其中 ${\mathbf{x}}_{s_i} = \alpha_{s_i}{\mathbf{x}}_0 + \sigma_{s_i}{\boldsymbol{\epsilon}}$，且 ${\mathbf{x}}_{s_{i+1}} = \alpha_{s_{i+1}}{\mathbf{x}}_0 + \sigma_{s_{i+1}}{\boldsymbol{\epsilon}}$。

直接使用 $\alpha_{s'}{\mathbf{x}}_0+\sigma_{s'}{\boldsymbol{\epsilon}}$ 近似 ${\boldsymbol{\Psi}}_{s\to s'}({\mathbf{x}}_s)$，而不取期望，会引入较高的方差[^4]。不过，可以回顾去噪 score 匹配中类似的条件期望论证（见[第 6.1 节](../06-unified-perspective/01-conditioning-tricks.md#sec-conditional-trick)）。这里，单点状态估计在传入非线性目标网络和差异度量之前是无偏的，因此最终的损失未必完全相同。以 ${\mathbf{x}}_s$ 为条件，一阶误差在期望中抵消；在合适的光滑性与矩条件下，剩余差异为 $\mathcal{O}(\Delta s^2)$，以下给出正式表述。

<span id="thm-ct-cm"></span>

**定理 11.2.1：CM 与 CT 在误差 $\mathcal{O}(\Delta s^2)$ 意义下等价**

令 $s' := s - \Delta s$，并定义

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CM}}({\boldsymbol{\theta}},{\boldsymbol{\theta}}^-)
&:= {\mathbb{E}}_{s,{\mathbf{x}}_0,{\boldsymbol{\epsilon}}} \big[w(s)\,d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s), {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_{s'}^{\mathrm{DDIM}},s')\big)\big],\\
\mathcal{L}_{\mathrm{CT}}({\boldsymbol{\theta}},{\boldsymbol{\theta}}^-)
&:= {\mathbb{E}}_{s,{\mathbf{x}}_0,{\boldsymbol{\epsilon}}} \big[w(s) d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s), {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_{s'},s')\big)\big],
\end{aligned}
$$

其中

$$
{\mathbf{x}}_{s'}^{\mathrm{DDIM}}
:= \frac{\alpha_{s'}}{\alpha_s}{\mathbf{x}}_s
+ \sigma_s^2 \left(\frac{\alpha_{s'}}{\alpha_s}-\frac{\sigma_{s'}}{\sigma_s}\right)\nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s)
$$

是 oracle DDIM 更新。${\mathbf{x}}_s = \alpha_s{\mathbf{x}}_0 + \sigma_s{\boldsymbol{\epsilon}}$ 与 ${\mathbf{x}}_{s'} = \alpha_{s'}{\mathbf{x}}_0 + \sigma_{s'}{\boldsymbol{\epsilon}}$ 使用同一对 $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$，其中 ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$ 且 ${\boldsymbol{\epsilon}} \sim \mathcal{N}(\boldsymbol{0},{\mathbf{I}})$。那么，

$$
\mathcal{L}_{\mathrm{CM}}({\boldsymbol{\theta}},{\boldsymbol{\theta}}^-)
= \mathcal{L}_{\mathrm{CT}}({\boldsymbol{\theta}},{\boldsymbol{\theta}}^-) + \mathcal{O}(\Delta s^2).
$$

**定理的证明。**

首先注意，使用 oracle score 的 DDIM 更新等于条件均值：

$$
{\mathbf{x}}_{s'}^{\mathrm{DDIM}} = {\mathbb{E}}[{\mathbf{x}}_{s'} | {\mathbf{x}}_s],
$$

这也可以从[公式 (11.2.5)](02-discrete-consistency-models.md#eq-one-pt-score)对 $p(\cdot|{\mathbf{x}}_s)$ 取期望得到。接下来，对

$$
d({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s), {\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\cdot,s'))
$$

在 ${\mathbf{x}}_{s'}^{\mathrm{DDIM}}={\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s]$ 附近进行 Taylor 展开。单点更新逐路径恢复 ${\mathbf{x}}_{s'}=\alpha_{s'}{\mathbf{x}}_0+\sigma_{s'}{\boldsymbol{\epsilon}}$，所用的是同一组 $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$。由于

$$
{\mathbb{E}}[{\mathbf{x}}_{s'}-{\mathbf{x}}_{s'}^{\mathrm{DDIM}}\mid{\mathbf{x}}_s]=0,
$$

以 ${\mathbf{x}}_s$ 为条件后，Taylor 展开的一次项消失。在上述光滑性与矩条件下，剩余 Taylor 余项的期望为 $\mathcal{O}(\Delta s^2)$。因此，$\mathcal{L}_{\mathrm{CT}} =\mathcal{L}_{\mathrm{CM}}+\mathcal{O}(\Delta s^2)$。详细推导见[第 D.5 节](../../appendices/d-supplementary-proofs/05-fast-generators.md#app-sec-flow-map)。

总体而言，CD 使用教师模型提供初始化与监督，这通常能稳定优化并降低方差。Consistency Training（CT）则不需要预训练模型，因而可以完全从头训练。尽管存在这一区别，CT 仍是一个完全独立的生成模型。

**实践中的考虑。** 在实践中，[Song et al. (2023)](../../references.md#ref-song2023consistency) 采用 [Karras et al. (2022)](../../references.md#ref-karras2022elucidating) 的 EDM 表述（见[第 D.6 节](../../appendices/d-supplementary-proofs/06-edm.md#sec-edm)），使用前向扰动核 ${\mathbf{x}}_s = {\mathbf{x}}_0 + s \boldsymbol{\epsilon}$，并采用其中提出的神经网络参数化（参见[公式 (D.6.1)](../../appendices/d-supplementary-proofs/06-edm.md#eq-d-parametrization)）：

$$
{\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}, s)
    = c_{\mathrm{skip}}(s) {\mathbf{x}}
      + c_{\mathrm{out}}(s)\,
        {\mathbf{F}}_{\boldsymbol{\theta}} \left(c_{\mathrm{in}}(s)\,{\mathbf{x}},\, c_{\mathrm{noise}}(s)\right),
$$

其中 ${\mathbf{F}}_{\boldsymbol{\theta}}$ 是神经网络，系数遵循[公式 (D.6.5)](../../appendices/d-supplementary-proofs/06-edm.md#eq-edm-simple-coefficients)。这一参数化具有重要的边界性质：

$$
{\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}, 0) = {\mathbf{x}},
$$

它在零时刻施加一致性，并确保没有噪声时网络的输出与输入相同。

## 11.2.2 使用 Consistency Model 采样 {#subsec-cm-sampling}

一致性模型 ${\mathbf{f}}_{\boldsymbol{\theta}^\times}$ 训练完成后，无论采用连续时间还是离散时间训练，都可用于单步或少步生成样本。[算法 9](02-discrete-consistency-models.md#alg-cm-sampling)概括了这一过程。

**单步生成。** 给定从先验分布采样的初始潜变量 $\hat{{\mathbf{x}}}_T$（实践中为 $\mathcal{N}(\boldsymbol{0}, T^2{\mathbf{I}})$），只需一次函数计算便可生成干净样本：${\mathbf{f}}_{\boldsymbol{\theta}^\times}(\hat{{\mathbf{x}}}_T, T)$。

**多步生成。** 预先选定时间步 $T>\tau_1 > \tau_2 > \cdots > \tau_{M-1}=0$，从初始噪声 $\hat{\mathbf{x}}_T$ 出发，交替注入噪声，并在更早的时间点通过一致性模型大步跳跃到干净状态，逐步改进样本：

$$
\begin{aligned}
\hat{\mathbf{x}}_T \xrightarrow[\text{get a clean}]{\text{long jump}} {\mathbf{f}}_{\boldsymbol{\theta}^\times}(\hat{\mathbf{x}}_T, T)
    \xrightarrow[\text{to level $\tau_1$}]{\text{add noise}} \hat{\mathbf{x}}_{\tau_1}
    \xrightarrow[\text{get a clean}]{\text{long jump}} {\mathbf{f}}_{\boldsymbol{\theta}^\times}(\hat{\mathbf{x}}_{\tau_1}, \tau_1)
    \xrightarrow[\text{to level $\tau_2$}]{\text{add noise}} \cdots.
\end{aligned}
$$

<section class="book-algorithm" id="alg-cm-sampling" aria-labelledby="alg-cm-sampling-title" markdown="1">

<div class="book-algorithm__title" id="alg-cm-sampling-title" markdown="span">
算法 9：CM 的单步或多步生成采样
</div>

<div class="book-algorithm__meta" markdown="1">

**输入：** 一致性模型 ${\mathbf{f}}_{\boldsymbol{\theta}^\times}(\cdot, \cdot)$，时间点序列 $T&gt;\tau_1 &gt; \tau_2 &gt; \cdots &gt; \tau_{M-1}=0$，初始噪声 $\hat{{\mathbf{x}}}_T$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="Algorithm 9 steps" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

**if** 单步 **then**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{x}} \leftarrow {\mathbf{f}}_{\boldsymbol{\theta}^\times}(\hat{{\mathbf{x}}}_T, T)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

**else**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{x}} \leftarrow {\mathbf{f}}_{\boldsymbol{\theta}^\times}(\hat{{\mathbf{x}}}_T, T)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

**for** $m=1$ **to** $M-1$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

采样 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">7</div>
<div class="book-algorithm__statement" markdown="1">

$\hat{{\mathbf{x}}}_{\tau_m} \leftarrow \alpha_{\tau_m}{\mathbf{x}} + \sigma_{\tau_m} \boldsymbol{\epsilon}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">8</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{x}} \leftarrow {\mathbf{f}}_{\boldsymbol{\theta}^\times}(\hat{{\mathbf{x}}}_{\tau_m}, \tau_m)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">9</div>
<div class="book-algorithm__statement" markdown="1">

**end for**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">10</div>
<div class="book-algorithm__statement" markdown="1">

**end if**

</div>
</div>

</div>
</div>

<div class="book-algorithm__meta" markdown="1">

**输出：** ${\mathbf{x}}$

</div>

</section>

[^1]: 半群性质由 ODE 初值问题的唯一性定理推出（见[附录 A](../../appendices/a-differential-equations/index.md#app-de)）。

[^2]: ODE 的一致性函数概念可以推广为 SDE 的函数 ${\mathbf{f}}({\mathbf{x}},t)$，使 ${\mathbf{f}}({\mathbf{x}}_t,t)$ 关于该 SDE 的自然滤过为（局部）鞅，即

    $$
    {\mathbb{E}}[{\mathbf{f}}({\mathbf{x}}_t,t)|{\mathbf{x}}_s]={\mathbf{f}}({\mathbf{x}}_s,s),\quad\text{for all }\quad t\ge s.
    $$

    这一推广在（[Daras et al., 2023](../../references.md#ref-daras2023consistent)；[Lai et al., 2023](../../references.md#ref-lai2023fp)）中被提出／观察到，相关理论联系汇总于（[Lai et al., 2023](../../references.md#ref-lai2023equivalence)）。

[^3]: 最后一个恒等式可由前向扰动过程 ${\mathbf{x}}_s=\alpha_s {\mathbf{x}}_0+\sigma_s {\boldsymbol{\epsilon}}$ 通过初等代数直接得到。

[^4]: 单点（条件）score 估计 $\widehat{\nabla_{{\mathbf{x}}}\log}p_s({\mathbf{x}}_s)$ 可以看作一个单样本 Monte Carlo 估计量，在给定 ${\mathbf{x}}_s$ 时*条件无偏*：对该估计量关于（通常不可直接计算的）干净数据后验 $p(\cdot|{\mathbf{x}}_s)$ 取平均，即可恢复真实 score：

    $$
    \nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s)
    = {\mathbb{E}}_{{\mathbf{x}}_0\sim p(\cdot|{\mathbf{x}}_s)}
     \left[\widehat{\nabla_{{\mathbf{x}}}\log}p_s({\mathbf{x}}_s)\right].
    $$

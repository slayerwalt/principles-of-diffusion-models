# 7.2 问题设定的分类 {#sec-ot-intro}

本节介绍将质量从 $p_{\mathrm{src}}$ 传输到 $p_{\mathrm{tgt}}$ 时，何谓最“高效”或“最优”。这些概念包括经典最优传输（OT）及其熵正则化变体，后者具有一种称为 Schrödinger 桥的等价表述。这一分类为后文阐明它们与扩散模型的联系提供背景。

## 7.2.1 最优传输（OT） {#subsec-optimal-transport}

<span id="fig-ot-2d-coupling"></span>

![相同二维分布之间的独立耦合与最优传输耦合](../../../en/assets/chapter-07/ot-2d-coupling.png)

**图 7.1：** **二维空间中相同 $p_{\mathrm{src}}$ 与 $p_{\mathrm{tgt}}$ 之间的两种耦合。** 左：独立（随机）配对会产生相互交叉、代价较高的路径。右：最优传输耦合将邻近的质量配对，避免路径交叉。

来源：作者通过 AI 辅助编程绘制。

**OT 问题的 Monge–Kantorovich（静态）表述。** 固定代价函数 $c:{\mathbb{R}}^D\times{\mathbb{R}}^D\to{\mathbb{R}}$，它规定了将概率质量从 ${\mathbf{x}}$ 送到 ${\mathbf{y}}$ 的开销。目标是以尽可能低的代价，将源分布 $p_{\mathrm{src}}$ 转换为目标分布 $p_{\mathrm{tgt}}$。

要定义代价，首先必须知道哪些点对 $({\mathbf{x}},{\mathbf{y}})$ 相互匹配。这正是*耦合*的作用：它是一个联合分布 $\gamma$，定义在 ${\mathbb{R}}^D\times{\mathbb{R}}^D$ 上，边缘分布分别为 $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$。换言之，采样 $({\mathbf{x}},{\mathbf{y}})\sim\gamma$ 表示将源分布中的 ${\mathbf{x}}$ 与目标分布中的 ${\mathbf{y}}$ 配对。如果 $\gamma$ 相对于 Lebesgue 测度具有密度 $\gamma({\mathbf{x}},{\mathbf{y}})$，则边缘约束写为

$$
\int_{{\mathbb{R}}^D} \gamma({\mathbf{x}},{\mathbf{y}})\,\mathop{}\!\mathrm{d} {\mathbf{y}} = p_{\mathrm{src}}({\mathbf{x}}),
\qquad
\int_{{\mathbb{R}}^D} \gamma({\mathbf{x}},{\mathbf{y}})\,\mathop{}\!\mathrm{d} {\mathbf{x}} = p_{\mathrm{tgt}}({\mathbf{y}}).
$$

也就是说，对 ${\mathbf{y}}$ 积分消去后得到关于 ${\mathbf{x}}$ 的源密度，而对 ${\mathbf{x}}$ 积分消去后得到关于 ${\mathbf{y}}$ 的目标密度。

下面给出两个标准例子：

1.  **离散支撑。** 如果 $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$ 的支撑集由有限个点组成，则耦合可以表示为非负矩阵 $(\gamma_{ij})$，其行和等于 $p_{\mathrm{src}}(i)$，列和等于 $p_{\mathrm{tgt}}(j)$。每个元素 $\gamma_{ij}$ 是从 $i$ 送至 $j$ 的质量。

2.  **确定性映射。** 如果存在可测映射 ${\mathbf{T}}$，满足 ${\mathbf{T}}_\#p_{\mathrm{src}}=p_{\mathrm{tgt}}$，则 $\gamma=({\mathbf{I}},{\mathbf{T}})_\#p_{\mathrm{src}}$ 是一个确定性耦合，将每个点 ${\mathbf{x}}$ 直接移到 ${\mathbf{T}}({\mathbf{x}})$。

一旦固定耦合 $\gamma$，传输代价就是该方案下的平均单位代价：

$$
\int c({\mathbf{x}},{\mathbf{y}})\,\mathop{}\!\mathrm{d}\gamma({\mathbf{x}},{\mathbf{y}})
={\mathbb{E}}_{({\mathbf{x}},{\mathbf{y}})\sim\gamma}\!\big[c({\mathbf{x}},{\mathbf{y}})\big].
$$

在离散情形中，它化为 $\sum_{i,j} c_{ij}\gamma_{ij}$；在连续情形中，它是一个二重积分。下文只关注连续情形。

最优传输问题就是从所有可行耦合中，选择使这一期望代价最小的耦合。

<span id="eq-ot"></span>

$$
\mathrm{OT}\big(p_{\mathrm{src}},p_{\mathrm{tgt}}\big)
:= \inf_{\gamma\in\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})}
\int c({\mathbf{x}},{\mathbf{y}})\,\mathop{}\!\mathrm{d}\gamma({\mathbf{x}},{\mathbf{y}}),
\tag{7.2.1}
$$

其中，可行集仅要求满足边缘约束，也即质量守恒约束：

$$
\begin{aligned}
\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})
= \Big\{\gamma \in &\mathcal{P}({\mathbb{R}}^D\times{\mathbb{R}}^D): \\
&\int \gamma({\mathbf{x}},{\mathbf{y}}) \mathop{}\!\mathrm{d} {\mathbf{y}} = p_{\mathrm{src}}({\mathbf{x}}),\,\,
\int \gamma({\mathbf{x}},{\mathbf{y}})\mathop{}\!\mathrm{d} {\mathbf{x}} = p_{\mathrm{tgt}}({\mathbf{y}})\Big\},
\end{aligned}
$$

这里，$\mathcal{P}({\mathbb{R}}^D\times{\mathbb{R}}^D)$ 表示 ${\mathbb{R}}^D\times{\mathbb{R}}^D$ 上所有概率测度的集合。

**一个特例：Wasserstein-2 距离。**

在二次代价 $c(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|^2$ 下，Monge–Kantorovich 问题的最优值是两个概率分布之间 Wasserstein-2 距离的平方：

$$
\begin{aligned}
\mathcal{W}_2^2(p_{\mathrm{src}}, p_{\mathrm{tgt}}) := \inf_{\gamma \in \Gamma(p_{\mathrm{src}}, p_{\mathrm{tgt}})} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma} \left[ \|\mathbf{x} - \mathbf{y}\|^2 \right].
\end{aligned}
$$

在对 $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$ 作出适当假设时，Brenier 定理（见[定理 7.1](05-probability-flow-odes-and-optimal-transport.md#thm-brenier)）[^1]保证，二次代价下的最优耦合 $\gamma$ 集中在某个确定性映射 ${\mathbf{T}}:\mathbb{R}^D \rightarrow \mathbb{R}^D$ 的图像上。因此，Wasserstein-2 距离的平方可以等价地表示为[^2]：

<span id="eq-monge-formulation"></span>

$$
\mathcal{W}_2^2(p_{\mathrm{src}}, p_{\mathrm{tgt}}) = \inf_{\substack{{\mathbf{T}}:\mathbb{R}^D \rightarrow \mathbb{R}^D,\\ \text{ s.t. }{\mathbf{T}}\#p_{\mathrm{src}} = p_{\mathrm{tgt}}}} \mathbb{E}_{{\mathbf{x}} \sim p_{\mathrm{src}}} \left[ \| {\mathbf{T}}({\mathbf{x}}) - {\mathbf{x}} \|^2 \right].
\tag{7.2.2}
$$

这里，${\mathbf{T}}\#p_{\mathrm{src}} = p_{\mathrm{tgt}}$ 表示 ${\mathbf{T}}$ 将 $p_{\mathrm{src}}$ 推前为 $p_{\mathrm{tgt}}$，即 ${\mathbf{T}}({\mathbf{x}}) \sim p_{\mathrm{tgt}}$，其中 ${\mathbf{x}} \sim p_{\mathrm{src}}$。

因此，Wasserstein-2 距离的平方表示在所有匹配给定边缘分布的耦合或传输映射中，最小的期望平方传输代价。记作 ${\mathbf{T}}^*(\mathbf{x})$ 的最优传输映射称为 *Monge 映射*，它给出了将 $p_{\mathrm{src}}$ 转换为 $p_{\mathrm{tgt}}$ 的最高效方式。

**OT 的 Benamou–Brenier（动态）表述。**

<span id="fig-ill-ot"></span>

![动态最优传输沿源分布与目标分布之间的直线位移插值进行](../../../en/assets/chapter-07/dynamic-optimal-transport.png)

**图 7.2：** **OT 的动态视角示意图。** 插值 $p_t^{\mathrm{OT}}$ 随时间连续演化，给出将 $p_{\mathrm{src}}$ 确定性地映射到 $p_{\mathrm{tgt}}$ 的最低代价传输方案（McCann 位移插值）。

来源：作者绘制。

除 Monge–Kantorovich 表述中直接以静态方式映射分布外，也可以将传输建模为连续时间流：

$$
p_0 := p_{\mathrm{src}} \to p_t \to p_1 := p_{\mathrm{tgt}}, \quad t \in [0,1].
$$

这一由 [Benamou and Brenier (2000)](../../references.md#ref-benamou2000computational) 提出的动态最优传输表述，寻找一个光滑的速度场 ${\mathbf{v}}_t({\mathbf{x}})$，描述 $p_t({\mathbf{x}})$ 中的质量如何随时间演化。

Benamou–Brenier 表述[^3]说明，对于二次代价 $c({\mathbf{x}}, {\mathbf{y}}) = \|{\mathbf{x}} - {\mathbf{y}}\|_2^2$（即 $\mathcal{W}_2$ 距离），[式 (7.2.1)](02-transport-problem-setups.md#eq-ot) 中静态 OT 问题的最优值，等于如下动能最小化问题的最优值：

<span id="eq-bb-ot"></span>

$$
\mathcal{W}_2^2(p_{\mathrm{src}}, p_{\mathrm{tgt}}) = \min_{\substack{ (p_t, {\mathbf{v}}_t) \text{ s.t. }
        \partial_t p_t + \nabla \cdot (p_t {\mathbf{v}}_t) = 0, \\
        p_0 = p_{\mathrm{src}},\,\, p_1 = p_{\mathrm{tgt}}
    }}
    \int_0^1 \int_{\mathbb{R}^D} \|{\mathbf{v}}_t({\mathbf{x}})\|^2 p_t({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}} \mathop{}\!\mathrm{d} t
\tag{7.2.3}
$$

其中，$p_t$ 是 $\mathbb{R}^D$ 上的概率分布，这对每个 $t \in [0,1]$ 都成立。具体来说，最优传输流 $p_t({\mathbf{x}})$ 遵循 *McCann 位移插值*：

$$
{\mathbf{T}}_t^*({\mathbf{x}}) = (1 - t){\mathbf{x}} + t{\mathbf{T}}^*({\mathbf{x}}),
$$

其中，${\mathbf{T}}^*({\mathbf{x}})$ 是将 $p_{\mathrm{src}}$ 传输到 $p_{\mathrm{tgt}}$ 的 OT 映射。这种线性插值使质量以恒定速度沿直线移动：$p_t = {\mathbf{T}}_t^*\#p_{\mathrm{src}}$ 对每个 $t \in [0,1]$ 都成立。

最优传输映射 ${\mathbf{T}}^*$ 满足 *Monge–Ampère 方程*：

<span id="eq-monge-ampere"></span>

$$
p_{\mathrm{tgt}}\left(\nabla \psi({\mathbf{x}})\right) \det\left(\nabla^2 \psi({\mathbf{x}})\right) = p_{\mathrm{src}}({\mathbf{x}}),
\tag{7.2.4}
$$

根据 Brenier 定理，有 ${\mathbf{T}}^*({\mathbf{x}}) = \nabla \psi({\mathbf{x}})$，其中 ${\boldsymbol{\psi}}$ 是某个凸函数。然而，这个非线性 PDE 通常难以求得显式解。注意，这恰好就是归一化流使用的变量替换关系（参见[式 (5.0.1)](../05-flow-based-perspective/index.md#eq-nf-change-of-var)）：流通过参数化可逆传输映射，使其 Jacobian 行列式可计算，但通常不施加势函数梯度结构 ${\mathbf{T}}^*=\nabla\psi$；因此，训练得到的流可能与 Brenier／OT 映射有很大差异。

## 7.2.2 熵正则化最优传输（EOT） {#subsec-entropic-optimal-transport}

为具体说明 EOT 的动机，考虑由样本构成的经验分布。设 $p_{\mathrm{src}}$ 的支撑点为 $\{{\mathbf{x}}^{(i)}\}_{i=1}^n \subset {\mathbb{R}}^D$，对应权重为 $a_i$；$p_{\mathrm{tgt}}$ 的支撑点为 $\{{\mathbf{y}}^{(j)}\}_{j=1}^n \subset {\mathbb{R}}^D$，对应权重为 $b_j$。此时，耦合是一个 $n\times n$ 非负矩阵 $\gamma=(\gamma_{ij})$，其行和匹配 $a$，列和匹配 $b$。每个元素 $\gamma_{ij}$ 表示从 ${\mathbf{x}}^{(i)}$ 传输到 ${\mathbf{y}}^{(j)}$ 的质量[^4]。

**为什么要对 OT 进行正则化？** 在这种离散设定中，经典 OT（在连续表述[式 (7.2.1)](02-transport-problem-setups.md#eq-ot)中采用计数测度即可得到）化为在所有可行耦合上最小化

$$
\min_{\gamma=(\gamma_{ij})}\sum_{i,j} C_{ij}\,\gamma_{ij},
$$

其中，可行耦合为 $\gamma=(\gamma_{ij})$；$C_{ij} = c({\mathbf{x}}^{(i)},{\mathbf{y}}^{(j)})$ 是将单位质量从源点 ${\mathbf{x}}^{(i)}$ 移到目标点 ${\mathbf{y}}^{(j)}$ 的代价，由给定的基础代价函数 $c:{\mathbb{R}}^D\times{\mathbb{R}}^D\to{\mathbb{R}}_{\ge 0}$（例如 $c({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_2^2$）确定。

这里有两个主要问题：

1.  **非唯一性与不稳定性：** 最小化解 $\gamma^*$ 不一定唯一。例如，如果两个传输方案达到相同的最小代价，求解器可能选择其中任意一个。因此，输入 $(a,b,C)$ 的微小变化（如移动样本、调整权重或轻微扰动代价）就可能使解发生突然跳变。

2.  **计算代价高：** 该问题是一个具有 $n^2$ 个变量和 $2n$ 个约束的线性规划。实际求解器（例如 Hungarian algorithm（[Kuhn, 1955](../../references.md#ref-kuhn1955hungarian); [Munkres, 1957](../../references.md#ref-munkres1957algorithms)）、网络单纯形法（[Peyré et al., 2019](../../references.md#ref-peyre2019computational)））通常具有 $\mathcal{O}(n^3)$ 的复杂度，当 $n$ 很大时难以使用。

为克服这些瓶颈，EOT 的目标函数在经典 OT 问题中加入由参数 $\varepsilon > 0$ 控制的正则项：

<span id="eq-eot-epsilon"></span>

$$
\begin{aligned}
        \text{EOT}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}}) &:= \min_{\gamma \in \Gamma(p_{\mathrm{src}}, p_{\mathrm{tgt}})} \int c({\mathbf{x}},  {\mathbf{y}})  \mathop{}\!\mathrm{d}\gamma({\mathbf{x}}, {\mathbf{y}})
        + \varepsilon \mathcal{D}_{\text{KL}}\left(\gamma \Vert M \right).
\end{aligned}
\tag{7.2.5}
$$

参考测度 $M$ 通常选为边缘分布的乘积 $p_{\mathrm{src}} \otimes p_{\mathrm{tgt}}$。KL散度项与传输方案 $\gamma$ 的 Shannon 熵直接相关：

$$
\mathcal{D}_{\text{KL}}(\gamma \,\Vert\, p_{\mathrm{src}} \otimes p_{\mathrm{tgt}}) = -\mathcal{H}(\gamma) + \text{Constant},
$$

其中 $\mathcal{H}(\gamma) := -\int \gamma({\mathbf{x}}, {\mathbf{y}}) \log \gamma({\mathbf{x}}, {\mathbf{y}}) \mathop{}\!\mathrm{d} {\mathbf{x}} \mathop{}\!\mathrm{d} {\mathbf{y}}$。

加入这一正则项会带来若干理论和实践上的好处，下面作简要介绍：

**熵正则项为何有帮助？**

1.  **质量分散。** 由于 $t\mapsto t\log t$ 是凸函数，且在 $t$ 较大时增长很快，最小化 $\int \gamma\log\gamma$ 会惩罚*尖峰状*的耦合（少数 $\gamma({\mathbf{x}},{\mathbf{y}})$ 很大，而大多数接近零）。在总质量 $\int \gamma=1$ 固定时，它偏好使 $\gamma({\mathbf{x}},{\mathbf{y}})$ 在 $({\mathbf{x}},{\mathbf{y}})\in\mathbb{R}^D\times\mathbb{R}^D$ 上分布更加均匀的方案。等价地，最大化 Shannon 熵会鼓励更高的“不确定性”（分散程度）。

2.  **严格凸性与唯一性。** 由于 $\mathcal{H}$ 严格凹，[式 (7.2.5)](02-transport-problem-setups.md#eq-eot-epsilon) 中的目标关于 $\gamma$ 严格凸，因此具有*唯一*的最小化解 $\gamma^*_\varepsilon$，且该解连续依赖于 $(p_{\mathrm{src}},p_{\mathrm{tgt}},c)$。

3.  **Sinkhorn 形式与正性。** 在较弱条件下[^5]，最优解具有 *Schrödinger／Sinkhorn 形式*

    $$
    \gamma^*_\varepsilon({\mathbf{x}},{\mathbf{y}}) = u({\mathbf{x}})\, \exp \bigl(-\tfrac{c({\mathbf{x}},{\mathbf{y}})}{\varepsilon}\bigr)  \,v({\mathbf{y}})p_{\mathrm{src}}({\mathbf{x}}) p_{\mathrm{tgt}}({\mathbf{y}}),
    $$

    其中缩放函数 $u,v$ 为正（相差一个全局因子意义下唯一）。实践中，通过有限样本近似连续表述，将 EOT 化为有限（采样的）Sinkhorn 迭代。熵正则化目标严格凸，缩放算法（Sinkhorn／IPFP）能够高效求解（[Sinkhorn, 1964](../../references.md#ref-sinkhorn1964relationship); [Cuturi, 2013](../../references.md#ref-cuturi2013sinkhorn)）。对于每个边缘分布有 $n$ 个支撑点的稠密问题（对应 $n\times n$ 核），每次 Sinkhorn 迭代需要 $\mathcal{O}(n^2)$ 时间和 $\mathcal{O}(n^2)$ 内存，因此更易扩展，也更实用（[Altschuler et al., 2017](../../references.md#ref-altschuler2017near)）。

4.  **$\varepsilon$ 的极限。** 当 $\varepsilon \to 0$ 时，最优方案 $\gamma^*_\varepsilon$ 越来越集中，趋近于一个（可能奇异的）经典 OT 耦合（我们将在[第 7.3.2 节](03-relationships-between-formulations.md#subsec-eot-ot)再次讨论这一联系）。随着 $\varepsilon$ 增大，$\gamma^*_\varepsilon$ 逐渐分散，并趋近于独立耦合 $p_{\mathrm{src}}\otimes p_{\mathrm{tgt}}$。

## 7.2.3 Schrödinger 桥（SB） {#subsec-sb}

**SB 的 KL 表述。** Schrödinger 桥（SB）问题由 Erwin Schrödinger 于 20 世纪 30 年代提出，它询问：假设粒子按照某种简单的参考动力学（例如 Brownian 运动）运动。现在设想我们在两个时刻观察粒子：在 $t=0$ 时，其分布为 $p_{\mathrm{src}}$；在 $t=1$ 时，其分布为 $p_{\mathrm{tgt}}$。在所有连接这两个分布的随机过程中，哪一个偏离参考动力学最少？这里用 KL散度度量“偏离”，因此 SB 问题的解，就是将 Brownian 运动变为满足给定边界条件的过程时，最可能的变形方式。

为精确定义，令 ${\mathbf{x}}_{0:T}:=\{{\mathbf{x}}_t\}_{t\in[0,T]}$ 表示该过程的一条完整轨迹。用 $P$ 表示*轨迹分布*，即定义在完整样本路径上的概率分布。时刻 $t$ 处，$P$ 的边缘分布记为 $p_t$（或 $P_t$），它描述单个时刻状态 ${\mathbf{x}}_t$ 的分布。形式上，对于可测集 $A\subseteq\mathbb R^D$，

$$
p_t(A) = P({\mathbf{x}}_t \in A).
$$

换言之，可以将 $p_t$ 看作从 $P$ 采样大量完整轨迹，再收集它们在时刻 $t$ 的状态而得到的经验分布；例如，状态是一维时可用直方图表示。

考虑由以下 SDE 支配的参考扩散 $\{{\mathbf{x}}_t\}_{t\in[0,T]}$：

<span id="eq-general-sb-sde"></span>

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}}_t  =  {\mathbf{f}}({\mathbf{x}}_t,t)\,\mathop{}\!\mathrm{d} t  +  g(t)\,\mathop{}\!\mathrm{d} {\mathbf{w}}_t,
\end{aligned}
\tag{7.2.6}
$$

其中 ${\mathbf{f}}\colon {\mathbb{R}}^D\times[0,T]\!\to\!{\mathbb{R}}^D$、$g\colon[0,T]\!\to\!{\mathbb{R}}$，且 $\{{\mathbf{w}}_t\}_{t\in[0,T]}$ 是标准 Brownian 运动。令 $R$ 表示完整轨迹 ${\mathbf{x}}_{0:T}:=\{{\mathbf{x}}_t\}_{t\in[0,T]}$ 的*路径分布*（联合分布），$R_0$ 表示参考过程的初始边缘分布；这个 $R$ 将作为*参考*轨迹分布。

采用这些记号，SB 问题寻找一个轨迹分布 $P$，使其在 KL散度意义下最接近 $R$，同时匹配给定的端点边缘分布：

<span id="eq-sb-epsilon-general"></span>

$$
\begin{aligned}
\mathrm{SB}(p_{\mathrm{src}}, p_{\mathrm{tgt}})
    := \min_{P}\, \mathcal{D}_{\mathrm{KL}}(P\|R)
    \quad \text{s.t.} \quad P_0 = p_{\mathrm{src}},\;\; P_T = p_{\mathrm{tgt}}.
\end{aligned}
\tag{7.2.7}
$$

最优解 $P^*$ 依赖于所选择的参考过程 $R$。

**SB 的随机控制视角。**

<span id="fig-ill-sb"></span>

![Schrödinger 桥的随机路径连接给定的源分布与目标分布](../../../en/assets/chapter-07/schrodinger-bridge.png)

**图 7.3：** **SB 的随机控制视角示意图。** 桥寻找在连接 $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$ 的同时，偏离参考过程最少的随机路径。

来源：作者绘制。

在[式 (7.2.7)](02-transport-problem-setups.md#eq-sb-epsilon-general)中直接优化任意路径分布 $P$ 之外，一种计算上更可行的方法是以参考动力学为基准，并允许它发生漂移。具体做法是引入随时间变化的漂移 ${\mathbf{v}}_t({\mathbf{x}}_t)$，扰动参考过程，产生一族候选轨迹分布。得到的动力学具有*受控扩散*形式：

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t
    = \bigl[{\mathbf{f}}({\mathbf{x}}_t,t) + {\mathbf{v}}_t({\mathbf{x}}_t)\bigr]\mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t,
$$

其中，${\mathbf{v}}_t:{\mathbb{R}}^D\to{\mathbb{R}}^D$ 是稍后需要优化的漂移（[式 (7.2.8)](02-transport-problem-setups.md#eq-sb-kinetic)）。在标准可积性条件（例如 Novikov 条件）下，根据 Girsanov 定理（见[第 C.2.1 节](../../appendices/c-ito-calculus/02-girsanov-theorem.md#subsec-girsanov)），受控分布 $P$ 与参考分布 $R$ 之间的 KL散度可写为动态（动能）形式

$$
\begin{aligned}
\mathcal{D}_{\mathrm{KL}}(P\|R)
    &=
    \mathcal{D}_{\mathrm{KL}}(p_{\mathrm{src}}\|R_0)
    +
    {\mathbb{E}}_{P}\!\left[
    \frac{1}{2}\int_0^T
    \frac{\|{\mathbf{v}}_t({\mathbf{x}}_t)\|^2}{g^2(t)}
    \mathop{}\!\mathrm{d} t
    \right]
    \\
    &=
    \mathcal{D}_{\mathrm{KL}}(p_{\mathrm{src}}\|R_0)
    +
    \frac{1}{2}\int_0^T\!\int_{{\mathbb{R}}^D}
    \frac{\|{\mathbf{v}}_t({\mathbf{x}})\|^2}{g^2(t)}
    p_t({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}\mathop{}\!\mathrm{d} t .
\end{aligned}
$$

其中，$p_t$ 表示在时刻 $t$，受控过程下 ${\mathbf{x}}_t$ 的边缘分布。第二个等号来自全期望公式。由于 SB 问题中的 $P_0=p_{\mathrm{src}}$ 固定，第一项相对于控制而言是常数，因此可以从优化中略去。特别地，当 $R_0=p_{\mathrm{src}}$ 时，该项为零。

因此，SB 问题可以重新表述为：在所有可行漂移 ${\mathbf{v}}_t$ 中最小化期望控制能量，这些漂移使过程从 $p_{\mathrm{src}}$（时刻 $t=0$）出发，到达 $p_{\mathrm{tgt}}$（时刻 $t=T$）（[Dai Pra, 1991](../../references.md#ref-dai1991stochastic); [Pra and Pavon, 1990](../../references.md#ref-pra1990markov); [Pavon and Wakolbinger, 1991](../../references.md#ref-pavon1991free); [Chen et al., 2016](../../references.md#ref-chen2016relation)）。由于初始边缘分布固定为 $p_{\mathrm{src}}$，可以令参考过程满足 $R_0=p_{\mathrm{src}}$，而不改变最优解；于是 $\mathcal{D}_{\mathrm{KL}}(p_{\mathrm{src}}\|R_0)=0$。这就得到*随机控制表述*：

<span id="eq-sb-kinetic"></span>

$$
\begin{aligned}
\mathrm{SB}(p_{\mathrm{src}}, p_{\mathrm{tgt}})
=
\min_{\substack{
\mathbf{v}_t \text{ s.t. }
\mathop{}\!\mathrm{d} {\mathbf{x}}_t =
\left[{\mathbf{f}}({\mathbf{x}}_t,t)+\mathbf{v}_t({\mathbf{x}}_t)\right]\mathop{}\!\mathrm{d} t
+g(t)\mathop{}\!\mathrm{d}{\mathbf{w}}_t,\\
{\mathbf{x}}_0\sim p_{\mathrm{src}},\,
{\mathbf{x}}_T\sim p_{\mathrm{tgt}}
}}
\frac{1}{2}
\int_0^T\int_{\mathbb{R}^D}
\frac{\left\lVert{\mathbf{v}}_t({\mathbf{x}})\right\rVert^2}{g^2(t)}
p_t({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}\mathop{}\!\mathrm{d} t .
\end{aligned}
\tag{7.2.8}
$$

这里的端点分布 $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$ 是任意的；选择控制 ${\mathbf{v}}_t$，正是为了使参考动力学在这两个边缘分布之间架起“桥梁”，同时在 KL散度意义下尽可能接近参考过程 $R$。

**一种特殊的 Brownian 参考过程。** [式 (7.2.8)](02-transport-problem-setups.md#eq-sb-kinetic) 与[式 (7.2.3)](02-transport-problem-setups.md#eq-bb-ot) 中 OT 的 Benamou–Brenier 表述相似，尤其是将参考过程 $R^\varepsilon$（满足 $\varepsilon>0$）选为 Brownian 运动时[^6]：

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = \sqrt{\varepsilon}\,\mathop{}\!\mathrm{d}{\mathbf{w}}_t,
$$

此时 ${\mathbf{f}} \equiv \mathbf{0}$ 且 $g(t) \equiv \sqrt{\varepsilon}$。

在这一设定中，SB 问题寻找路径分布 $P$，使其在 KL散度意义下最接近 Brownian 参考过程 $R^\varepsilon$，同时匹配端点边缘分布：

<span id="eq-sb-epsilon"></span>

$$
\begin{aligned}
\mathrm{SB}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}})
    := \min_{P} \mathcal{D}_{\mathrm{KL}}(P \| R^\varepsilon)
    \quad \text{s.t.} \quad P_0 = p_{\mathrm{src}}, \;\; P_T = p_{\mathrm{tgt}}.
\end{aligned}
\tag{7.2.9}
$$

等价的随机控制表述则变为

<span id="eq-sb-epsilon-kinetic"></span>

$$
\begin{aligned}
    \mathrm{SB}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}})
    = \min_{\substack{
        {\mathbf{v}}_t \text{ s.t. } \mathop{}\!\mathrm{d} {\mathbf{x}}_t = {\mathbf{v}}_t({\mathbf{x}}_t)\,\mathop{}\!\mathrm{d} t + \sqrt{\varepsilon}\,\mathop{}\!\mathrm{d} {\mathbf{w}}_t, \\
        {\mathbf{x}}_0 \sim p_{\mathrm{src}},\;\; {\mathbf{x}}_T \sim p_{\mathrm{tgt}}
    }}
    \frac{1}{2\varepsilon}\int_0^T \!\!\int_{{\mathbb{R}}^D}
        \|{\mathbf{v}}_t({\mathbf{x}})\|^2 \, p_t({\mathbf{x}})\,\mathop{}\!\mathrm{d} {\mathbf{x}}\,\mathop{}\!\mathrm{d} t,
\end{aligned}
\tag{7.2.10}
$$

其中，$p_t$ 表示时刻 $t$ 的边缘分布，对应扩散由漂移 ${\mathbf{v}}_t$ 驱动：

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = {\mathbf{v}}_t({\mathbf{x}}_t)\,\mathop{}\!\mathrm{d} t + \sqrt{\varepsilon}\,\mathop{}\!\mathrm{d} {\mathbf{w}}_t.
$$

**为什么需要指定参考分布？** 经典 OT 不需要参考路径分布，因为给定的基础代价已经说明了源点与目标点配对的开销。因此，优化直接通过这一代价比较可行耦合。然而，在 SB 问题中，我们比较的是完整随机轨迹，而不仅是它们的端点。因此，需要一个参考路径测度 $R$ 来规定未受控动力学通常是什么样的。于是，$\mathcal{D}_{\mathrm{KL}}(P\|R)$ 度量了可行过程 $P$ 为满足给定端点分布，必须偏离参考动力学的程度。

**耦合 PDE 刻画。** 描述 SB 解的一种便捷方式是使用两个时空势函数 $\Psi(x,t)$ 和 $\widehat{\Psi}(x,t)$。令 $p_t^{\mathrm{SB}}$ 表示时刻 $t\in[0,T]$ 处的边缘分布，它来自[式 (7.2.7)](02-transport-problem-setups.md#eq-sb-epsilon-general)中的最优轨迹分布 $P^*$。则有如下对称分解（[Dai Pra, 1991](../../references.md#ref-dai1991stochastic)）

<span id="eq-sb-symmetry"></span>

$$
\begin{aligned}
p_t^{\mathrm{SB}}(x) = \Psi(x,t) \widehat{\Psi}(x,t),
\end{aligned}
\tag{7.2.11}
$$

其中，$\Psi$ 和 $\widehat{\Psi}$ 满足（线性的）*Schrödinger 系统*（[Caluya and Halder, 2021](../../references.md#ref-caluya2021wasserstein); [Chen et al., 2021](../../references.md#ref-chen2021stochastic); [Chen et al., 2022](../../references.md#ref-chenlikelihood)）：

<span id="eq-pde-schrodinger"></span>

$$
\begin{aligned}
\frac{\partial \Psi}{\partial t}(\mathbf{x},t)
  &= -\nabla_{\mathbf{x}} \Psi(\mathbf{x},t) \cdot {\mathbf{f}}(\mathbf{x},t)
     - \frac{g^2(t)}{2}\,\Delta_\mathbf{x} \Psi(\mathbf{x},t),  \\
\frac{\partial \widehat{\Psi}}{\partial t}(\mathbf{x},t)
  &= -\nabla_{\mathbf{x}}\!\cdot\!\bigl(\widehat{\Psi}(\mathbf{x},t)\,{\mathbf{f}}(\mathbf{x},t)\bigr)
     + \frac{g^2(t)}{2}\,\Delta_\mathbf{x}\widehat{\Psi}(\mathbf{x},t)
\\ \text{subject to}\quad
\Psi(\mathbf{x},0)&\,\widehat{\Psi}(\mathbf{x},0) = p_{\mathrm{src}}(\mathbf{x}),
\quad
\Psi(\mathbf{x},T)\,\widehat{\Psi}(\mathbf{x},T) = p_{\mathrm{tgt}}(\mathbf{x}).
\end{aligned}
\tag{7.2.12}
$$

**正时间 Schrödinger 桥 SDE。** 一旦知道 $\Psi$，最优动力学就是由时空因子 $\Psi$ 倾斜后的参考扩散：

<span id="eq-sb-forward-sde"></span>

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = \bigl[{\mathbf{f}}({\mathbf{x}}_t, t) + g^2(t)\,\nabla_{\mathbf{x}} \log \Psi({\mathbf{x}}_t, t)\bigr]\mathop{}\!\mathrm{d} t
+ g(t)\mathop{}\!\mathrm{d} {\mathbf{w}}_t, \quad {\mathbf{x}}_0 \sim p_{\mathrm{src}}.
\end{aligned}
\tag{7.2.13}
$$

令 $Q$ 表示[式 (7.2.13)](02-transport-problem-setups.md#eq-sb-forward-sde) 的轨迹分布（因此由[式 (7.2.11)](02-transport-problem-setups.md#eq-sb-symmetry)及[式 (7.2.12)](02-transport-problem-setups.md#eq-pde-schrodinger)可得 $Q_0=p_{\mathrm{src}}$ 和 $Q_T=p_{\mathrm{tgt}}$）。于是 *$Q=P^*$*，且[式 (7.2.8)](02-transport-problem-setups.md#eq-sb-kinetic)的最小化解 ${\mathbf{v}}^*$ 为（见 [Chen et al., 2021](../../references.md#ref-chen2021stochastic) 的第 4.6 节）：

$$
{\mathbf{v}}_t^{*}({\mathbf{x}}) = g^2(t) \nabla_{\mathbf{x}} \log \Psi({\mathbf{x}},t).
$$

也就是说，漂移修正 $g^2\nabla_{\mathbf{x}}\log\Psi$ 恰好是在匹配端点边缘分布所需的扰动中，相对于参考过程 KL散度最小的扰动。

**逆时间 Schrödinger 桥 SDE。** 同一个最优路径分布也可以沿逆时间生成。理解逆时间漂移形式的一种便捷方式，是从概念上使用扩散的标准时间反演恒等式：

$$
{\mathbf{b}}^{-}({\mathbf{x}},t)={\mathbf{b}}^{+}({\mathbf{x}},t)-g^2(t)\nabla_{\mathbf{x}}\log p_t^{\mathrm{SB}}({\mathbf{x}}),
$$

其中 ${\mathbf{b}}^{+}={\mathbf{f}}+g^2\nabla\log\Psi$ 且 $p_t=\Psi\,\widehat{\Psi}$。由此得到

$$
{\mathbf{b}}^{-}({\mathbf{x}},t)={\mathbf{f}}({\mathbf{x}},t)-g^2(t) \nabla_{\mathbf{x}}\log \widehat{\Psi}({\mathbf{x}},t).
$$

因此，逆时间 SDE 写为

<span id="eq-sb-backward-sde"></span>

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}}_t
= \Big[{\mathbf{f}}({\mathbf{x}}_t,t) - g^2(t) \nabla_{{\mathbf{x}}}\log \widehat{\Psi}({\mathbf{x}}_t,t)\Big]\mathop{}\!\mathrm{d} t
+ g(t)\,\mathop{}\!\mathrm{d} \bar{{\mathbf{w}}}_t, \quad {\mathbf{x}}_T \sim p_{\mathrm{tgt}} .
\end{aligned}
\tag{7.2.14}
$$

等价地，用 ${\mathbf{y}}_\tau := {\mathbf{x}}_{T-\tau}$ 重新参数化时间，使 $\tau$ 从 $0$ 增至 $T$。于是，${\mathbf{y}}_\tau$ 沿 $\tau$ 的正方向、从 ${\mathbf{y}}_0 \sim p_{\mathrm{tgt}}$ 出发演化如下：

<span id="eq-sb-reverse-forward-tau"></span>

$$
\begin{aligned}
    \mathop{}\!\mathrm{d} {\mathbf{y}}_\tau
= \big[-{\mathbf{f}}({\mathbf{y}}_\tau,T-\tau) + g^2(T-\tau) \nabla_{{\mathbf{y}}}\log \widehat{\Psi}({\mathbf{y}}_\tau,T-\tau)\big]\mathop{}\!\mathrm{d} \tau
+ g(T-\tau) \mathop{}\!\mathrm{d} {\mathbf{w}}_\tau.
\end{aligned}
\tag{7.2.15}
$$

正向与逆向描述都给出相同的最优路径分布 $P^*$，二者通过下式联系起来

$$
\nabla\log p_t^{\mathrm{SB}}=\nabla\log\Psi+\nabla\log\widehat\Psi,
\qquad
{\mathbf{b}}^{-}={\mathbf{b}}^{+}-g^2\,\nabla\log p_t^{\mathrm{SB}},
$$

因此，它们在每个时刻的边缘分布都等于 $p_t^{\mathrm{SB}}$。附加漂移项 $g^2\nabla\log\Psi$（正向）和 $-\,g^2\nabla\log\widehat{\Psi}$（逆时间）起到控制力的作用，引导参考扩散匹配端点边缘分布，同时在相对熵意义下最接近参考过程。

**耦合 PDE 方法的实际障碍。** 要基于[式 (7.2.14)](02-transport-problem-setups.md#eq-sb-backward-sde)构造生成过程，就必须求解[式 (7.2.12)](02-transport-problem-setups.md#eq-pde-schrodinger)中的耦合 PDE，得到后向 Schrödinger 势函数 $\widehat{\Psi}$。然而，即使在低维设定中，这些 PDE 也很难求解，因此难以直接用于生成建模。为绕过这一问题，一些研究提出了替代策略：利用 Score SDE 技术迭代求解各个半桥问题（$p_{\mathrm{tgt}} \leftarrow p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}} \rightarrow p_{\mathrm{src}}$）（[De Bortoli et al., 2021](../../references.md#ref-de2021diffusion)）；优化似然的代理界（[Chen et al., 2022](../../references.md#ref-chenlikelihood); [Liu et al., 2023](../../references.md#ref-liu20232)）；或基于后验 ${\mathbf{x}}_t \vert {\mathbf{x}}_0, {\mathbf{x}}_T$ 在样本对 $({\mathbf{x}}_0, {\mathbf{x}}_T) \sim p_{\mathrm{src}} \otimes p_{\mathrm{tgt}}$ 下的解析解，设计无模拟训练（[Liu et al., 2023](../../references.md#ref-liu20232)）。这里不展开技术细节，只在[第 7.4 节](04-diffusion-sdes-and-schrodinger-bridges.md#sec-dm-and-sb)简要讨论扩散模型与 SB 的联系。

## 7.2.4 全局推前与局部动力学：深度生成模型的 OT 类比 {#subsec-global-pushforwards-local-dynamics}

从最优传输视角（[式 (7.2.1)](02-transport-problem-setups.md#eq-ot)）看，可以使用深度生成模型学习从简单先验到数据的传输（推前）映射，即 ${\mathbf{G}}_{\boldsymbol{\phi}\#}p_{\mathrm{prior}} \approx p_{\mathrm{data}}$。尽管 ${\mathbf{G}}_{\boldsymbol{\phi}}$ 一般不等于最优传输映射（在适当条件下施加 OT 目标的工作除外（[Genevay et al., 2018](../../references.md#ref-genevay2018learning); [Onken et al., 2021](../../references.md#ref-onken2021ot)）），Benamou–Brenier 表述（[式 (7.2.3)](02-transport-problem-setups.md#eq-bb-ot)）提供了互补的动态视角。它不直接学习单个全局映射，而是将传输描述为由时变局部向量场生成的连续流，在 $p_{\mathrm{prior}}$ 与 $p_{\mathrm{data}}$ 之间描绘出一条光滑路径。这种动态表述类似于静态 Schrödinger 桥问题（[式 (7.2.7)](02-transport-problem-setups.md#eq-sb-epsilon-general)）与其随机控制对应形式（[式 (7.2.8)](02-transport-problem-setups.md#eq-sb-kinetic)）之间的关系，后者将最优耦合实现为受控扩散过程。生成建模中也存在类似关系：GAN、VAE 等标准深度生成模型学习全局推前映射，而扩散模型学习随时间变化的局部向量场，用于驱动生成动力学。

[^1]: Brenier 定理讨论二次代价下最优传输映射的存在性与结构。具体而言，如果 $p_{\mathrm{src}}$ 不赋予维数至多为 $D-1$ 的集合任何质量，则最优传输映射 ${\mathbf{T}}^*$ 唯一存在。

[^2]: $\mathcal{W}_2$ 距离有三种常见表述：Monge 表述（基于最优传输映射）、Kantorovich 表述（基于耦合），以及 Benamou–Brenier 动态表述（见[式 (7.2.3)](02-transport-problem-setups.md#eq-bb-ot)）。在适当正则性条件下，它们彼此等价。

[^3]: Benamou–Brenier 表述描述了如何在测度与速度的连续路径上最小化动能，从而计算 $\mathcal{W}_2$ 距离。

[^4]: 经验（离散）测度为连续分布提供了有理论依据的近似。当基础代价为 $c(x,y)=d(x,y)^p$（因此 OT 值等于 $W_p^p$），且测度具有有限的 $p$ 阶矩时，经验测度会在 $W_p$ 意义下以可量化的速率收敛到总体；见 [Fournier and Guillin (2015)](../../references.md#ref-fournier2015rate) 及 [Peyré et al. (2019)](../../references.md#ref-peyre2019computational) 中的综述。

[^5]: 假设 $c<\infty$ 在 $p_{\mathrm{src}}\otimes p_{\mathrm{tgt}}$ 几乎处处成立，且边缘核积分有限且为正。为简单起见，我们关注 $\gamma^*_\varepsilon$、$p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$ 相对于 Lebesgue 测度具有密度的情况。

[^6]: 这种相似性可以精确表述。在等价的时间对称流体动力学表述中，如果 $\tilde{{\mathbf{v}}}_t$ 表示满足连续性方程 $\partial_t \rho_t + \nabla \cdot (\rho_t \tilde{{\mathbf{v}}}_t)=0$ 的流速，则 Brownian 参考过程下的 SB 目标（[式 (7.2.10)](02-transport-problem-setups.md#eq-sb-epsilon-kinetic)）可以写成一个 Benamou–Brenier 型动能项（相差惯例中的常数因子）加上一个 Fisher 信息项：

    $$
    \int_0^T\!\!\int
    \left[
    \frac{1}{2}\|\tilde{{\mathbf{v}}}_t({\mathbf{x}})\|^2
    +
    \frac{\varepsilon^2}{8}\|\nabla_{{\mathbf{x}}}\log \rho_t({\mathbf{x}})\|^2
    \right]\rho_t({\mathbf{x}})\,\mathop{}\!\mathrm{d}{\mathbf{x}}\,\mathop{}\!\mathrm{d} t.
    $$

    因此，在 Brownian 参考过程／二次代价的情形下，可以将熵正则化 OT 看作对动态 OT 加入 Fisher 信息正则项后的形式；见 [Chen et al. (2021)](../../references.md#ref-chen2021stochastic) 中的问题 4.6。

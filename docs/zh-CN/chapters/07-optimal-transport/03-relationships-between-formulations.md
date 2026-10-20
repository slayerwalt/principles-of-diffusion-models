# 7.3 不同最优传输表述之间的关系 {#sec-relation-ot}

<span id="fig-ot-dm"></span>

![路径空间与随机控制形式的 Schrödinger 桥、熵正则化最优传输和经典最优传输之间的联系](../../../en/assets/chapter-07/transport-formulations.png)

**图 7.4：** **最优传输各变体之间的关系，其中 SB 使用 $c({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_2^2$ 及参考过程 $R^\varepsilon$。** 这些关系概括如下：(i) $\mathrm{SB}_\varepsilon$ 的随机控制表述与路径空间 KL 表述等价；(ii) 对于 Brownian 参考过程及相应的二次代价归一化，$\mathrm{SB}_\varepsilon$ 的端点投影给出 $\mathrm{EOT}_\varepsilon$；(iii) $\mathrm{EOT}_\varepsilon$ 在 $\varepsilon\to0$ 时收敛到静态 OT；(iv) 当 $\varepsilon\to0$ 时，经过适当缩放的 Brownian 参考 Schrödinger 问题收敛到动态 OT。

来源：作者绘制。

图中链接：[式 (7.2.10)](02-transport-problem-setups.md#eq-sb-epsilon-kinetic) 和[式 (7.2.3)](02-transport-problem-setups.md#eq-bb-ot)。

在介绍技术细节之前，先澄清最优传输及其熵正则化的不同表述如何相互联系，会有所帮助。从整体上看，这些问题具有如下关系（关系图见[图 7.4](03-relationships-between-formulations.md#fig-ot-dm)）：

-   **(i)** **路径空间 SB $\Longleftrightarrow$ 随机控制 SB。** $\mathrm{SB}_\varepsilon$ 的路径空间 KL 表述及其随机控制表述，描述的是相对于选定参考过程的同一个桥问题（见[第 7.2.3 节](02-transport-problem-setups.md#subsec-sb)）。

-   **(ii)** **Brownian 参考 SB $\Longleftrightarrow$ EOT$_\varepsilon$。** 对于 Brownian 参考过程

    $$
    \mathop{}\!\mathrm{d} {\mathbf{x}}_t=\sqrt{\varepsilon}\,\mathop{}\!\mathrm{d}{\mathbf{w}}_t,
    $$

    以及相应的二次代价归一化，最优 SB 路径分布诱导的端点耦合求解了对应的 $\mathrm{EOT}_\varepsilon$ 问题（见[第 7.3.1 节](03-relationships-between-formulations.md#subsec-sb-eot)）。

-   **(iii)** **EOT$_\varepsilon$ $\longrightarrow$ 静态 OT。** 当 $\varepsilon\to0$ 时，熵正则化端点耦合趋近于相应的经典 OT 问题；在[第 7.3.2 节](03-relationships-between-formulations.md#subsec-eot-ot)讨论的假设下，最优值收敛到 OT 值。

-   **(iv)** **SB$_\varepsilon$ $\longrightarrow$ 动态 OT。** 在相应的小噪声缩放下，当 $\varepsilon\to0$ 时，Brownian 参考 Schrödinger 问题收敛到 OT 的 Benamou–Brenier 动态表述（见[第 7.3.3 节](03-relationships-between-formulations.md#subsec-sb-ot)）。

这些并不显然的关系，共同将随机控制、熵正则化及经典 OT 框架联系起来。

<span id="fig-eot-epsilon"></span>

![增大熵正则化会使端点耦合更分散，并增大传输路径的波动](../../../en/assets/chapter-07/entropic-regularization.png)

**图 7.5：** **正则化参数 $\varepsilon$ 对熵正则化最优传输的影响（关系 (iii) 和 (iv)）。** 随着 $\varepsilon$ 增大，两件事同时发生：耦合 $\gamma^*_\varepsilon$ 将质量分散到更远的目标，而传输路径也越来越具有随机性。左（$\varepsilon=0.01$）：路径近乎直线，每个源点映射到单个邻近目标，恢复经典 OT。中（$\varepsilon=0.5$）：路径出现明显波动，耦合开始分散。右（$\varepsilon=2.5$）：路径具有很强的随机性，趋近于 Schrödinger 桥的情形；在极限 $\varepsilon\to\infty$ 下，耦合退化为独立乘积 $p_{\mathrm{src}}\otimes p_{\mathrm{tgt}}$。耦合通过 Sinkhorn 算法计算（$\varepsilon\to 0$ 时使用 Hungarian algorithm）；给定端点后，路径是扩散系数为 $\sigma=\sqrt{\varepsilon/2}$ 的 Brownian 桥，这来自 Mikami 对偶关系（[第 7.3.1 节](03-relationships-between-formulations.md#subsec-sb-eot)）。

来源：作者通过 AI 辅助编程绘制。

## 7.3.1 SB 与 EOT（对偶）等价 {#subsec-sb-eot}

本节从两个互补的视角说明，SB 与 EOT 本质上等价。经典最优传输产生单个确定性映射，而 SB 产生粒子的*随机*流：质量以概率方式传输，边缘分布则按照类似扩散的动力学演化。

从静态视角看，SB 与 EOT 一致，目标都是寻找两个端点分布之间的耦合，在传输代价与熵之间取得平衡。从动态视角看，SB 描述一个受控扩散过程，使其在匹配所需端点的同时，尽可能接近简单参考过程（如 Brownian 运动）。每个视角都独立建立了这一等价性，为将 SB／EOT 理解为分布到分布转换的标准表述，提供了两种相互一致的方式。

**静态 Schrödinger 桥。** 令

$$
\widetilde R^\varepsilon({\mathbf{x}},{\mathbf{y}})
  := \frac{1}{Z_\varepsilon}\,e^{-c({\mathbf{x}},{\mathbf{y}})/\varepsilon}\,p_{\mathrm{src}}({\mathbf{x}})\,p_{\mathrm{tgt}}({\mathbf{y}}),
$$

归一化常数为

$$
Z_\varepsilon := \iint e^{-c({\mathbf{x}}, {\mathbf{y}})/\varepsilon}\, p_{\mathrm{src}}({\mathbf{x}})\, p_{\mathrm{tgt}}({\mathbf{y}})\,\mathrm d{\mathbf{x}}\,\mathrm d{\mathbf{y}}.
$$

那么，对于任意可行耦合 $\gamma\in\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})$，直接计算可得

$$
\int c\, \mathrm d\gamma
+\varepsilon \mathcal{D}_\mathrm{KL}\!\bigl(\gamma \Vert p_{\mathrm{src}}\!\otimes\!p_{\mathrm{tgt}}\bigr)
=
\varepsilon \mathcal{D}_\mathrm{KL}\!\bigl(\gamma \Vert \widetilde R^\varepsilon\bigr)
-\varepsilon\log Z_\varepsilon.
$$

因此，

<span id="eq-eot-sb-static"></span>

$$
\begin{aligned}
    \min_{\gamma\in\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})}
  \Bigl\{\int c \,\mathrm d\gamma
  + \varepsilon \mathcal{D}_\mathrm{KL}\!\bigl(\gamma \Vert p_{\mathrm{src}}\!\otimes\!p_{\mathrm{tgt}}\bigr)\Bigr\}
  = \varepsilon \min_{\gamma\in\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})}
  &\mathcal{D}_\mathrm{KL}\!\bigl(\gamma \Vert \widetilde R^\varepsilon\bigr)
  \\&-\varepsilon\log Z_\varepsilon,
\end{aligned}
\tag{7.3.1}
$$

所以，熵正则化 OT 问题在相差一个加性常数的意义下，等价于静态 Schrödinger 桥问题

$$
\min_{\gamma\in\Gamma(p_{\mathrm{src}},p_{\mathrm{tgt}})}
\mathcal{D}_\mathrm{KL}(\gamma\Vert \widetilde R^\varepsilon).
$$

**动态等价性（Brownian 参考过程）。** 也可以从动力学视角理解这一等价性。一个经典结果（[Mikami and Thieullen, 2006](../../references.md#ref-mikami2006duality)）指出，二次代价下的熵正则化 OT

$$
c({\mathbf{x}},{\mathbf{y}})=\frac{\|{\mathbf{y}}-{\mathbf{x}}\|^2}{2T}
$$

与以下 SB 问题仿射等价：其参考路径分布 $R^\varepsilon$ 是 $[0,T]$ 上的 Brownian 运动，

$$
\mathrm{d}{\mathbf{x}}_t=\sqrt{\varepsilon}\,\mathrm{d}{\mathbf{w}}_t.
$$

这里，“仿射等价”表示最优值仅相差一个正比例缩放和一个加性常数（与决策变量无关），因此最小化解相同。具体而言，令 $P^*$ 为 SB 的最优路径分布。只观察这一最优随机过程的两个端点，就能得到它与 EOT 的关键联系。如果 ${\mathbf{x}}_{[0:T]}\sim P^*$，则端点对 $({\mathbf{x}}_0,{\mathbf{x}}_T)$ 诱导耦合

$$
\gamma^*:=({\mathbf{x}}_0,{\mathbf{x}}_T)_\#P^*,
$$

而 $\gamma^*$ 是对应 EOT 问题的最优传输方案。

反过来，从一个最优 EOT 耦合 $\gamma^*$ 出发，可以先采样一对端点 $({\mathbf{x}}_0,{\mathbf{x}}_T)\sim\gamma^*$，再以这些端点为条件，从连接 ${\mathbf{x}}_0$ 到 ${\mathbf{x}}_T$ 的条件参考扩散中采样轨迹。按照 $\gamma^*$ 混合这些条件扩散路径，就能恢复最优 SB 路径分布 $P^*$。

也就是说，动态 SB 问题的最优随机过程诱导了静态 EOT 问题的最优端点耦合。反之，最优 EOT 耦合规定端点如何配对，参考扩散则规定每对端点之间的最优随机路径。

推导这一事实的关键是，路径上的 KL散度可以按照端点进行分解，这意味着 Schrödinger 桥问题可化为只涉及 $({\mathbf{x}}_0,{\mathbf{x}}_T)$ 联合分布的 KL散度。对于 Brownian 运动，${\mathbf{x}}$ 与 ${\mathbf{y}}$ 之间的转移密度具有 Gaussian 形式，因此其负对数是二次函数：

$$
-\varepsilon\log p_T({\mathbf{y}}\mid{\mathbf{x}})=\frac{\|{\mathbf{y}}-{\mathbf{x}}\|^2}{2T}+\text{const}.
$$

这说明，端点 KL 与二次代价下的熵正则化 OT 目标完全相同，只相差一个无关常数。

**一般参考过程下的 SB 决定端点 EOT 代价。** SB 与 EOT 的联系不限于 Brownian 运动。对于一般适定的参考过程，其转移规律告诉我们，从 ${\mathbf{x}}$（时刻 $0$）到 ${\mathbf{y}}$（时刻 $T$）的运动，在参考动力学下有多“自然”。因此，参考过程下不太可能出现的端点对应具有更大的传输代价。

更精确地，令 $p_T({\mathbf{y}}|{\mathbf{x}})$ 表示参考过程在 $[0,T]$ 上的转移密度。除去比例缩放以及仅依赖固定端点边缘分布的加性项，相应的端点代价具有如下形式

$$
c({\mathbf{x}},{\mathbf{y}})
\propto
-\log p_T({\mathbf{y}}|{\mathbf{x}}).
$$

因此，在端点耦合层面，参考动力学诱导了对应 EOT 问题使用的代价：较可能的转移代价较小，较不可能的转移代价较大。

完整参考过程所含的信息不止这一端点代价。一旦选定端点对 $({\mathbf{x}},{\mathbf{y}})$，参考动力学还决定连接这些端点的条件随机路径。在这一意义上，EOT 决定端点如何最优配对，而 SB 还进一步描述每对端点之间的随机传输。

## 7.3.2 EOT$_\varepsilon$ 在 $\varepsilon\to 0$ 时化为 OT {#subsec-eot-ot}

令 $\gamma^*_\varepsilon$ 表示 $\mathrm{EOT}_\varepsilon$ 的最优方案，$\gamma^*$ 表示[式 (7.2.1)](02-transport-problem-setups.md#eq-ot)中无正则化 OT 问题的一个最优方案。在标准假设下，当 $\varepsilon\to0$ 时，EOT 的最优值收敛到 OT 值，且任何熵正则化最优方案的极限都是 OT 最优的；当 OT 最优解唯一时，最优方案收敛到 $\gamma^*$。

这一收敛结果既是基础性的，也具有实践意义。其中一个原因是，熵正则化 OT 问题 $\mathrm{EOT}_\varepsilon$ 可以通过 Sinkhorn 等算法高效地数值求解。因此，该结果为使用 $\mathrm{EOT}_\varepsilon$（取较小的 $\varepsilon$）作为[式 (7.2.1)](02-transport-problem-setups.md#eq-ot)中经典 OT 问题的可计算近似提供了理论依据，即使代价函数 $c({\mathbf{x}}, {\mathbf{y}})$ 比二次情形更一般也是如此。

<span id="thm-eot-ot"></span>

**定理 7.3.1：（非正式）EOT$_\varepsilon$ 收敛到 OT。**

在保证熵正则化 OT 极限成立的标准假设下，当 $\varepsilon \to 0$ 时，最优值收敛：

$$
\lim_{\varepsilon \to 0}
\mathrm{EOT}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}})
=
\mathrm{OT}(p_{\mathrm{src}}, p_{\mathrm{tgt}}).
$$

此外，任何最优方案的极限都是无正则化 OT 问题的最优耦合。特别地，当 OT 最优解唯一时，熵正则化最优方案收敛到该最优解。

**定理证明。**

严格证明参见相关文献（[Mikami and Thieullen, 2008](../../references.md#ref-mikami2008optimal); [Peyré et al., 2019](../../references.md#ref-peyre2019computational)）。下面给出最优值收敛的启发式推导。

为简化记号，将相应的最优值记为

$$
V_\varepsilon := \mathrm{EOT}_\varepsilon(p_{\mathrm{src}}, p_{\mathrm{tgt}}), \quad
V_0 := \mathrm{OT}(p_{\mathrm{src}}, p_{\mathrm{tgt}})
$$

下述基本上界论证适用于某个 OT 最优耦合相对于 $p_{\mathrm{src}}\otimes p_{\mathrm{tgt}}$ 具有有限相对熵的情况。一般情况下，可以用有限熵耦合近似最优耦合，或更系统地通过 $\Gamma$ 收敛论证，严格建立最优值的收敛性。

**上界。** 根据 $\gamma^*_\varepsilon$ 的最优性，其值 $V_\varepsilon$ 不超过使用方案 $\gamma^*$ 时的代价：

$$
V_\varepsilon \leq \int c  \mathop{}\!\mathrm{d}\gamma^* + \varepsilon \mathcal{D}_{\text{KL}}(\gamma^* \| p_{\mathrm{src}} \otimes p_{\mathrm{tgt}}).
$$

假设 KL 项是有限常数 $K$，则得到 $V_\varepsilon \leq V_0 + \varepsilon K$。取上极限可得 $\limsup_{\varepsilon \to 0} V_\varepsilon \leq V_0$。

**下界。** 由于 KL散度非负，$V_\varepsilon \geq \int c \mathop{}\!\mathrm{d}\gamma^*_\varepsilon$。按照 $V_0$ 作为最小传输代价的定义，任何方案的代价都至少为 $V_0$，因此 $\int c \mathop{}\!\mathrm{d}\gamma^*_\varepsilon \geq V_0$。这意味着 $V_\varepsilon \geq V_0$ 对所有 $\varepsilon > 0$ 成立，从而 $\liminf_{\varepsilon \to 0} V_\varepsilon \geq V_0$。

结合上下界可得最优值的收敛性，即 $\lim_{\varepsilon \to 0} V_\varepsilon = V_0$。相应的最优方案收敛性还需要额外的紧性论证：任何极限方案都是 OT 最优的，而当 OT 最优解唯一时，可以得到整个序列的收敛。

## 7.3.3 SB$_\varepsilon$ 在 $\varepsilon\to 0$ 时化为 OT {#subsec-sb-ot}

为与[式 (7.2.3)](02-transport-problem-setups.md#eq-bb-ot)直接比较，本小节将时间缩放为 $T=1$。对于每个 $\varepsilon > 0$，令 $\mathbf{v}_t^\varepsilon$ 为[式 (7.2.10)](02-transport-problem-setups.md#eq-sb-epsilon-kinetic)中 SB 问题的一个最小化解，$p_t^\varepsilon$ 为受控 SDE $\mathbf{x}_t$ 的边缘分布，该 SDE 由 $\mathbf{v}_t^\varepsilon$ 诱导。那么，$p_t^\varepsilon$ 满足相应的 Fokker–Planck 方程。另一方面，用 $(p_t^0, \mathbf{v}_t^0)$ 表示最优传输 Benamou–Brenier 表述的一个最小化解（见[式 (7.2.3)](02-transport-problem-setups.md#eq-bb-ot)）。

以下定理[^1]指出，当 $\varepsilon \to 0$ 时，SB 问题收敛到 OT 问题。这一结果具有实践意义，原因与[定理 7.3.1](03-relationships-between-formulations.md#thm-eot-ot)类似。目标 $\mathrm{SB}_\varepsilon$ 可以使用 Sinkhorn 类算法高效求解，为最优传输提供数值上可计算且可微的近似。在高维或大规模设定下，直接求解器（如基于 Benamou–Brenier 表述的求解器）计算开销很大，因此这一点尤其有价值。

<span id="thm-sb-ot"></span>

**定理 7.3.2：（非正式）SB$_\varepsilon$ 收敛到 OT。**

当 $\varepsilon \to 0$ 时，有：

$$
\lim_{\varepsilon \to 0}
2\varepsilon\,\mathrm{SB}_\varepsilon
(p_{\mathrm{src}},p_{\mathrm{tgt}})
=
\mathcal{W}_2^2(p_{\mathrm{src}},p_{\mathrm{tgt}}),
$$

其中，OT 采用[式 (7.2.3)](02-transport-problem-setups.md#eq-bb-ot)中的 Benamou–Brenier 表述。此外，在适当的紧性和正则性假设下，边缘分布 $p_t^\varepsilon$ 及相应速度场 $\mathbf{v}_t^\varepsilon$ 沿子序列在适当弱意义下收敛到极限对 $(p_t^0,\mathbf{v}_t^0)$，后者求解 Benamou–Brenier 问题。特别地，当 Benamou–Brenier 最小化解唯一时，最优 Schrödinger 动力学在相应极限意义下收敛到这一 OT 解。

**定理证明。**

这一收敛结果的完整严格证明超出本书范围；详细推导可参见 [Léonard (2012)](../../references.md#ref-leonard2012schrodinger) 和 [Léonard (2014)](../../references.md#ref-leonard2014survey)。不过，我们可以启发式地理解这一收敛为何可能成立。

在 SB 问题的随机控制表述[式 (7.2.10)](02-transport-problem-setups.md#eq-sb-epsilon-kinetic)中，受控 SDE 为：

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = \mathbf{v}_t^\varepsilon({\mathbf{x}}_t)\,\mathop{}\!\mathrm{d} t + \sqrt{\varepsilon}\,\mathop{}\!\mathrm{d} \mathbf{w}_t.
$$

当 $\varepsilon \to 0$ 时，噪声项消失，SDE 在形式上趋近于确定性 ODE：

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = \mathbf{v}_t^0({\mathbf{x}}_t)\,\mathop{}\!\mathrm{d} t.
$$

结合[式 (7.2.10)](02-transport-problem-setups.md#eq-sb-epsilon-kinetic)，这表明经过适当缩放的 SB 目标趋近于 Benamou–Brenier 作用量：

$$
\lim_{\varepsilon \to 0}
2\varepsilon\,\mathrm{SB}_\varepsilon
(p_{\mathrm{src}},p_{\mathrm{tgt}})
=
\mathcal{W}_2^2(p_{\mathrm{src}},p_{\mathrm{tgt}}).
$$

同时，边缘密度 $p_t^\varepsilon$ 满足 Fokker–Planck 方程：

$$
\partial_t p_t^\varepsilon + \nabla \cdot \left(p_t^\varepsilon \mathbf{v}_t^\varepsilon\right) = \frac{\varepsilon}{2}\,\Delta p_t^\varepsilon.
$$

同样，当 $\varepsilon \to 0$ 时，扩散项消失，方程在形式上化为连续性方程：

$$
\partial_t p_t^0 + \nabla \cdot \left(p_t^0 \mathbf{v}_t^0\right) = 0.
$$

至此，我们已经介绍 EOT 与 SB 在各自假设下的基本等价性，以及它们通过极限过程与 OT 建立的重要联系，如[图 7.4](03-relationships-between-formulations.md#fig-ot-dm)所示。接下来将探讨扩散模型与这些概念之间的联系。

[^1]: 严格表述使用经过适当缩放的 Schrödinger 泛函的 $\Gamma$ 收敛性。结合适当的紧性假设，可以得到最优值以及最小化解的相应极限点的收敛。由于这需要更多技术背景，这里略去细节，只陈述概念性结果。

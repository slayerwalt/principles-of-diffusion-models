# 5.1 基于流的模型：归一化流与神经 ODE {#sec-flow-based-method}

本节介绍基于流的模型，包括归一化流（Normalizing Flows，NF）（[Rezende and Mohamed, 2015](../../references.md#ref-rezende2015variational)）与神经常微分方程（Neural Ordinary Differential Equations，NODE）（[Chen et al., 2018](../../references.md#ref-chen2018neural)）。

NF 对简单基分布施加一系列可逆变换，从而进行灵活且可计算的概率密度估计。NODE 将这一框架扩展到连续时间，其中变换由 ODE 支配。通过将变换视为连续时间动力学，NODE 为 NF 范式提供了光滑且可扩展的推广。

<span id="fig-sample-and-change-of-variable"></span>

![可逆归一化流映射下的样本移动与密度变化](../../../en/assets/chapter-05/sample-and-change-of-variable.png)

**图 5.2：** **NF 在可逆映射下的样本移动示意图。** 它由一系列可逆函数 ${\mathbf{f}}: {\mathbf{z}} \mapsto {\mathbf{x}}$ 组成，将潜变量 ${\mathbf{z}}$ 转化为数据 ${\mathbf{x}}$，以及用于重构数据的逆映射 ${\mathbf{f}}^{-1}: {\mathbf{x}} \mapsto {\mathbf{z}}$。NF 类似于编码器与解码器结构，但编码器由光滑可逆映射实现，解码器恰好是其逆映射。相应的密度变化可通过[式 (5.0.1)](index.md#eq-nf-change-of-var)中的变量替换公式计算。

来源：作者绘制。

## 5.1.1 归一化流 {#normalizing-flows}

NF（[Rezende and Mohamed, 2015](../../references.md#ref-rezende2015variational)）建模复杂数据分布 $p_{\text{data}}({\mathbf{x}})$，方法是对简单先验 $p_{\text{prior}}({\mathbf{z}})$（例如标准 Gaussian 分布 $\mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$）施加可逆映射：

$$
{\mathbf{f}}_{\boldsymbol{\phi}}: \mathbb{R}^D \to \mathbb{R}^D,
$$

其中 ${\mathbf{x}} = {\mathbf{f}}_{\boldsymbol{\phi}}({\mathbf{z}})$ 且 ${\mathbf{z}} \sim p_{\text{prior}}$。这里，${\mathbf{x}}$ 与 ${\mathbf{z}}$ 维数相同。利用[式 (5.0.1)](index.md#eq-nf-change-of-var)的变量替换公式，模型似然为[^1]

<span id="eq-nf-change-of-var-log"></span>

$$
\begin{aligned}
\log p_{\boldsymbol{\phi}}({\mathbf{x}}) = \log p_{\text{prior}}({\mathbf{z}}) + \log \left| \det \frac{\partial {\mathbf{f}}_{\boldsymbol{\phi}}^{-1}({\mathbf{x}})}{\partial {\mathbf{x}}} \right|.
\end{aligned}
\tag{5.1.1}
$$

**训练目标。** 参数 $\boldsymbol{\phi}$ 通过最大化数据上的似然来学习：

<span id="eq-nf-training"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{NF}}(\boldsymbol{\phi}) = \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}} \left[ \log p_{\boldsymbol{\phi}}({\mathbf{x}}) \right].
\end{aligned}
\tag{5.1.2}
$$

计算[式 (5.1.1)](01-normalizing-flows-and-neural-odes.md#eq-nf-change-of-var-log)中的 Jacobian 行列式可能成本很高，一般复杂度为 $\mathcal{O}(D^3)$。

**构造可逆变换。**

单个复杂可逆网络可能因计算 Jacobian 行列式而十分昂贵。相反，简单变换（例如线性变换）虽然高效，却缺乏表达能力。

为平衡二者，NF 使用由 $L$ 个可训练可逆映射 $\{{\mathbf{f}}_k\}_{k=0}^{L-1}$ 组成的序列，每个映射的 Jacobian 都能高效计算：

$$
{\mathbf{f}}_{\boldsymbol{\phi}} = {\mathbf{f}}_{L-1} \circ {\mathbf{f}}_{L-2} \circ \cdots \circ {\mathbf{f}}_0.
$$

每个 ${\mathbf{f}}_k$ 都由神经网络参数化，但为简化记号，我们省略对 ${\boldsymbol{\phi}}$ 的显式依赖。

样本按以下方式变换：

<span id="eq-nf-seq"></span>

$$
\begin{aligned}
{\mathbf{x}}_{k+1} = {\mathbf{f}}_k({\mathbf{x}}_k), \quad k = 0, \ldots, L-1,
\end{aligned}
\tag{5.1.3}
$$

其中 ${\mathbf{z}}={\mathbf{x}}_0 \sim p_{\text{prior}}$ 且 ${\mathbf{x}} = {\mathbf{x}}_L$，后者对应数据。所得密度或对数密度推导为

<span id="eq-multi-nf-density"></span>

$$
\begin{aligned}
    p_{\boldsymbol{\phi}}({\mathbf{x}}) &= p_{\text{prior}}({\mathbf{x}}_0) \prod_{k=0}^{L-1} \left| \det \frac{\partial {\mathbf{f}}_k}{\partial {\mathbf{x}}_k} \right|^{-1}, \text{ or equivalently,}\\
\log p_{\boldsymbol{\phi}}({\mathbf{x}}) &= \log p_{\text{prior}}({\mathbf{x}}_0) + \sum_{k=0}^{L-1} \log \left| \det \frac{\partial {\mathbf{f}}_k}{\partial {\mathbf{x}}_k} \right|^{-1}.
\end{aligned}
\tag{5.1.4}
$$

<span id="fig-nf-stack"></span>

![归一化流堆叠可逆映射](../../../en/assets/chapter-05/nf-stack.png)

**图 5.3：** **NF 示意图。** NF 由堆叠的可逆映射 ${\mathbf{f}}_{\boldsymbol{\phi}} = {\mathbf{f}}_{L-1} \circ {\mathbf{f}}_{L-2} \circ \cdots \circ {\mathbf{f}}_0$ 组成。该变换将潜在样本 ${\mathbf{x}}_0 \sim p_{\mathrm{prior}}$ 映射为数据样本 ${\mathbf{x}}_L \sim p_{\mathrm{data}}$。

来源：作者绘制。

**可逆流的示例。** 大量文献关注如何设计单层流，以高效计算 Jacobian。下面介绍两种代表性类型：Planar Flows（[Rezende and Mohamed, 2015](../../references.md#ref-rezende2015variational)）与 Residual Flows（[Chen et al., 2019](../../references.md#ref-chen2019residual)；[Behrmann et al., 2019](../../references.md#ref-behrmann2019invertible)），后者将引出[第 5.1.2 节](01-normalizing-flows-and-neural-odes.md#subsec-node)的发展。

**Planar Flows：** Planar Flows（[Rezende and Mohamed, 2015](../../references.md#ref-rezende2015variational)）应用一个简单变换

$$
{\mathbf{f}}({\mathbf{z}})
=
{\mathbf{z}}
+
{\mathbf{u}} h({\mathbf{w}}^\top{\mathbf{z}}+b),
$$

其中 ${\mathbf{u}},{\mathbf{w}}\in\mathbb{R}^D$、$b\in\mathbb{R}$，且 $h(\cdot)$ 为激活函数。通过适当约束或重参数化参数，可保证变换可逆。其 Jacobian 行列式具有简单形式

$$
\left|
\det\frac{\partial{\mathbf{f}}({\mathbf{z}})}{\partial{\mathbf{z}}}
\right|
=
\left|
1+h'({\mathbf{w}}^\top{\mathbf{z}}+b)\,{\mathbf{w}}^\top{\mathbf{u}}
\right|,
$$

从而使变量替换计算高效。

**Residual Flows：** 将变换 ${\mathbf{f}}$ 定义为

<span id="eq-residual-flow"></span>

$$
\begin{aligned}
{\mathbf{f}}({\mathbf{z}}) = {\mathbf{z}} + {\mathbf{v}}({\mathbf{z}}),
\end{aligned}
\tag{5.1.5}
$$

其中 ${\mathbf{v}}$ 为压缩映射（Lipschitz 常数 $< 1$）。根据 Banach 不动点定理，这保证了可逆性。

Jacobian 的对数行列式可化为迹展开：

<span id="eq-log-det-tr"></span>

$$
\begin{aligned}
\log \Bigg|\det \frac{\partial {\mathbf{f}}({\mathbf{z}})}{\partial {\mathbf{z}}} \Bigg| &= \log \left( \det \frac{\partial {\mathbf{f}}({\mathbf{z}})}{\partial {\mathbf{z}}} \right) \\
    &= \operatorname{Tr}\left(\log\left(\frac{\partial {\mathbf{f}}({\mathbf{z}})}{\partial {\mathbf{z}}}\right)\right) \\
    &= \operatorname{Tr}\left(\log\left({\mathbf{I}} + \frac{\partial {\mathbf{v}}({\mathbf{z}})}{\partial {\mathbf{z}}}\right)\right) \\
    &= \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k} \operatorname{Tr}\left(\left(\frac{\partial {\mathbf{v}}({\mathbf{z}})}{\partial {\mathbf{z}}}\right)^k\right),
\end{aligned}
\tag{5.1.6}
$$

从而可通过迹估计器高效求值（[Hutchinson, 1989](../../references.md#ref-hutchinson1989stochastic)）。

**采样与推断。** 从 NF 采样很直接：抽取 ${\mathbf{x}}_0 \sim p_{\text{prior}}$，再计算 ${\mathbf{x}} = {\mathbf{f}}_{\boldsymbol{\phi}}({\mathbf{x}}_0)$。精确似然由[式 (5.1.4)](01-normalizing-flows-and-neural-odes.md#eq-multi-nf-density)得到。

## 5.1.2 神经 ODE {#subsec-node}

<span id="fig-nf-cnf"></span>

![离散时间与连续时间归一化流](../../../en/assets/chapter-05/nf-cnf.png)

**图 5.4：** **离散时间与连续时间归一化流。** （左）离散 NF 通过有限个可逆映射 $\mathbf{x}_{k+1} = \mathbf{f}_k(\mathbf{x}_k)$ 输运样本，得到逐步、不相交的轨迹（带箭头的点）。（右）连续 NF（神经 ODE）沿 $\frac{\mathop{}\!\mathrm{d}\mathbf{x}(t)}{\mathop{}\!\mathrm{d} t} = \mathbf{v}_{\boldsymbol{\phi}}(\mathbf{x}(t),t)$ 的积分曲线演化状态，图中灰色向量场上展示了带切向箭头的黑色路径。

来源：作者绘制。

**从离散时间 NF 到连续时间 NF（神经 ODE）。** 离散 NF 通过复合 $L$ 个可逆映射输运样本。为得到有意义的连续深度极限，将每层写为小幅残差更新：

$$
{\mathbf{x}}_{k+1}
=
{\mathbf{x}}_k+\Delta t\,{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_k,t_k),
\qquad
t_k=k\Delta t,
\qquad
\Delta t=\frac{T}{L}.
$$

这里，${\mathbf{v}}_{\boldsymbol{\phi}}$ 指定每个点移动的方向，$\Delta t$ 控制一层的步幅。

[图 5.5](01-normalizing-flows-and-neural-odes.md#fig-bijection-continuity)展示这一极限的几何含义。随着 $L$ 增大，同样的源到目标输运被分成更多接近恒等映射的双射。样本轨迹的分辨率提高，周围参考网格的形变也逐渐变得更光滑。

当 $L\rightarrow\infty$ 且 $\Delta t\rightarrow 0$ 时，离散层索引变为连续时间，得到 ODE

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}(t),t).
$$

这个连续深度模型称为神经 ODE 或连续归一化流。

<span id="fig-bijection-continuity"></span>

![离散双射收敛到连续输运](../../../en/assets/chapter-05/bijection-continuity.png)

**图 5.5：** **从离散双射到连续输运流。** 每幅图展示通过复合 $L$ 个可逆层构造的输运。从源分布（Gaussian）出发，各层逐渐将彩色样本移向具有三个模式的目标分布。橙色参考网格展示周围空间在输运中的形变。层数较少时，每层产生相对较大且离散的变化。随着 $L$ 增加，输运被划分为更多、更小的更新，因此样本轨迹与网格形变变得更光滑。若总输运时间 $T$ 固定，且每层步长为 $\Delta t=T/L$，则当 $L\rightarrow\infty$ 时，堆叠的双射趋近于随时间变化的连续流。输运后的密度按[式 (5.2.8)](02-flow-matching-framework.md#eq-continuity-eq)中的连续性方程演化。

来源：作者借助 AI 辅助编程绘制。

**神经 ODE 的形式化设定。** 神经 ODE 通过下式定义连续变换：

<span id="eq-neural-ode"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}(t), t),\quad t\in [0,T]
\end{aligned}
\tag{5.1.7}
$$

其中：

-   ${\mathbf{x}}(t) \in \mathbb{R}^D$ 是时刻 $t$ 的状态；为简洁起见，有时写作 ${\mathbf{x}}_t$；

-   ${\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}(t), t)$ 是由 $\boldsymbol{\phi}$ 参数化的神经网络。

**NODE 的目标。** 从初始条件 ${\mathbf{x}}(0) \sim p_{\text{prior}}$ 出发，ODE 使状态随时间连续演化，诱导出一族边缘分布 $p_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$（与 PF-ODE 类似！）[^2]。

目标是学习神经向量场 ${\mathbf{v}}_{\boldsymbol{\phi}}$，直观上它表示一种速度，沿数据空间中的连续轨迹输运各点。通过学习这一速度，使 $t=T$ 时的终端分布匹配目标分布 $p_{\mathrm{data}}(\cdot)$。这一连续变换将离散归一化流与神经 ODE 统一到同一框架中。

**连续时间变量替换公式。** 类似于[式 (5.0.1)](index.md#eq-nf-change-of-var)或[式 (5.1.4)](01-normalizing-flows-and-neural-odes.md#eq-multi-nf-density)，[Chen et al. (2018)](../../references.md#ref-chen2018neural) 推导了变量替换公式的连续时间对应形式。对于随时间变化的密度 $p_{\boldsymbol{\phi}}({\mathbf{x}}(t),t)$，其对应过程 ${\mathbf{x}}(t)$ 按[式 (5.1.7)](01-normalizing-flows-and-neural-odes.md#eq-neural-ode)演化，所谓的*瞬时变量替换公式*为

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}
\log p_{\boldsymbol{\phi}}({\mathbf{x}}(t),t)
=
-\nabla_{{\mathbf{x}}}\cdot
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}(t),t).
$$

这里的导数沿 ODE 轨迹 ${\mathbf{x}}(t)$ 求取。

因此，若 ${\mathbf{x}}(0)\sim p_{\mathrm{prior}}$，且 ODE 将先验从 $t=0$ 输运到 $t=T$，则对这一恒等式积分得到

<span id="eq-node-likelihood"></span>

$$
\begin{aligned}
\log p_{\boldsymbol{\phi}}({\mathbf{x}}(T),T)
=
\log p_{\mathrm{prior}}({\mathbf{x}}(0))
-
\int_0^T
\nabla_{{\mathbf{x}}}\cdot
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}(t),t)
\,\mathop{}\!\mathrm{d} t.
\end{aligned}
\tag{5.1.8}
$$

这一表达式可通过数值积分计算似然，从而支持模型的最大似然训练。

虽然这个公式乍看可能陌生，但它与 Fokker–Planck 方程密切相关。特别地，它可理解为*连续性方程*沿轨迹的形式，而连续性方程正是 Fokker–Planck 方程在确定性、零扩散情形下的特例（见[附录 B](../../appendices/b-density-evolution/index.md#app-continuity)）。它也可以解释为[式 (5.1.4)](01-normalizing-flows-and-neural-odes.md#eq-multi-nf-density)中离散变量替换公式的连续时间极限。下面总结这一结果。

<span id="instant-change-of-var"></span>

**引理 5.1.1：瞬时变量替换**

令 ${\mathbf{z}}(t)$ 为具有随时间变化密度 $p({\mathbf{z}}(t),t)$ 的连续随机过程，假设它按以下 ODE 演化：

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}{\mathbf{z}}(t)}{\mathop{}\!\mathrm{d} t}
=
\mathbf{F}({\mathbf{z}}(t),t).
\end{aligned}
$$

假设具有足够的正则性，使 ODE 流与演化密度都有良好定义，则沿轨迹的对数密度满足

<span id="eq-instant-change-of-var"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}
\log p({\mathbf{z}}(t),t)
=
-\nabla_{{\mathbf{z}}}\cdot
\mathbf{F}({\mathbf{z}}(t),t).
\end{aligned}
\tag{5.1.9}
$$

**引理证明。**

这由连续性方程与链式法则得到。[第 D.3 节](../../appendices/d-supplementary-proofs/03-flow-based-perspective.md#app-sec-flow-proof)给出了其他详细推导。

**与离散时间公式的联系。** [式 (5.1.8)](01-normalizing-flows-and-neural-odes.md#eq-node-likelihood)中的 NODE 似然

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}(T), T) = \log p_{\text{prior}}({\mathbf{x}}(0), 0) - \int_0^T \nabla_{\mathbf{x}} \cdot {\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}(t), t) \mathop{}\!\mathrm{d} t,
$$

可以视为[式 (5.1.4)](01-normalizing-flows-and-neural-odes.md#eq-multi-nf-density)中离散归一化流形式的连续时间对应物：

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}_L) = \log p_{\text{prior}}({\mathbf{x}}_0) - \sum_{k=0}^{L-1} \log \left| \det \frac{\partial {\mathbf{f}}_k}{\partial {\mathbf{x}}_k} \right|.
$$

积分对应求和，迹算子替换对数行列式，正如[式 (5.1.6)](01-normalizing-flows-and-neural-odes.md#eq-log-det-tr)所述。引理的证明将进一步讨论这些对应关系。

**训练 NODE。** 基于[式 (5.1.8)](01-normalizing-flows-and-neural-odes.md#eq-node-likelihood)，NODE 学习参数化速度场 ${\mathbf{v}}_{\boldsymbol{\phi}}$，使终端分布满足

$$
p_{\boldsymbol{\phi}}(\cdot, T) \approx p_{\text{data}},
$$

其中轨迹从潜变量 ${\mathbf{x}}(0) \sim p_{\text{prior}}$ 出发，通过 ODE 流演化。训练遵循[式 (1.1.2)](../01-deep-generative-modeling/01-what-is-deep-generative-modeling.md#eq-mle)的 MLE 框架：

$$
\mathcal{L}_{\text{NODE}}(\boldsymbol{\phi}) := \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}} \big[\log p_{\boldsymbol{\phi}}({\mathbf{x}}, T)\big].
$$

**精确的对数似然计算。** 为计算 $\log p_{\boldsymbol{\phi}}(\mathbf{x}, T)$，即数据点 $\mathbf{x}$ 的对数似然，对[式 (5.1.8)](01-normalizing-flows-and-neural-odes.md#eq-node-likelihood)的变量替换公式积分：

<span id="eq-node-exact-likelihood"></span>

$$
\begin{aligned}
\log p_{\boldsymbol{\phi}}(\mathbf{x}, T) = \log p_{\text{prior}}(\mathbf{z}(0)) - \int_0^T \nabla_{\mathbf{z}} \cdot \mathbf{v}_{\boldsymbol{\phi}}(\mathbf{z}(t), t) \mathop{}\!\mathrm{d} t.
\end{aligned}
\tag{5.1.10}
$$

这里，$\mathbf{z}(t)$ 反向求解 ODE，从 $t = T$ 到 $t = 0$：

$$
\frac{\mathop{}\!\mathrm{d}\mathbf{z}(t)}{\mathop{}\!\mathrm{d} t} = \mathbf{v}_{\boldsymbol{\phi}}(\mathbf{z}(t), t)
$$

其中 $\mathbf{z}(T) = \mathbf{x}$。对于标准分布，先验项 $\log p_{\text{prior}}(\mathbf{z}(0))$ 可计算。这使神经 ODE 能够进行基于精确似然的训练与评估。

**基于梯度的优化。** 最大化 $\mathcal{L}_{\text{NODE}}$ 需要通过 ODE 求解器反向传播。伴随敏感度方法（[Pontryagin, 2018](../../references.md#ref-pontryagin2018mathematical)；[Chen et al., 2018](../../references.md#ref-chen2018neural)）通过辅助 ODE 计算梯度，内存复杂度为 $\mathcal{O}(1)$；但由于每步都需要数值积分，NODE 的训练成本仍然较高。

**使用 NODE 推断。** 使用训练好的模型 ${\mathbf{v}}_{\boldsymbol{\phi}^\times}$ 进行采样时，先抽取 ${\mathbf{x}}(0) \sim p_{\text{prior}}$，再通过数值求解器正向积分：

$$
{\mathbf{x}}(T) = {\mathbf{x}}(0) + \int_0^T {\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t), t)   \mathop{}\!\mathrm{d} t.
$$

终端状态 ${\mathbf{x}}(T)$ 近似为来自 $p_{\text{data}}$ 的样本。

此外，对任意向量场 ${\mathbf{F}}$，以下恒等式成立：

$$
\operatorname{Tr}\left(\frac{\partial {\mathbf{F}}}{\partial {\mathbf{z}}(t)} \right) = \nabla_{{\mathbf{z}}} \cdot {\mathbf{F}}.
$$

因此，可以使用 Hutchinson 估计器（[Hutchinson, 1989](../../references.md#ref-hutchinson1989stochastic)）等随机迹估计器高效估计散度，无须显式构造完整 Jacobian 矩阵。这使高维场景中的似然计算更加实用。不过，由于迹是从随机样本估计的，所得数值对数似然估计具有随机性，并非对散度的精确求值。

在[图 5.5](01-normalizing-flows-and-neural-odes.md#fig-bijection-continuity)中看到许多小双射如何趋近连续输运流之后，我们现在提出一个实践问题：

<span id="question-5-1-1"></span>

**问题 5.1.1**

如何在训练时无须反复求解 ODE，就能学习它的速度场？

这一问题自然引出流匹配，它通过无须模拟的回归目标直接学习速度场，提供扩散模型基于流的视角。

[^1]: 若进一步约束映射为凸势函数的梯度，即 ${\mathbf{f}}_{\boldsymbol{\phi}} = \nabla \psi_{\boldsymbol{\phi}}$，其中 $\psi_{\boldsymbol{\phi}}$ 为凸函数，则[式 (5.1.1)](01-normalizing-flows-and-neural-odes.md#eq-nf-change-of-var-log)化为[式 (7.2.4)](../07-optimal-transport/02-transport-problem-setups.md#eq-monge-ampere)的 Monge–Ampère 关系。这个 PDE 刻画二次代价下将一个分布变换为另一个分布的最优变换。更多细节见[第 7 章](../07-optimal-transport/index.md#ch-ot-eot)与（[Huang et al., 2021](../../references.md#ref-huangconvex)）。

[^2]: 我们采用翻转后的时间约定，$t = 0$ 表示先验（源）分布，$t = 1$ 表示数据（目标）分布。先验交替写作 $p_{\boldsymbol{\phi}}({\mathbf{x}}(0), 0)$、$p_{\text{prior}}({\mathbf{x}}(0), 0)$，或简写为 $p_{\text{prior}}({\mathbf{z}})$。
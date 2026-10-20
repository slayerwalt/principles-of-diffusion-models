# 12.2 超越连续状态：离散空间中的概率演化 {#sec-epilogue-discrete}

到目前为止，本书中的所有模型都采用一个共同的结构假设：生成动力学作用的状态位于连续 Euclidean 空间 ${\mathbb{R}}^D$ 中。无论 ${\mathbf{x}}$ 的各分量表示像素强度、音频采样值还是连续潜变量，数学形式都相同：粒子沿 ODE 或 SDE 支配的连续轨迹运动，概率密度则通过连续性方程或 Fokker–Planck 方程演化。这一框架依赖 ${\mathbb{R}}^D$ 的微分结构：梯度和散度通过导数定义，确定性动力学可以由向量场生成的可微轨迹描述。

这一划分针对建模状态，而非原始模态。对于离散观测，也可以先选择连续编码和读出映射来处理。对于原始符号 $a$，令 ${\mathbf{E}}(a)\in{\mathbb{R}}^d$ 为其编码，并令 ${\mathbf{D}}$ 将编码映射回符号，或更一般地映射到一个类别分布：

$$
\text{raw discrete symbol } a
\;\xrightarrow{\;{\mathbf{E}}\;}\;
\text{continuous code } {\mathbf{E}}(a)\in{\mathbb{R}}^d
\;\xrightarrow{\;{\mathbf{D}}\;}\;
\text{discrete readout}.
$$

表示可以固定，例如独热编码、类别向量或取值于单纯形的编码（[Richemond et al., 2022](../../references.md#ref-richemond2022categorical)）；对于单射码本，读出可以是确定性的，并可能满足 ${\mathbf{D}}({\mathbf{E}}(a))=a$。表示也可以通过学习获得，例如 token 嵌入或自编码器潜变量（[Li et al., 2022](../../references.md#ref-li2022diffusionlm)；[Dieleman et al., 2022](../../references.md#ref-dieleman2022continuous)）；此时 ${\mathbf{D}}$ 是学得的读出映射，${\mathbf{D}}\circ{\mathbf{E}}$ 未必精确可逆。一旦表示固定，扩散便属于连续状态情形：在环境编码空间中进行 Gaussian 加噪，或在受约束的编码空间中采用适应其几何的连续过程，都可用本书前述连续方法处理。下文研究的有限状态路线有所不同：演化中的状态本身始终保持离散。

本节研究另一种情形：状态本身在整个前向与逆向过程中始终有限且离散。许多数据类型天然适合这一视角：文本是词表中 token 构成的序列，分子图使用有限种原子和化学键类型，蛋白质序列则是有限字母表上的字符串。在这样的有限状态空间中，状态之间没有标准的无穷小位移，没有穿过 token 取值的光滑轨迹，也没有关于状态本身的普通梯度。这引出了一个自然问题：

<span id="question-12-2-1"></span>

**问题 12.2.1**

如果扩散关乎概率质量的输运，那么当状态空间有限时，这一思想会变成什么？

结构基础保持不变，但数学对象发生变化。在连续空间中，概率分布在 ODE／SDE 动力学作用下，通过连续性方程或 Fokker–Planck 方程演化。在有限状态空间中，概率质量按照 Markov 转移核或跳跃率在状态之间移动，分布则遵循有限维守恒律演化，即*主方程*，也称 Kolmogorov 前向方程。在抽象层面，两种设定遵循同一流程：

**观察**

选择前向扰动过程，理解其诱导的概率路径，并学习逆向输运机制。

在连续 Gaussian 扩散中，逆向机制表现为 Gaussian 逆向条件分布、score 或速度场。在有限状态空间中，它表现为逆向转移概率、概率比或跳跃率。这些数学对象不同，却承担相同的概念角色。

本章旨在建立一条可随文献发展而保持稳定的认识路线，而非将某一种离散扩散算法视为定论。这条路线按描述逆向输运所用的对象组织：转移核、概率比或跳跃率。我们分四步展开：离散时间转移（[第 12.2.1 节](02-discrete-spaces.md#subsec-epilogue-dtmc)）、其连续时间极限（[第 12.2.2 节](02-discrete-spaces.md#subsec-epilogue-ctmc)）、与连续情形的变分、score／比值和流视角相呼应的三种逆向建模视角（[第 12.2.3 节](02-discrete-spaces.md#subsec-epilogue-three-views)），以及突出与连续设定实质差异的比较（[第 12.2.4 节](02-discrete-spaces.md#subsec-epilogue-genuine-diff)）。

## 12.2.1 离散时间转移 {#subsec-epilogue-dtmc}

考虑具有 $K$ 个离散类别的前向过程，其中 $K$ 包括所选前向过程需要的全部特殊扰动 token，例如掩码 token。每个状态表示为*独热*列向量 ${\mathbf{x}}\in\{0,1\}^{K}$，恰有一个分量等于 $1$；所有此类向量组成的集合为[^1]

$$
\mathcal{V}
= \bigl\{{\mathbf{x}}\in\{0,1\}^{K}:\textstyle\sum_{i=1}^{K}x_i=1\bigr\}
= \{{\mathbf{e}}_1,\ldots,{\mathbf{e}}_K\}.
$$

用 ${\mathbf{e}}_i$ 表示第 $i$ 个标准基向量，即类别 $i$ 的独热编码。例如，每个 ${\mathbf{e}}_i$ 可以表示一个 token、离散潜编码、原子类型、化学键类型或氨基酸。

对于文本或蛋白质等序列数据，一个样本由 $N$ 个 token 组成，每个 token 在大小为 $K$ 的词表中取值。完整状态空间为 $\mathcal{V}^N$，其规模随序列长度呈指数增长。在许多离散扩散模型中，*前向*扰动在各位置独立作用，因此单 token 的 Markov 过程是基本的局部构件。我们先聚焦一个 token，以避免繁重记号，同时揭示概率输运的核心机制。对于完整序列，同样的公式适用于完整配置。如果前向扰动在各位置独立作用，条件前向核就逐位置分解。[^2] 但学得的逆向模型不必在位置之间独立：Transformer 或图神经网络可以观察整个扰动后的对象，利用全局上下文预测局部或全局逆向转移。

<span id="fig-notation-overview"></span>

![序列中的 token 对应于有限词表中的独热状态](../../../en/assets/chapter-12/notation-overview.png)

**图 12.2：** **从序列到独热状态。** 文本或蛋白质序列等数据样本由 $N$ 个 token 组成，每个 token 在大小为 $K$ 的有限字母表或词表中取值。本章聚焦单个 token 在 $K$ 个状态之间的演化。放大观察一个位置，这里为“cat”，该 token 由基向量 ${\mathbf{e}}_2$ 表示，并编码为 $\{0,1\}^K$ 中的独热向量，即一个 $K$ 维向量，其第二个分量为 $1$，其余所有分量均为 $0$。状态空间

$$
\mathcal{V}=\{{\mathbf{x}}\in\{0,1\}^K:\sum_{i=1}^K x_i=1\} =\{{\mathbf{e}}_1,\ldots,{\mathbf{e}}_K\}
$$

是所有此类独热向量的集合。词表条目的顺序可以任意选定，与 token 在序列中的位置无关。

来源：作者借助 AI 辅助编程制作。

在步骤 $k$，系统处于这 $K$ 个状态之一。用以下记号表示处于状态 ${\mathbf{e}}_i$ 的标量概率：

$$
p_k(i) := \Pr({\mathbf{x}}_k={\mathbf{e}}_i),
\qquad i=1,\ldots,K.
$$

将这些标量概率汇集起来，得到边缘概率向量

$$
{\mathbf{p}}_k
:=
\bigl(p_k(1),\;\dots,\;p_k(K)\bigr)^\top
=
{\mathbb{E}}[{\mathbf{x}}_k]
\in \Delta^{K-1}\subset{\mathbb{R}}^K,
\qquad
\sum_{i=1}^K p_k(i)=1,
$$

其中 $\Delta^{K-1}:=\bigl\{{\mathbf{p}}\in{\mathbb{R}}^K:p_i\geq0,\; \sum_{i=1}^K p_i=1\bigr\}$ 表示概率单纯形。等价地，$\Pr({\mathbf{x}}_k={\mathbf{e}}_i)=p_k(i)$，或简记为 ${\mathbf{x}}_k\sim\mathrm{Cat}({\mathbf{p}}_k)$。

因此，${\mathbf{p}}_k$ 是离散随机状态 ${\mathbf{x}}_k$ 的完整分布，而 $p_k(i)$ 是它的第 $i$ 个标量分量。连续时间下采用相同约定：

$$
p_t(i):=\Pr({\mathbf{x}}_t={\mathbf{e}}_i),
\qquad
{\mathbf{p}}_t:=\bigl(p_t(1),\ldots,p_t(K)\bigr)^\top.
$$

向量 ${\mathbf{p}}_t$ 是密度的离散对应物：它描述总概率质量如何分布在 $K$ 个状态上。

沿用本书的记号约定，小写粗体字母（${\mathbf{p}}$、${\mathbf{x}}$）表示向量，大写粗体字母（${\mathbf{P}}$、${\mathbf{Q}}$）表示矩阵，带下标的非粗体量（例如 $p_k(i)$、$[{\mathbf{P}}_k]_{ij}$ 和 $[{\mathbf{Q}}_t]_{ij}$）表示标量元素。以随机变量为自变量的非粗体 $p$，例如 $p({\mathbf{x}}_{k-1}|{\mathbf{x}}_k)$，表示固定前向过程下的概率质量函数或转移核；$p_{{\boldsymbol{\phi}}}$ 表示学得的逆向模型。

系统如何在状态之间移动？单次转移由*转移矩阵* ${\mathbf{P}}_k \in {\mathbb{R}}^{K \times K}$ 指定。元素 $[{\mathbf{P}}_k]_{ij}$ 给出系统当前处于状态 ${\mathbf{e}}_i$ 时，一步后移到状态 ${\mathbf{e}}_j$ 的概率：

$$
[{\mathbf{P}}_k]_{ij} \;=\; \Pr({\mathbf{x}}_{k+1} = {\mathbf{e}}_j | {\mathbf{x}}_k = {\mathbf{e}}_i).
$$

因此，整个矩阵具有如下结构：

$$
{\mathbf{P}}_k \;=\;
\underbrace{
\begin{pmatrix}
[{\mathbf{P}}_k]_{11} & [{\mathbf{P}}_k]_{12} & \cdots & [{\mathbf{P}}_k]_{1K} \\
[{\mathbf{P}}_k]_{21} & [{\mathbf{P}}_k]_{22} & \cdots & [{\mathbf{P}}_k]_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
[{\mathbf{P}}_k]_{K1} & [{\mathbf{P}}_k]_{K2} & \cdots & [{\mathbf{P}}_k]_{KK}
\end{pmatrix}
}_{\substack{\text{columns: next state } j}}
\quad
\left.
\vphantom{
\begin{pmatrix}
a \\ a \\ \vdots \\ a
\end{pmatrix}
}
\right\}
\text{ rows: current state } i.
$$

沿第 $i$ 行阅读，可以回答：“给定系统当前处于状态 ${\mathbf{e}}_i$，它到达各个可能下一状态的概率是多少？”因此，每一行都是关于目的状态的概率分布：

$$
[{\mathbf{P}}_k]_{ij} \geq 0
\quad \text{for all } i,j,
\qquad\qquad
\sum_{j=1}^{K} [{\mathbf{P}}_k]_{ij} = 1
\quad \text{for each } i.
$$

非负条件保证每个转移概率有效，行和约束则保证离开状态 ${\mathbf{e}}_i$ 的全部质量都被计入：系统总会落在某个状态。

**分布如何演化。** 有了转移矩阵，就可以提出一个基本问题：如果系统从分布 ${\mathbf{p}}_k$ 出发，一步后的分布是什么？答案来自简单的概率核算。转移后处于状态 ${\mathbf{e}}_j$ 的概率，是从所有可能源状态到达 ${\mathbf{e}}_j$ 的质量总和：

<span id="eq-discrete-transport-one-step"></span>

$$
p_{k+1}(j) = \sum_{i=1}^K p_k(i)\,[{\mathbf{P}}_k]_{ij}.
\tag{12.2.1}
$$

求和中的每一项，都是状态 ${\mathbf{e}}_i$ 最初拥有的质量，乘以从 ${\mathbf{e}}_i$ 跳到 ${\mathbf{e}}_j$ 的概率。写成向量形式为

$$
{\mathbf{p}}_{k+1} = {\mathbf{P}}_k^\top {\mathbf{p}}_k.
$$

等价地，给定当前独热状态 ${\mathbf{x}}_k$ 时，下一状态的条件分布为

$$
\Pr({\mathbf{x}}_{k+1} | {\mathbf{x}}_k)
= \mathrm{Cat}({\mathbf{x}}_{k+1};\,{\mathbf{P}}_k^\top\,{\mathbf{x}}_k),
$$

因为 ${\mathbf{P}}_k^\top\,{\mathbf{e}}_i$ 提取第 $i$ 行，所取矩阵为 ${\mathbf{P}}_k$，该行包含从状态 ${\mathbf{e}}_i$ 出发的转移概率。

按这一规则演化的系统称为*离散时间 Markov 链*：以当前状态 ${\mathbf{x}}_k$ 为条件，${\mathbf{x}}_{k+1}$ 的分布不依赖更早的状态。它在有限状态中承担连续空间分布演化公式的角色。在连续空间中，双射 ${\boldsymbol{\Psi}}$ 重塑密度，Jacobian 行列式补偿局部体积变化。在有限状态空间中，没有需要跟踪的体积元；转移矩阵 ${\mathbf{P}}_k$ 直接指定多少概率质量离开每个源状态，并到达每个目的状态。行和约束保证总概率守恒，正如连续性方程中的散度结构所起的作用。

使用转移矩阵 ${\mathbf{P}}_0, \dots, {\mathbf{P}}_{L-1}$ 重复多个步骤，得到

$$
{\mathbf{p}}_L = {\mathbf{P}}_{L-1}^\top \cdots {\mathbf{P}}_1^\top {\mathbf{P}}_0^\top\, {\mathbf{p}}_0.
$$

这对应于[公式 (B.1.2)](../../appendices/b-density-evolution/01-change-of-variable.md#eq-cov-multi-layer-nolog)中的多层变量变换公式。在连续确定性设定中，密度演化通过 Jacobian 行列式累积；这里，概率向量通过连续的 Markov 算子传播。底层思想相同：跟踪概率质量通过一系列给定变换的过程。

**逆转单个步骤。** 给定前向过程，自然会问如何逆转一次转移以进行生成。假设观察到系统处于状态 ${\mathbf{e}}_j$，对应步骤 $k{+}1$，那么它之前处于状态 ${\mathbf{e}}_i$、对应前一步 $k$ 的概率是多少？这可直接应用 Bayes 法则：

<span id="eq-discrete-reverse-kernel"></span>

$$
\begin{aligned}
\Pr({\mathbf{x}}_k = {\mathbf{e}}_i | {\mathbf{x}}_{k+1} = {\mathbf{e}}_j)
= \frac{\Pr({\mathbf{x}}_{k+1} = {\mathbf{e}}_j | {\mathbf{x}}_k = {\mathbf{e}}_i)\;\Pr({\mathbf{x}}_k = {\mathbf{e}}_i)}
        {\Pr({\mathbf{x}}_{k+1} = {\mathbf{e}}_j)}
= [{\mathbf{P}}_k]_{ij}\frac{p_k(i)}{p_{k+1}(j)},
\end{aligned}
\tag{12.2.2}
$$

只要 $p_{k+1}(j)>0$ 成立。这是由前向转移 ${\mathbf{P}}_k$ 与当前边缘分布共同确定的离散时间逆向核。其形式很直观：在质量可能到达状态 ${\mathbf{e}}_j$ 的所有方式中，各源状态 ${\mathbf{e}}_i$ 的贡献，与该状态原有的质量（通过 $p_k(i)$ 体现）及相应跳跃的可能性（通过 $[{\mathbf{P}}_k]_{ij}$ 体现）成正比。[公式 (12.2.2)](02-discrete-spaces.md#eq-discrete-reverse-kernel)的分子与分母各有明确解释：

-   **分子：** $[{\mathbf{P}}_k]_{ij}\,p_k(i)$ 是以下事件的联合概率：处于状态 ${\mathbf{e}}_i$（对应步骤 $k$），随后跳到 ${\mathbf{e}}_j$（对应步骤 $k{+}1$）。

-   **分母：** 项

    $$
    p_{k+1}(j) = \sum_{i'} p_k(i')\,[{\mathbf{P}}_k]_{i'j}
    $$

    是从任意源状态到达状态 ${\mathbf{e}}_j$ 的总概率。它充当归一化常数，确保逆向概率对 $i$ 求和为一。

关键在于，逆向核*并非* ${\mathbf{P}}_k$ 的矩阵逆。Markov 转移可能将多个状态的概率质量合并到一个状态，作为线性映射未必可逆。概率意义下的逆向过程是 Bayes 逆，它不仅依赖前向转移矩阵，也依赖边缘分布 ${\mathbf{p}}_k$。这恰好是生成建模面临的困难：${\mathbf{p}}_k$ 由未知数据分布经前向过程推前得到，通常没有可用的闭式表达式。下文的逆向建模视角，可理解为表示或学习实现这一 Bayes 逆所需缺失信息的不同方式。

## 12.2.2 连续时间极限：主方程 {#subsec-epilogue-ctmc}

在连续空间的讨论中，将复合双射的步长缩至零，便得到连续性方程（[第 B.1.2 节](../../appendices/b-density-evolution/01-change-of-variable.md#subsec-cov-continuity-eq)）。同样的极限思想也适用于这里。当时间步变得无穷小时，离散时间 Markov 链变为连续时间 Markov 链（CTMC），分布按照主方程演化。

<span id="fig-ctmc-trajectory"></span>

![连续时间 Markov 链的样本路径在有限状态之间跳跃](../../../en/assets/chapter-12/ctmc-trajectory.png)

**图 12.3：** **CTMC 样本路径示意图。** 设状态空间为 $\mathcal{V}=\{{\mathbf{e}}_1,{\mathbf{e}}_2,{\mathbf{e}}_3\}$，在纵轴上标为 $S_1,S_2,S_3$。轨迹分段常数：过程在 ${\mathbf{x}}_t={\mathbf{e}}_i$ 停留一段随机时间，随后瞬间跳到另一状态 ${\mathbf{e}}_j$，跳跃时刻为 $t_1,t_2,\ldots$。实心圆标出每次跳跃后立即占据的状态，空心圆标出跳跃前的状态，体现右连续约定。点线段表示超出图示窗口后的延续。跳跃时刻和目的状态由速率矩阵 ${\mathbf{Q}}_t$ 支配：从状态 ${\mathbf{e}}_i$ 出发，过程跳到 ${\mathbf{e}}_j$（$j\neq i$）的瞬时速率为 $[{\mathbf{Q}}_t]_{ij}$。

来源：改编自 [Campbell et al. (2022)](../../references.md#ref-campbell2022continuous)。

**无穷小转移。** 上一小节中，每个转移矩阵 ${\mathbf{P}}_k$ 描述从时刻 $k$ 到时刻 $k{+}1$ 的完整一步。现在设想不断细化这些步骤。令 ${\mathbf{P}}_{t,\Delta t}$ 表示使系统向前推进一小段时长 $\Delta t$ 的转移矩阵，起始时刻为 $t$：其元素 $[{\mathbf{P}}_{t,\Delta t}]_{ij}$ 是从状态 ${\mathbf{e}}_i$ 移到状态 ${\mathbf{e}}_j$ 的概率，对应区间 $[t,\,t{+}\Delta t]$。

当 $\Delta t$ 很小时，大多数质量留在原处，只有一小部分发生跳跃。这意味着 ${\mathbf{P}}_{t,\Delta t}$ 接近单位矩阵，可展开为

<span id="eq-generator-expansion"></span>

$$
{\mathbf{P}}_{t,\Delta t}
\;=\; {\mathbf{I}} + {\mathbf{Q}}_t\,\Delta t + \mathcal{O}(\Delta t^2),
\tag{12.2.3}
$$

其中 ${\mathbf{Q}}_t \in {\mathbb{R}}^{K \times K}$ 是时刻 $t$ 的*速率矩阵*，也称生成元。若 ${\mathbf{Q}}_t$ 关于时间具有足够的正则性，余项在局部可写为 $\mathcal{O}(\Delta t^2)$。${\mathbf{Q}}_t$ 的元素是瞬时*速率*，而非概率。具体而言：

-   $[{\mathbf{Q}}_t]_{ij} \geq 0$，其中 $i \neq j$：这是过程从状态 ${\mathbf{e}}_i$ 跳到状态 ${\mathbf{e}}_j$ 的速率。乘以 $\Delta t$，得到短区间内跳跃概率的主导项。

-   $[{\mathbf{Q}}_t]_{ii} = -\sum_{j \neq i} [{\mathbf{Q}}_t]_{ij}$：对角元素为负，记录概率*离开*状态 ${\mathbf{e}}_i$ 的总速率。这确保 ${\mathbf{Q}}_t$ 的每一行之和为零，是概率守恒在速率层面的表达。[^3]

根据展开式，在短区间内从 ${\mathbf{e}}_i$ 跳到 ${\mathbf{e}}_j$（其中 $j\neq i$）的概率近似为 $[{\mathbf{Q}}_t]_{ij}\,\Delta t$，留在 ${\mathbf{e}}_i$ 的概率近似为 $1 - \sum_{j \neq i} [{\mathbf{Q}}_t]_{ij}\,\Delta t$。

**主方程。** 有了无穷小转移，就能直接得到连续时间极限。将[公式 (12.2.3)](02-discrete-spaces.md#eq-generator-expansion)的展开代入 ${\mathbf{p}}_{t+\Delta t} = {\mathbf{P}}_{t,\Delta t}^\top\,{\mathbf{p}}_t$，并取 $\Delta t \to 0$，得到

<span id="eq-master-equation-vector"></span>

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\,{\mathbf{p}}_t = {\mathbf{Q}}_t^\top\,{\mathbf{p}}_t.
\tag{12.2.4}
$$

这就是*主方程*，也称 CTMC 的 *Kolmogorov 前向方程*（[Norris, 1998](../../references.md#ref-norris1998markov)）。回顾 ${\mathbf{p}}_t$ 是概率向量，其第 $j$ 个分量 $p_t(j)$ 给出处于状态 ${\mathbf{e}}_j$ 的概率，对应时刻 $t$。写出[公式 (12.2.4)](02-discrete-spaces.md#eq-master-equation-vector)的第 $j$ 个分量，便可明确其含义：

<span id="eq-master-equation-component"></span>

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\,p_t(j)
= \underbrace{\sum_{i \neq j} p_t(i)\,[{\mathbf{Q}}_t]_{ij}}_{\text{inflow: mass arriving from other states}}
\;-\;
\underbrace{p_t(j)\sum_{\ell \neq j} [{\mathbf{Q}}_t]_{j\ell}}_{\text{outflow: mass departing to other states}}.
\tag{12.2.5}
$$

$p_t(j)$ 的变化率就是流入减去流出。这是用微分方程表达的概率守恒。

**与连续性方程的比较。** 在连续空间中，连续性方程

$$
\partial_t\,p_t({\mathbf{x}}) + \nabla \cdot
\big(p_t({\mathbf{x}})\,{\mathbf{v}}({\mathbf{x}},t)\big) = 0
$$

说明密度变化由速度场 ${\mathbf{v}}$ 下概率质量的净通量驱动。主方程用有限状态语言表达同一内容：$p_t(j)$ 的变化来自速率矩阵 ${\mathbf{Q}}_t$ 支配的净概率通量。速度场与速率矩阵承担类似的结构角色：两者都规定概率质量如何重新分配，也都施加守恒。主要差别在几何层面：在 ${\mathbb{R}}^D$ 中，质量通过相邻空间位置流动；在有限状态空间中，质量沿转移图允许的边跳跃。

<span id="fig-mask-uniform"></span>

![掩码扩散与均匀离散扩散的前向扰动和逆向生成](../../../en/assets/chapter-12/masked-uniform-diffusion.png)

**图 12.4：** **掩码扩散与均匀扩散的前向扰动和逆向生成示意图。** 这里的两个例子都有形式为 $\Pr({\mathbf{x}}_t|{\mathbf{x}}_0)=\mathrm{Cat}({\mathbf{x}}_t;\alpha_t{\mathbf{x}}_0+(1-\alpha_t)\boldsymbol{\pi})$ 的插值边缘分布，其中极限先验 $\boldsymbol{\pi}$ 与衰减因子 $\alpha_t$ 由所选前向速率矩阵确定。在掩码扩散中，token 被吸收到掩码状态，极限先验为 $\boldsymbol{\pi}={\mathbf{m}}={\mathbf{e}}_K$。在均匀扩散中，token 被反复随机化，极限先验为 $\boldsymbol{\pi}_{\mathrm{unif}}=\frac{1}{K}\mathbf{1}$。生成从相应的简单参考分布或与其接近的有限时间近似出发，逆转所选扰动过程。

来源：作者借助 AI 辅助编程制作。

**前向扰动。** 与连续情形一样，我们需要一个前向过程，逐渐破坏数据分布的结构，并将它推向易于采样的简单参考分布。这对应于连续扩散中选择 Gaussian 加噪过程。设计目标相同：扰动应循序渐进，能够解析计算或在计算上可行，并且足够强，使终止时刻 $T$ 时数据分布的原有结构已基本被抹去。

速率矩阵有多种选择。它可以是均匀的、吸收式的、最近邻的、基于图的、带有化学结构的、感知语法的，或以其他方式适应数据领域。以下两个例子并非穷尽所有类别，而是有助于清楚呈现机制的标准情形。

**掩码扩散（吸收式）。** 将最后一个类别保留为掩码 token，记为 ${\mathbf{m}} := {\mathbf{e}}_K$。干净数据 ${\mathbf{x}}_0$ 占据前 $K{-}1$ 个类别之一，因此 $\langle{\mathbf{x}}_0,{\mathbf{m}}\rangle = 0$。定义速率矩阵，使每个普通状态都以速率 $\beta(t)$ 转移到掩码状态，而掩码状态永不转出：

$$
[{\mathbf{Q}}_t]_{iK}=\beta(t),\qquad
[{\mathbf{Q}}_t]_{ii}=-\beta(t),\qquad
[{\mathbf{Q}}_t]_{ij}=0
\quad
\text{for } i<K,\; j\notin\{i,K\},
$$

以及

$$
[{\mathbf{Q}}_t]_{Kj}=0
\quad
\text{for all } j.
$$

其思想很简单：每个 token 独立地以速率 $\beta(t)$ 被替换为掩码，一旦被掩码覆盖就保持不变。

以干净状态 ${\mathbf{x}}_0$ 为条件的前向边缘分布为

$$
\Pr({\mathbf{x}}_t | {\mathbf{x}}_0)
      = \mathrm{Cat}\bigl({\mathbf{x}}_t;\;
          \alpha_t\,{\mathbf{x}}_0 + (1-\alpha_t)\,{\mathbf{m}}\bigr),
$$

其中

$$
\alpha_t
      =
      \exp\left(-\int_0^t\beta(s)\,\mathop{}\!\mathrm{d} s\right).
$$

在时刻 $t$，token 等于其干净值 ${\mathbf{x}}_0$ 的概率为 $\alpha_t$，被掩码覆盖的概率为 $1-\alpha_t$。随着时间推移，越来越多的概率质量积聚在掩码状态上。若 $\int_0^T\beta(s)\mathop{}\!\mathrm{d} s$ 很大，则 $\alpha_T$ 很小，终止分布就接近全掩码参考分布。

**均匀扩散。** 这一过程逐渐随机化 token，而非将其擦除。一种方便的连续时间归一化方式是，让 token 以相同速率跳到其余 $K-1$ 个状态中的每一个：

$$
[{\mathbf{Q}}_t]_{ij} = \frac{\beta(t)}{K - 1}
      \quad\text{for } i \neq j,
      \qquad
      [{\mathbf{Q}}_t]_{ii} = -\beta(t).
$$

第一个公式说明，从状态 ${\mathbf{e}}_i$ 出发，token 以相同瞬时速率跳到每个其他状态。第二个公式说明，离开状态 ${\mathbf{e}}_i$ 的总速率为 $\beta(t)$。

前向边缘分布再次具有插值形式

$$
\Pr({\mathbf{x}}_t|{\mathbf{x}}_0)
      = \mathrm{Cat} \bigl({\mathbf{x}}_t;\;
          \alpha_t\,{\mathbf{x}}_0 + (1-\alpha_t)\,\boldsymbol{\pi}_{\mathrm{unif}}
        \bigr),
$$

其中 $\boldsymbol{\pi}_{\mathrm{unif}}=\tfrac{1}{K}\mathbf{1}\in{\mathbb{R}}^K$ 是均匀概率向量。在上述速率归一化下，衰减因子为

$$
\alpha_t
      =
      \exp\left(
      -\frac{K}{K-1}\int_0^t \beta(s)\,\mathop{}\!\mathrm{d} s
      \right).
$$

随着时间推移，反复随机化会抹去数据分布的结构，并将其推向 $\mathcal{V}$ 上的均匀分布。

更一般地，前向过程不必具有简单的混合形式

$$
\Pr({\mathbf{x}}_t|{\mathbf{x}}_0)
= \mathrm{Cat}\bigl({\mathbf{x}}_t;\;\alpha_t{\mathbf{x}}_0+(1-\alpha_t)\boldsymbol{\pi}\bigr).
$$

这一形式有助于直观理解，并出现在常见的掩码与均匀构造中，但 Markov 分布演化视角更为一般：转移矩阵支配离散时间演化，速率矩阵则通过主方程支配连续时间演化。结构化离散扩散模型可以使用遵循编辑距离、图邻接关系、分子有效性、词表结构或其他领域知识的转移规则。无论哪种情形，都遵循同一原则：前向 Markov 过程定义概率路径，生成则学习沿该路径逆向移动。

## 12.2.3 逆向建模的三个视角 {#subsec-epilogue-three-views}

一旦建立前向过程及其分布演化，剩余任务就与连续情形相同：学习一个逆向过程，使其边缘分布按逆时间重走前向演化，将简单参考样本变为数据。在 B 部分中，我们将连续扩散模型按三个视角组织：变分（[第 2 章](../02-variational-perspective/index.md#ch-variational)）、基于 score（[第 3 章](../03-score-based-perspective/index.md#ch-score-based)和[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)），以及基于流（[第 5 章](../05-flow-based-perspective/index.md#ch-flow-based)）。同样的三分视角也适用于离散设定，但需要说明：它们并非相互排斥的流派或固定的论文分类，而是辨认逆转概率路径所需学习的未知对象的三种方式。

**变分视角：学习逆向转移核。** 在连续情形中（[第 2.2 节](../02-variational-perspective/02-ddpm.md#sec-ddpm)），变分视角将扩散视为层次潜变量模型。前向过程定义一条噪声逐步增大的潜变量链，目标是学习逆向转移，逐步撤销扰动。真实逆向核 $\Pr({\mathbf{x}}_{k-1}|{\mathbf{x}}_k)$ 不可直接计算，因为它依赖 ${\mathbf{x}}_k$ 的边缘分布，而该分布涉及未知的数据分布。DDPM 的关键洞见是以干净样本 ${\mathbf{x}}_0$ 为条件：前向后验 $\Pr({\mathbf{x}}_{k-1}|{\mathbf{x}}_k, {\mathbf{x}}_0)$ 是具有闭式表达式的 Gaussian 分布；[定理 2.2.1](../02-variational-perspective/02-ddpm.md#thm-equiv-marginal-kl)表明，针对这一条件后验训练，与针对不可直接计算的边缘逆向核训练等价，仅相差与学得的逆向模型无关的项。ELBO 随后分解为 KL散度项之和（[公式 (2.2.6)](../02-variational-perspective/02-ddpm.md#eq-ddpm-diffusion)），每个噪声水平对应一项，每一项都要求学得的模型匹配可计算的前向后验。

同样的逻辑也适用于离散设定。用 $p$ 表示从数据分布抽取 ${\mathbf{x}}_0$、再应用固定前向链所得的联合分布，用 $p_{{\boldsymbol{\phi}}}$ 表示学得的逆向模型。考虑一条扰动后的独热状态链

$$
{\mathbf{x}}_0 \;\xrightarrow{\;{\mathbf{P}}_0\;}\; {\mathbf{x}}_1 \;\xrightarrow{\;{\mathbf{P}}_1\;}\;
\cdots \;\xrightarrow{\;{\mathbf{P}}_{L-1}\;}\; {\mathbf{x}}_L,
$$

其中 ${\mathbf{x}}_0$ 从数据分布抽取，而 ${\mathbf{x}}_L$ 接近全掩码或均匀分布等简单参考分布。此时转移是类别分布而非 Gaussian 分布，但变分结构相同：

-   真实逆向核 $p({\mathbf{x}}_{k-1}|{\mathbf{x}}_k)$ 不可直接计算，因为它依赖步骤 $k$ 的边缘分布。

-   以 ${\mathbf{x}}_0$ 为条件后，前向后验可通过 Bayes 法则计算：

    $$
    p({\mathbf{x}}_{k-1}|{\mathbf{x}}_k, {\mathbf{x}}_0)
          =
          \frac{p({\mathbf{x}}_k|{\mathbf{x}}_{k-1})\;p({\mathbf{x}}_{k-1}|{\mathbf{x}}_0)}
                 {p({\mathbf{x}}_k|{\mathbf{x}}_0)},
    $$

    其中右侧的每个因子都由前向过程确定。

-   针对这些条件后验训练，与针对不可直接计算的边缘逆向核训练具有相同的最优逆向核；二者之差由与 $p_{{\boldsymbol{\phi}}}$ 无关的项组成。

概括地说，离散扩散的 ELBO 包含以下形式的去噪项：

$$
{\mathbb{E}}_{p({\mathbf{x}}_{0:L})}
 \Big[
\mathcal{D}_{\mathrm{KL}} \big(
  p({\mathbf{x}}_{k-1}|{\mathbf{x}}_k, {\mathbf{x}}_0)
  \;\|\;
  p_{{\boldsymbol{\phi}}}({\mathbf{x}}_{k-1}|{\mathbf{x}}_k)
\big)
\Big],
\qquad k=1,\ldots,L.
$$

根据具体模型，完整 ELBO 还包含边界项，例如终止先验项及 $k=0$ 附近的重建项。关键在于，每个去噪项都要求学得的逆向核匹配由前向扰动确定、可计算的 Bayes 后验。

这里，$p_{{\boldsymbol{\phi}}}({\mathbf{x}}_{k-1}={\mathbf{e}}_i|{\mathbf{x}}_k={\mathbf{e}}_j)$ 表示由神经网络参数化的逆向转移概率：给定系统处于状态 ${\mathbf{e}}_j$（对应步骤 $k$），它给出系统来自状态 ${\mathbf{e}}_i$（对应步骤 $k{-}1$）的概率。对于每个固定的 ${\mathbf{e}}_j$，这些值构成关于前一状态的类别分布。

与连续 DDPM 的主要形式差别在于，这些 KL散度是对离散状态的有限求和，而非 Gaussian 分布的闭式表达式。这反映了 Gaussian 分布的一条特殊性质：关于同一变量的两个 Gaussian 密度之积，与另一个 Gaussian 密度成正比。在连续 DDPM 中，前向后验通过 Gaussian 因子相乘计算，因此结果仍为 Gaussian 分布，KL散度便化为仅涉及均值与协方差的简单代数表达式（[第 2.2 节](../02-variational-perspective/02-ddpm.md#sec-ddpm)）。在离散设定中，Bayes 法则给出类别分布，每个 KL散度项都是对状态的求和。这一求和精确且概念简单，但面对大词表或完整序列空间时，可能需要结构、因子分解、子采样或专门的参数化。

参数化还有一个实践要点。ELBO 要求学得的逆向核 $p_{{\boldsymbol{\phi}}}({\mathbf{x}}_{k-1}|{\mathbf{x}}_k)$ 匹配 Bayes 后验，但并不强制该核采用唯一的神经网络参数化。一个常见选择是干净状态预测，即 $x_0$ 预测参数化。精确的边缘逆向核满足恒等式

$$
p({\mathbf{x}}_{k-1}|{\mathbf{x}}_k)
=
\sum_{a}
p({\mathbf{x}}_{k-1}|{\mathbf{x}}_k,{\mathbf{x}}_0={\mathbf{e}}_a)\,
p({\mathbf{x}}_0={\mathbf{e}}_a|{\mathbf{x}}_k),
$$

其中对有效的干净类别求和。这只是对未知干净状态进行边缘化。由于后验 $p({\mathbf{x}}_0|{\mathbf{x}}_k)$ 依赖数据分布，没有可用的闭式表达式，模型可以预测干净状态分布 $\widehat p_{{\boldsymbol{\phi}}}({\mathbf{x}}_0|{\mathbf{x}}_k)$，并用它定义

$$
p_{{\boldsymbol{\phi}}}({\mathbf{x}}_{k-1}|{\mathbf{x}}_k)
:=
\sum_{a}
p({\mathbf{x}}_{k-1}|{\mathbf{x}}_k,{\mathbf{x}}_0={\mathbf{e}}_a)\,
\widehat p_{{\boldsymbol{\phi}}}({\mathbf{x}}_0={\mathbf{e}}_a|{\mathbf{x}}_k).
$$

若 $\widehat p_{{\boldsymbol{\phi}}}({\mathbf{x}}_0|{\mathbf{x}}_k)$ 等于真实后验 $p({\mathbf{x}}_0|{\mathbf{x}}_k)$，这一混合就会恢复精确的逆向核。实践中，它是将学得的逆向转移与已知前向后验联系起来的一种实用方式。

这种干净状态参数化常见于 multinomial／D3PM 风格的离散时间扩散模型（[Hoogeboom et al., 2021](../../references.md#ref-hoogeboom2021argmax)；[Austin et al., 2021](../../references.md#ref-austin2021structured)），但并不构成离散扩散的定义。D3PM 本身也指出，可以直接预测逆向 logits；后续模型则采用针对前向过程设计的其他结构化参数化（[Sahoo et al., 2024](../../references.md#ref-sahoo2024simple)）。因此，干净状态预测应视为变分视角的一种重要参数化，而非逆向过程建模的唯一方式。

**Score／比值视角：学习局部概率比较。** 在连续情形中（[第 3 章](../03-score-based-perspective/index.md#ch-score-based)和[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)），基于 score 的视角从一个简单问题出发：给定扰动后的样本 ${\mathbf{x}}_t \sim p_t$，逆转扩散需要当前密度的哪些局部信息？答案是 score 函数 $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$，它出现在逆时间 SDE 和概率流 ODE 中。因此，用神经网络学习 score 就足以进行生成。

在有限状态空间中，没有环境 Euclidean 几何，没有普通意义的无穷小位移，也没有关于状态的梯度。样本通过从一个状态跳到另一个状态来改变，而非迈出一个小的 Euclidean 步。因此，score $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$ 没有直接的字面对应物。但逆向过程仍必须回答一个类似问题：*如果系统当前处于状态 ${\mathbf{e}}_j$，各候选源状态 ${\mathbf{e}}_i$ 相对它有多大可能性？*

**推导逆向跳跃率。** 为确定所需的量，考虑前向 CTMC $\{{\mathbf{x}}_t\}_{0\leq t\leq T}$，其速率矩阵为 ${\mathbf{Q}}_t$，边缘分布为 ${\mathbf{p}}_t$。元素 $[{\mathbf{Q}}_t]_{ij}$ 是从状态 ${\mathbf{e}}_i$ 跳到状态 ${\mathbf{e}}_j$ 的速率。在短区间 $\Delta t$ 内，无穷小前向事件 ${\mathbf{x}}_t={\mathbf{e}}_i$ 且 ${\mathbf{x}}_{t+\Delta t}={\mathbf{e}}_j$ 的概率，在 $i\neq j$ 时近似为

$$
p_t(i)\,[{\mathbf{Q}}_t]_{ij}\,\Delta t.
$$

现在定义逆时间过程 $\bar{{\mathbf{x}}}_s := {\mathbf{x}}_{T-s}$，其中 $0\leq s\leq T$。其在逆时间 $s$ 的边缘分布为 $\bar{{\mathbf{p}}}_s={\mathbf{p}}_{T-s}$。若逆向过程的速率矩阵为 $\bar{{\mathbf{Q}}}_s$，则从 ${\mathbf{e}}_j$ 到 ${\mathbf{e}}_i$ 的相应逆向跳跃，必须在倒转的时间顺序下复现相同的无穷小路径概率。取极限 $\Delta t\to 0$，得到非对角项的时间反转公式

<span id="eq-reverse-rate-discrete-reverse-time"></span>

$$
[\bar{{\mathbf{Q}}}_s]_{ji}
= [{\mathbf{Q}}_{T-s}]_{ij}\;\frac{p_{T-s}(i)}{p_{T-s}(j)},
\qquad i \neq j,
\tag{12.2.6}
$$

只要 $p_{T-s}(j)>0$ 成立。等价地，若 $t$ 表示相应的前向时间，$\widetilde{{\mathbf{Q}}}_t$ 表示以前向时间为索引的逆向生成元，则

<span id="eq-reverse-rate-discrete"></span>

$$
[\widetilde{{\mathbf{Q}}}_t]_{ji}
= [{\mathbf{Q}}_t]_{ij}\;\frac{p_t(i)}{p_t(j)},
\qquad i \neq j.
\tag{12.2.7}
$$

对角元素由行和为零的约束确定：

$$
[\widetilde{{\mathbf{Q}}}_t]_{jj}
=
-\sum_{i\neq j}[\widetilde{{\mathbf{Q}}}_t]_{ji}.
$$

这是 CTMC 的标准时间反转公式（[Kelly, 1979](../../references.md#ref-kelly1979reversibility)；[Anderson, 2012](../../references.md#ref-anderson2012continuous)），由 [Campbell et al. (2022)](../../references.md#ref-campbell2022continuous) 应用于离散扩散。该公式只在边缘概率为正的状态上解释。此外，逆向过程只反转前向过程允许的转移上的概率通量：若 $[{\mathbf{Q}}_t]_{ij}=0$，则从 ${\mathbf{e}}_j$ 到 ${\mathbf{e}}_i$ 的相应逆向速率也为零。这一结论是无穷小路径概率的局部时间反转恒等式，不是平衡态下的细致平衡假设。

**作为 score 离散对应物的比值。** 现在观察[公式 (12.2.7)](02-discrete-spaces.md#eq-reverse-rate-discrete)。前向速率 $[{\mathbf{Q}}_t]_{ij}$ 是已知的，因为前向过程由设计固定。未知量是比值 $p_t(i)/p_t(j)$，它比较当前边缘分布下状态 ${\mathbf{e}}_i$ 相对于状态 ${\mathbf{e}}_j$ 的可能性。用内积记号，有 $p_t(i) = \langle{\mathbf{p}}_t,{\mathbf{e}}_i\rangle$，因此比值可写为 $\langle{\mathbf{p}}_t,{\mathbf{e}}_i\rangle / \langle{\mathbf{p}}_t,{\mathbf{e}}_j\rangle$。

为什么这可以作为 score 的离散对应物？在连续空间中，score $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$ 度量对数密度在无穷小位移下如何变化。在有限状态空间中，自然的局部比较则跨越状态间允许的跳跃。从 ${\mathbf{e}}_j$ 跳到 ${\mathbf{e}}_i$ 时，对数概率的变化为

$$
\log p_t(i) - \log p_t(j)
= \log \frac{p_t(i)}{p_t(j)},
$$

这是对数概率的有限差分，承担了 $\log p_t$ 在 ${\mathbb{R}}^D$ 中的方向导数所承担的角色。这种比较天然具有*边局部性*：只需为前向跳跃结构连接的状态对提供比值。因此，离散 score 是沿允许转移的一组局部概率比较，而非定义在 $\mathbb{R}^D$ 上的 Euclidean 向量场。对于完整序列，同样的结论成立，只需将 $i$ 和 $j$ 换成完整配置，通常是 Hamming 型或基于图的转移结构下的相邻配置。

**网络学习什么。** 在连续的基于 score 的模型中，训练神经网络 ${\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}, t)$ 近似 score $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$，再将学得的 score 代入逆时间 SDE 生成样本。离散情形在信息层面遵循同一逻辑：网络必须提供随时间变化的比值信息，以通过[公式 (12.2.7)](02-discrete-spaces.md#eq-reverse-rate-discrete)确定逆向跳跃率。随后，利用学得的转移概率或速率，模拟逆时间 Markov 链或 CTMC，完成生成。

不同表述以不同方式表示这一信息。SEDD（[Lou et al., 2024](../../references.md#ref-lou2024discrete)）等基于比值的模型，使用受 score 匹配启发的损失估计比值 $p_t(i)/p_t(j)$ 或其对数形式。其他连续时间表述（[Campbell et al., 2022](../../references.md#ref-campbell2022continuous)；[Sun et al., 2023](../../references.md#ref-sun2023scorebased)）则通过匹配条件转移概率等目标，学习等价的逆向量。将局部概率比较视为 score 的离散对应物这一思想，在 [Meng et al. (2022)](../../references.md#ref-meng2022concrete) 提出的 *concrete score* 中得到形式化。尽管参数化不同，底层原则相同：学习时间反转所需的局部概率比较，构造逆向转移或速率，再进行逆向模拟。

**流视角：学习离散输运场。** 在连续情形中（[第 5 章](../05-flow-based-perspective/index.md#ch-flow-based)），基于流的视角采用另一条路线：直接学习速度场 ${\mathbf{v}}_t({\mathbf{x}})$，其 ODE 流将参考分布输运到数据分布，而非学习 score 后反转 SDE。训练策略是流匹配（[第 5.2 节](../05-flow-based-perspective/02-flow-matching-framework.md#sec-flow-matching-framework)）：边缘速度场不可直接计算，但以数据或端点样本为条件的*条件*速度场 ${\mathbf{v}}_t({\mathbf{x}} | {\mathbf{x}}_0)$ 有可用的闭式表达式。边缘目标与条件目标通过下式关联：

$$
{\mathbf{v}}_t({\mathbf{x}})
= {\mathbb{E}} \big[{\mathbf{v}}_t({\mathbf{x}} | {\mathbf{x}}_0) | {\mathbf{x}}_t = {\mathbf{x}}\big],
$$

这是在给定当前状态时，对所有可能的条件变量取平均。显式计算这一平均值需要知道连续边缘密度 $p_t({\mathbf{x}})$。关键的训练技巧（[第 5.3.2 节](../05-flow-based-perspective/03-probability-paths-and-velocities.md#subsec-fm-instantiations)）是：在平方误差回归下，条件目标与不可直接计算的边缘目标具有相同的总体最小化解。因此，训练可以使用可计算条件路径中的样本，避免显式计算边缘分布。

离散情形采用同一思想，以速率矩阵替代速度场。由于离散状态不能沿穿过 token 空间的连续轨迹移动，速度场的角色由速率矩阵 ${\mathbf{Q}}_t$（也称 Markov 生成元）承担。回顾元素 $[{\mathbf{Q}}_t]_{ij}$ 指定从状态 ${\mathbf{e}}_i$ 到状态 ${\mathbf{e}}_j$ 的跳跃率；每单位时间从 ${\mathbf{e}}_i$ 实际传送到 ${\mathbf{e}}_j$ 的概率质量，就是通量

$$
J_t(i,j)=p_t(i)\,[{\mathbf{Q}}_t]_{ij}.
$$

在这个意义上，速率矩阵是离散输运场：它确定概率质量如何随时间重新分配。

其中有一个关键细节。概率路径 $\{{\mathbf{p}}_t\}_{t\in[0,T]}$ 不能唯一确定速率矩阵。许多不同生成元都可以在主方程中产生同一个导数 $\mathop{}\!\mathrm{d}{\mathbf{p}}_t/\mathop{}\!\mathrm{d} t$，正如不同输运场有时能在不同几何约束下实现相同的边缘演化。因此，离散流匹配方法不仅要指定概率路径，还要指定沿该路径分配条件跳跃率或概率流（probability current）的规则。

与连续流匹配一样，边缘速率矩阵通常不可直接计算，但条件版本往往可以显式构造。若以端点为条件，例如干净数据状态和／或参考状态，就能指定条件概率路径，并推导沿该路径的可计算条件跳跃率。随后，用适当的合宜损失（proper loss）训练神经网络，使其根据扰动样本预测对应的转移或速率信息。边缘生成元通过相同的条件期望原则恢复：给定当前状态，对未知端点分布下的条件输运对象取平均，但通过对条件目标进行有监督回归或分类来学习这一平均值，而不显式计算它。

因此，与连续流匹配的对应关系为：

-   连续流匹配使用条件速度目标训练网络；边缘速度作为流匹配损失下的最优预测器出现。

-   离散流匹配使用条件转移或跳跃率目标训练网络；边缘离散输运场通过类似的条件期望机制出现。

两种情形都利用可计算的条件路径及采样的中间状态，避免处理不可直接计算的边缘分布。

[Campbell et al. (2024)](../../references.md#ref-campbell2024generative) 和 [Gat et al. (2024)](../../references.md#ref-gat2024discrete) 以离散流模型或离散流匹配的名称发展了这一视角。更广泛地说，流视角说明的是*学习什么对象*：离散状态空间上的输运场，可由转移、速率或概率流表示。

## 12.2.4 哪些相似，哪些存在实质差异？ {#subsec-epilogue-genuine-diff}

下表概括连续与离散扩散模型的对应关系。逐行比较，可以看到相同概念要素如何在两个不同设定中具体实现：

<span id="tb-continuous-discrete-diffusion"></span>

**表 12.1：** **概率输运的连续与离散视角。** 连续状态空间与有限离散状态空间中出现相同的生成建模要素，但数学对象发生变化：密度变为概率向量，ODE／SDE 动力学变为 Markov 链或 CTMC，逆向建模则通过转移核、score／比值信息或离散输运场表达。

<table markdown="1" style="display: table; overflow: visible;">
<tr markdown="1">
<th markdown="block">

**状态**

</th>
<th markdown="block">

**连续** $({\mathbf{x}} \in {\mathbb{R}}^D)$

</th>
<th markdown="block">

**离散** $({\mathbf{x}} \in \mathcal{V} \subset \{0,1\}^K)$

</th>
</tr>
<tr markdown="1">
<td markdown="block">

**分布**

</td>
<td markdown="block">

密度 $p_t({\mathbf{x}})$

</td>
<td markdown="block">

概率向量 ${\mathbf{p}}_t \in \Delta^{K-1}$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**样本动力学**

</td>
<td markdown="block">

ODE 或 SDE 轨迹

</td>
<td markdown="block">

Markov 链或 CTMC 跳跃

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**分布演化**

</td>
<td markdown="block">

连续性方程或 Fokker–Planck 方程

</td>
<td markdown="block">

转移矩阵传播或主方程

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**输运对象**

</td>
<td markdown="block">

速度、漂移、扩散或 score 场

</td>
<td markdown="block">

转移核、速率矩阵、比值或概率流

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**参考分布**

</td>
<td markdown="block">

简单分布，例如 Gaussian 分布

</td>
<td markdown="block">

简单分布，例如均匀或全掩码分布

</td>
</tr>
<tr markdown="1">
<th colspan="3" markdown="block">

**逆向建模的三个视角**

</th>
</tr>
<tr markdown="1">
<td markdown="block">

**变分**

</td>
<td markdown="block">

使用 Gaussian 或连续转移的 ELBO

</td>
<td markdown="block">

使用类别转移的 ELBO

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**Score／比值**

</td>
<td markdown="block">

学习 score $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$

</td>
<td markdown="block">

学习比值或对数比值，例如 $p_t(i)/p_t(j)$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

**流**

</td>
<td markdown="block">

学习速度场 ${\mathbf{v}}_t({\mathbf{x}})$

</td>
<td markdown="block">

学习转移场、速率场或概率流场

</td>
</tr>
</table>

两者的对应关系很清楚，但离散设定并不是简单地将连续理论中的梯度替换为比值。还有若干更深层的差异。

**没有字面意义上的逆映射。** 在确定性的连续归一化流中，逆向映射就是前向映射的逆。在离散 Markov 过程中，前向核可能合并多个状态的质量，未必可逆。因此，逆向过程并非由矩阵求逆得到，而是依赖当前边缘分布的 Bayes 构造或时间反转构造。这正是逆向转移概率包含 $p_k(i)/p_{k+1}(j)$ 等因子、逆向跳跃率包含 $p_t(i)/p_t(j)$ 等比值的原因。

**几何与样本路径。** 在连续 Euclidean 空间中，单个样本沿连续时间轨迹演化：ODE 解光滑，SDE 样本路径虽然通常处处不可微，但依然连续。这种微分结构使速度场、score 函数和散度等对象具有意义。在有限状态空间中，单条轨迹则是*纯跳跃路径*：它在某个状态停留一段随机时间，然后突然跳到另一状态。光滑演化的是概率向量 ${\mathbf{p}}_t$，它在概率单纯形上按照主方程随时间连续演化；样本路径本身并不光滑。因此，离散扩散仍可在概率演化层面得到统一描述，但样本层面的动力学由跳跃过程支配，而非数据空间中的连续 ODE／SDE 路径。

**前向扰动结构与逆向参数化。** 在离散扩散中，前向扰动过程往往是实质性的建模选择。在连续 Gaussian 扩散中，可以改变调度 $\alpha_t$ 和 $\sigma_t$，但前向核通常仍为 Gaussian，因此主要影响是控制扰动的速率与尺度。在离散扩散中，转移规则本身也可以改变：可以采用均匀扰动、掩码、最近邻转移、图结构转移、特定领域转移，或更一般的概率路径。如 D3PM（[Austin et al., 2021](../../references.md#ref-austin2021structured)）所强调，前向过程可以直接纳入特定领域的归纳偏置，进而塑造逆向生成问题。

这一区别也改变了逆向参数化的地位。在 Gaussian 设定中，噪声、去噪数据、score 和速度等常见目标通过简单的时间依赖仿射变换联系起来。在离散设定中，逆向转移概率、干净状态预测器、概率比和跳跃率则通过归一化、Bayes 法则和所选转移图关联。它们相互联系，却不能像 Gaussian 情形那样仅作为可互换的重参数化。因此，变分、score／比值和流描述是逆向问题的不同视角，并非同一个代数目标的不同名称。

**采样机制。** 采样过程也有所不同。在连续扩散中，生成通常通过在选定时间网格上数值积分 ODE 或 SDE 实现。在离散扩散中，生成意味着模拟逆向 Markov 链或 CTMC。离散时间下，这等同于采样类别型逆向转移；连续时间下，则可能从学得的速率模拟跳跃，例如通过基于事件的模拟或时间离散化。因此，连续扩散中数值 ODE／SDE 求解器的角色，在离散设定中由类别转移采样或跳跃过程模拟承担。

**为何到此为止？** 本节旨在辨认跨越不同状态空间仍保持不变的原则，而非完整发展离散扩散模型的方法体系。关键在于结构：连续空间中的分布演化由变量变换原则及其微分形式支配，包括连续性方程和 Fokker–Planck 方程；离散空间中的对应角色由 Markov 算子与主方程承担。两种设定的建模逻辑相同：定义前向扰动过程，理解它如何输运分布，再学习用于生成的逆向输运机制。这一逻辑不依赖底层状态空间是连续还是离散。

变化的是实现这一方案所用的数学对象，方案本身保持不变。连续空间使用密度、梯度、速度场及 SDE 或 ODE 动力学；离散空间使用概率向量、转移核、跳跃率、比值和概率流。这些差异在技术上很重要，会引出不同的参数化、训练和采样问题，但不会改变更深层的结论：概率输运视角不局限于某个状态空间、网络结构或论文脉络。它是一条足够广泛的一般组织原则，能够在同一个概念框架内容纳连续与离散扩散。

[^1]: CTMC 和随机过程文献中的标准等价表述（[Norris, 1998](../../references.md#ref-norris1998markov)；[Kelly, 1979](../../references.md#ref-kelly1979reversibility)），将每个状态表示为标量索引 $x\in\{1,\dots,K\}$，将转移率写为 $[{\mathbf{Q}}_t]_{ij}$，将边缘分布写为 $p_t(x)$。这里采用的独热编码在记号上等价，但更贴近离散扩散文献（[Austin et al., 2021](../../references.md#ref-austin2021structured)；[Sahoo et al., 2024](../../references.md#ref-sahoo2024simple)；[Lou et al., 2024](../../references.md#ref-lou2024discrete)），其中 $\langle{\mathbf{x}}_{{\boldsymbol{\phi}}},{\mathbf{x}}\rangle$ 等内积和 ${\mathbf{P}}^{\top}{\mathbf{x}}$ 等矩阵与向量的乘积会自然出现在训练目标中。

[^2]: 所得扰动序列的边缘分布通常不必分解，因为干净数据分布可能包含位置之间的依赖。

[^3]: 这直接来自 ${\mathbf{P}}_{t,\Delta t}$ 的行和约束。由于转移矩阵每一行的和为 $1$，代入展开式 ${\mathbf{P}}_{t,\Delta t} = {\mathbf{I}} + {\mathbf{Q}}_t\,\Delta t + o(\Delta t)$ 得到 $1 + \Delta t \sum_{j} [{\mathbf{Q}}_t]_{ij} + o(\Delta t) = 1$，从而要求 $\sum_{j} [{\mathbf{Q}}_t]_{ij} = 0$。分离对角项便得到 $[{\mathbf{Q}}_t]_{ii} = -\sum_{j \neq i} [{\mathbf{Q}}_t]_{ij}$。

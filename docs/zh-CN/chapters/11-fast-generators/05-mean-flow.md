# 11.5 一般流映射：Mean Flow {#sec-mf}

正如扩散模型具有多种等价的参数化和训练目标，一般流映射 ${\boldsymbol{\Psi}}_{s\to t}$ 也可以通过多种合理方式学习。本节介绍 *Mean Flow*（MF）（[Geng et al., 2025](../../references.md#ref-geng2025mean)），它是一般流映射方法族 ${\boldsymbol{\Psi}}_{s\to t}$ 中较晚出现的代表，展示了如何有效学习这类映射的另一种具有理论依据的视角。

## 11.5.1 Mean Flow 的建模与训练 {#modeling-and-training}

CM 和 CTM 建立在 EDM 框架之上，MF 则基于流匹配表述（$\alpha_t = 1-t$ 且 $\sigma_t = t$，其中 $t\in[0,1]$）。MF 不直接参数化流映射，而是学习区间 $[t,s]$（其中 $t<s$）上的*平均漂移*：

$$
{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s, t)  \approx
{\mathbf{h}}^*({\mathbf{x}}_s, s, t) := \frac{1}{t-s}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u.
$$

相应的 oracle 损失为

<span id="eq-mf-oracle"></span>

$$
\begin{aligned}
\mathbb{E}_{t<s} \mathbb{E}_{{\mathbf{x}}_s\sim p_s}\Big[w(s)
\|{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t) - {\mathbf{h}}^*({\mathbf{x}}_s,s,t)\|_2^2\Big].
\end{aligned}
\tag{11.5.1}
$$

特别地，当 $s \to t$ 时，损失函数退化为流匹配损失：

<span id="eq-mf-to-fm"></span>

$$
\begin{aligned}
\mathbb{E}_{t} \mathbb{E}_{{\mathbf{x}}_t\sim p_t}\Big[w(t)
\|{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_t,t,t) - {\mathbf{v}}^*({\mathbf{x}}_t,t)\|_2^2\Big],
\end{aligned}
\tag{11.5.2}
$$

用于学习瞬时速度。后文[第 11.5.3 节](05-mean-flow.md#subsec-equivalence-ctm-mf)将说明，MF 仍然符合[公式 (10.1.4)](01-prologue.md#eq-general-cm-loss-rep)中的一般目标，只是采用了不同但等价的视角。由于 oracle 回归目标 ${\mathbf{h}}^*({\mathbf{x}}_s,s,t)$ 通常没有闭式表达式，MF 利用对下式求导所得的恒等式构造替代目标：

$$
(t-s)\,{\mathbf{h}}^*({\mathbf{x}}_s,s,t) = \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u
$$

关于 $s$ 求导，得到 *MF 恒等式*：

<span id="eq-mf-identity"></span>

$$
\begin{aligned}
    {\mathbf{h}}^*({\mathbf{x}}_s,s,t)
&= {\mathbf{v}}^*({\mathbf{x}}_s,s) - (s-t) \tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{h}}^*({\mathbf{x}}_s,s,t) \\
&= {\mathbf{v}}^*({\mathbf{x}}_s,s) - (s-t)\Big[(\partial_{{\mathbf{x}}}{\mathbf{h}}^*)({\mathbf{x}}_s,s,t) {\mathbf{v}}^*({\mathbf{x}}_s,s)
+ \partial_s {\mathbf{h}}^*({\mathbf{x}}_s,s,t)\Big],
\end{aligned}
\tag{11.5.3}
$$

其中第二行使用链式法则，并结合

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{x}}_s = {\mathbf{v}}^*({\mathbf{x}}_s,s).
$$

受这一恒等式启发，MF 用停止梯度的替代量取代不可直接计算的 oracle，得到实际训练目标

<span id="eq-mf-practice"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{MF}}({\boldsymbol{\theta}})
:= \mathbb{E}_{t<s} \mathbb{E}_{{\mathbf{x}}_s\sim p_s}\Big[w(s)\,
\|{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t) - {\mathbf{h}}_{{\boldsymbol{\theta}}^-}^{\mathrm{tgt}}({\mathbf{x}}_s,s,t)\|_2^2\Big],
\end{aligned}
\tag{11.5.4}
$$

其中回归目标定义为

$$
\begin{aligned}
{\mathbf{h}}_{{\boldsymbol{\theta}}^-}^{\mathrm{tgt}}&({\mathbf{x}}_s,s,t)
:= \\ &{\mathbf{v}}^*({\mathbf{x}}_s,s) - (s-t)\Big[\underbrace{(\partial_{{\mathbf{x}}}{\mathbf{h}}_{{\boldsymbol{\theta}}^-})({\mathbf{x}}_s,s,t) {\mathbf{v}}^*({\mathbf{x}}_s,s)
+ \partial_s {\mathbf{h}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s,t)}_{\mathrm{JVP}}\Big].
\end{aligned}
$$

实践中，oracle 速度 ${\mathbf{v}}^*$ 无法以闭式计算，必须近似。有两种常见策略：依赖预训练扩散模型（*蒸馏*），或直接从数据构造估计量（*从头训练*）。无论选择哪种策略，最终都需要计算目标网络 ${\mathbf{h}}_{{\boldsymbol{\theta}}^-}$ 的 *Jacobian–向量积*（JVP）：

$$
\left[\partial_{\mathbf{x}} {\mathbf{h}}_{{\boldsymbol{\theta}}^-}, \partial_s {\mathbf{h}}_{{\boldsymbol{\theta}}^-}, \partial_t {\mathbf{h}}_{{\boldsymbol{\theta}}^-}\right]^\top \cdot \left[{\mathbf{v}}^*, 1, 0\right]
$$

**蒸馏。** 使用以流匹配为基础的预训练扩散模型，即 ${\mathbf{v}}_{{\boldsymbol{\phi}}^\times} \approx {\mathbf{v}}^*$。

**从头训练。** 使用单点条件速度 $\alpha_s'{\mathbf{x}}_0 + \sigma_s'{\boldsymbol{\epsilon}}$，它由前向加噪 ${\mathbf{x}}_s=\alpha_s{\mathbf{x}}_0+\sigma_s{\boldsymbol{\epsilon}}$ 得到，其中 ${\boldsymbol{\epsilon}}\sim\mathcal N(\boldsymbol{0},{\mathbf{I}})$。对配对样本求值时，这给出了噪声水平 $s$ 处瞬时漂移的无偏单样本估计，对应的配对样本为 $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$。

## 11.5.2 Mean Flow 的采样 {#sampling}

MF ${\mathbf{h}}_{{\boldsymbol{\theta}}^\times}$ 训练好后，自然给出流映射的近似。对于任意起点 ${\mathbf{x}}_s$，从 $s$ 到 $t$ 的映射（近似）为

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s) = {\mathbf{x}}_s + (t-s)\,{\mathbf{h}}^*({\mathbf{x}}_s,s,t)
  \approx   {\mathbf{x}}_s + (t-s)\,{\mathbf{h}}_{{\boldsymbol{\theta}}^\times}({\mathbf{x}}_s,s,t).
$$

这同时支持单步和多步采样。例如，抽取 ${\mathbf{x}}_T\sim p_{\mathrm{prior}}$ 后，干净样本的单步生成为

$$
{\mathbf{x}}_0  \leftarrow  {\mathbf{x}}_T - T\,{\mathbf{h}}_{{\boldsymbol{\theta}}^\times}({\mathbf{x}}_T,T,0).
$$

另一种方式是预先准备时间网格，再按顺序应用映射，以与 CTM 相同的时间步进方式进行多步生成。由于 MF 学习一般流映射，它也支持 CTM 的 $\gamma$ 采样，其中可控超参数 $\gamma$ 向采样过程注入随机性。

## 11.5.3 CTM 与 MF 的关系 {#subsec-equivalence-ctm-mf}

乍看之下，CTM 与 MF 似乎没有关联。实际上，两者对同一个 oracle 流映射 ${\boldsymbol{\Psi}}_{s\to t}$ 提供不同的参数化，相应的平方 oracle 回归误差通过依赖时间的权重联系起来（[Hu et al., 2025](../../references.md#ref-hu2025cmt)）。

**参数化的关系。** 两种方法都在同一个一般框架下运作，但以不同方式表示所学习的函数。流映射可等价地写为

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
& = {\mathbf{x}}_s + \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u, u)\mathop{}\!\mathrm{d} u \\
&= \frac{t}{s}{\mathbf{x}}_s
   + \frac{s-t}{s}\underbrace{\Bigg[{\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u\Bigg]}_{\approx\,{\mathbf{g}}_{\boldsymbol{\theta}}} \\
&= {\mathbf{x}}_s + (t-s)\underbrace{\Bigg[\frac{1}{t-s}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u\Bigg]}_{\approx\,{\mathbf{h}}_{\boldsymbol{\theta}}}.
\end{aligned}
$$

这里，第一个形式是流映射的定义；第二个形式突出通过 ${\mathbf{g}}_{\boldsymbol{\theta}}$ 表达的 CTM 参数化（见[公式 (11.4.1)](04-consistency-trajectory-models.md#eq-ctm-g-oracle)和[公式 (11.4.2)](04-consistency-trajectory-models.md#eq-ctm-para-1)）；最后一个形式突出通过 ${\mathbf{h}}_{\boldsymbol{\theta}}$ 表达的 MF 参数化。

**训练损失的关系。** 根据上述解释，oracle 流映射 ${\boldsymbol{\Psi}}_{s\to t}$ 可以用 CTM 参数化表示为

$$
{\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t) \approx {\mathbf{g}}^*({\mathbf{x}}_s,s,t):={\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u
$$

也可以用 MF 参数化表示为

$$
{\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t) \approx {\mathbf{h}}^*({\mathbf{x}}_s,s,t):=\frac{1}{t-s}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u,
$$

现在说明，CTM 与 MF 的训练损失实际上等价。

考虑关系

$$
{\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t) := {\mathbf{x}}_s - s {\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t),
$$

并以 $d({\mathbf{x}},{\mathbf{y}}):=\|{\mathbf{x}}-{\mathbf{y}}\|^2$ 为例。代入[公式 (10.1.4)](01-prologue.md#eq-general-cm-loss-rep)，并将 ${\mathbf{G}}_{\boldsymbol{\theta}}$ 视为 CTM 的流映射参数化（[公式 (11.4.2)](04-consistency-trajectory-models.md#eq-ctm-para-1)），可得

<span id="eq-g-para"></span>
<span id="eq-h-para"></span>

$$
\begin{align*}
&~d\big({\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t), {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)\big)  \\ = &~\left\lVert{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t) - {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)\right\rVert^2  \\ = &~\left\lVert\left(\frac{t}{s}{\mathbf{x}}_s + \frac{s-t}{s} {\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\right)-\left(\frac{t}{s}{\mathbf{x}}_s + \frac{s-t}{s}\Big[{\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\Big]\right)\right\rVert^2  \\ = &~\left(\frac{s-t}{s}\right)^2\left\lVert{\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)-\left({\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\right)\right\rVert^2 \tag{11.5.5} \\ = &~\left(\frac{s-t}{s}\right)^2\left\lVert\left({\mathbf{x}}_s - s {\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\right)-\left({\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\right)\right\rVert^2  \\ = &~\left(\frac{s-t}{s}\right)^2\left\lVert\left({\mathbf{x}}_s - s {\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\right)-\left({\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\right)\right\rVert^2  \\ = &~\left(s-t\right)^2\left\lVert {\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)- \left(\frac{1}{t-s}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\mathop{}\!\mathrm{d} u \right)\right\rVert^2 \tag{11.5.6}
\end{align*}
$$

因此，

$$
\begin{aligned}
\frac{1}{s^2} \Bigg\|{\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)-{\mathbf{g}}^*({\mathbf{x}}_s,s,t) \Bigg\|^2 = \Bigg\|{\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)-{\mathbf{h}}^*({\mathbf{x}}_s,s,t) \Bigg\|^2.
\end{aligned}
$$

不考虑学习流映射所用的具体算法，CTM 与 MF 的损失具有相同的数学形式，仅相差一个权重函数。此外，在任一情形中令 $t=0$，就恢复 CM 的设定（${\boldsymbol{\Psi}}_{s\to 0}$），即每个状态直接映射到干净数据。

**实践中的辅助损失。** CTM 将[公式 (11.4.7)](04-consistency-trajectory-models.md#eq-ctm-consist-scratch)中的一致性损失与[公式 (11.4.8)](04-consistency-trajectory-models.md#eq-ctm-dsm)中自身定义的扩散模型损失联合用于训练。MF 也采用类似策略。如[公式 (11.5.2)](05-mean-flow.md#eq-mf-to-fm)所示，当 $s \to t$ 时，MF 损失退化为标准流匹配目标。实践中，MF 控制满足 $s \neq t$ 的时间对与满足 $s = t$ 的时间对之间的比例；因此，整体优化混合了[公式 (11.5.4)](05-mean-flow.md#eq-mf-practice)中的 MF 目标与[公式 (11.5.2)](05-mean-flow.md#eq-mf-to-fm)中的流匹配目标。

这两种参数化都能平滑地从扩散模型训练过渡到流映射学习：前者用固定回归目标学习瞬时速度，后者使用停止梯度的伪回归目标。

**CTM 与 MF 参数化均支持灵活推理。** CTM（${\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)$）与 MF（${\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)$）都旨在近似底层流映射 ${\boldsymbol{\Psi}}_{s\to t}$：

$$
{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\approx{\boldsymbol{\Psi}}_{s\to t},\quad  \text{and}
\quad
{\mathbf{x}}_s+(t-s){\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\approx{\boldsymbol{\Psi}}_{s\to t}.
$$

由于两种模型都学习任意两个时间步之间的显式映射，它们自然支持 CTM 的 $\gamma$ 采样，也兼容最初为扩散模型开发的推理技术，例如引导（[第 8 章](../08-guidance/index.md#ch-guidance)）、精确似然计算（[公式 (4.3.7)](../04-score-sde-framework/03-training-and-sampling.md#eq-score-sde-likelihood)）以及使用高阶求解器的加速采样（[第 9 章](../09-numerical-solvers/index.md#ch-numerical-solvers)）。这种兼容性来自其参数化在无穷小极限 $t\to s$ 下能够恢复瞬时扩散漂移：

$$
{\mathbf{g}}^*({\mathbf{x}}_s,s,s)={\mathbf{x}}_s-s\,{\mathbf{v}}^*({\mathbf{x}}_s,s),
\quad  \text{and}
\quad
{\mathbf{h}}^*({\mathbf{x}}_s,s,s)={\mathbf{v}}^*({\mathbf{x}}_s,s).
$$

CM 方法族等专门的流映射表述 ${\boldsymbol{\Psi}}_{s\to0}$ 不具有这一性质。因此，CTM 与 MF 都可视为灵活的一般流映射表述，将基于扩散的推理推广为直接的时刻间映射。

**结论。**

CTM 与 MF 之间的关系，类似于扩散模型不同参数化之间的关系（[第 6.3 节](../06-unified-perspective/03-equivalent-parameterizations.md#sec-equivalent-parametrizations)）：在 oracle 层面，它们编码同一个底层流映射。在平方误差回归和上述参数化关系下，两者的 oracle 回归误差仅相差一个已知的、依赖时间的缩放因子。但它们实际使用的训练目标仍然不同，因为采用了不同的替代目标与优化过程。

尽管 CTM 与 MF 存在这种联系，它们学习目标的方式在概念和算法上都有差别。CTM 的目标直接作用于流映射，因而以“积分”形式定义；MF 的目标则依赖流映射的有限差分，以“微分”形式定义。由于积分形式的目标无法直接计算，CTM 必须在训练中求解 ODE。MF 则利用 MF 恒等式（见[公式 (11.5.3)](05-mean-flow.md#eq-mf-identity)）构造可计算的微分形式目标。这类似于基于能量的方法与基于 score 的方法之间的关系：前者学习概率密度本身（积分形式），后者匹配该密度的梯度（微分形式）。实践中，由于损失权重、网络结构及优化动力学等因素，CTM 与 MF 可能表现不同，在某些条件下其中一种方法可能优于另一种。

这一视角表明，CTM 与 MF 并非仅有的可行表述。流映射的其他参数化也可能支持高效、稳定的训练，进而产生新的独立生成模型。探索这些替代方式，可以进一步丰富扩散模型及其流映射扩展，拓展少步生成的能力边界。

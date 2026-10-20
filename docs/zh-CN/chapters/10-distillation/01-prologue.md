# 10.1 引言 {#sec-distillation-prologue}

采样速度慢是扩散模型的一个主要瓶颈。

由 Tweedie 公式（[第 6.3.1 节](../06-unified-perspective/03-equivalent-parameterizations.md#subsec-four-equiv-para)）可知，扩散模型可以理解为一个“${\mathbf{x}}$ 预测”模型 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t, t)$，通过训练从含噪输入 ${\mathbf{x}}_t$ 恢复干净数据的期望，其中噪声水平为 $t$：

$$
{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t, t) \approx \mathbb{E}[{\mathbf{x}}_0 |{\mathbf{x}}_t],
$$

其中期望相对于 $p({\mathbf{x}}_0 |{\mathbf{x}}_t)$ 计算，该分布表示与 ${\mathbf{x}}_t$ 对应的所有可能的干净数据。一个自然的想法是使用 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t, t)$ 进行单步生成。然而，由于这个去噪器对许多可能的结果取平均，预测可能过于平滑，只用少量去噪步骤生成时便会得到模糊、质量较低的样本。

另一方面，如[第 4.3.2 节](../04-score-sde-framework/03-training-and-sampling.md#subsec-sde-sample)所述，扩散采样通过一长串迭代步骤沿 ODE 或 SDE 轨迹推进。这能生成高保真样本，但反复计算神经网络会带来高昂的采样开销。对于一个 $n$ 阶确定性 ODE 求解器，在标准正则性假设下，局部截断误差为 $\mathcal{O}(h^{n+1})$，固定时间区间上的全局离散化误差为 $\mathcal{O}(h^n)$，其中 $h=\max_i |t_i-t_{i-1}|$ 是步长。因此，对于固定的已学习向量场，采用更粗的离散化通常会增加数值误差，不过样本保真度未必随求解器步数的减少而单调下降。这使得实际应用中需要权衡采样精度与计算成本。

为克服这一瓶颈，一个主要研究方向是*蒸馏*：假设已有训练充分的扩散模型（*教师*），再训练一个生成器（*学生*），通过一次前向传播或少量步骤的计算来复现教师的行为。这样便将教师的众多采样步骤压缩为一个快速过程，在维持高样本保真度的同时，有效绕过缓慢的迭代求解器。

下面我们介绍蒸馏的两个视角：*分布层面的蒸馏*与*流映射层面的蒸馏*[^1]。

## 10.1.1 分布层面的蒸馏 {#distribution-level-distillation}

基于分布的蒸馏旨在训练一个单步生成器 ${\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})$，将噪声 ${\mathbf{z}} \sim p_{\mathrm{prior}}$ 映射为样本 $\hat{\mathbf{x}} = {\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})$，从而诱导分布 $p_{\boldsymbol{\theta}}(\hat{\mathbf{x}})$，以近似目标数据分布 $p_{\mathrm{data}}({\mathbf{x}})$。通常通过最小化某种统计散度来实现这一目标：

$$
\min_{\boldsymbol{\theta}}  \mathcal{D}\!\left(p_{\boldsymbol{\theta}}(\hat{\mathbf{x}}),  p_{\mathrm{data}}(\hat{\mathbf{x}})\right),
$$

其中 $\mathcal{D}$ 表示合适的散度度量，例如 KL散度。

在实践中，基于分布的方法将生成器的分布与预训练扩散模型产生的经验分布 $p_{\boldsymbol{\phi}^\times}({\mathbf{x}})$ 对齐：

$$
\min_{\boldsymbol{\theta}}  \mathcal{D}\!\left(p_{\boldsymbol{\theta}}(\hat{\mathbf{x}}),  p_{\boldsymbol{\phi}^\times}(\hat{\mathbf{x}})\right),
$$

其中 $p_{\boldsymbol{\phi}^\times}$ 作为 $p_{\mathrm{data}}$ 的替代。这些方法不显式计算该散度，而是利用预训练教师，以及从学生当前分布估计得到的量，构造其梯度的可计算估计量。对于基于 KL散度的分布匹配，该梯度表现为学生 score 函数与目标 score 函数之差。

这一表述通过分布对齐，将扩散模型的多步生成过程蒸馏为单步模型。我们将在[第 10.2 节](02-distribution-based-distillation.md#sec-vsd)详细介绍这一方法。

## 10.1.2 流映射层面的蒸馏 {#flow-map-level-distillation}

我们考虑 PF-ODE，它可以用任意一种预测模型表示（见[公式 (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)）：

<span id="eq-pf-ode-ct"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(\tau)}{\mathop{}\!\mathrm{d} \tau}
    = f(\tau) {\mathbf{x}}(\tau) - \tfrac{1}{2}g^2(\tau) \nabla_{\mathbf{x}} \log p_\tau({\mathbf{x}}(\tau))
    =: {\mathbf{v}}^*({\mathbf{x}}(\tau), \tau).
\end{aligned}
\tag{10.1.1}
$$

从 ${\mathbf{x}}_s$ 出发，由时刻 $s$ 逆向演化至时刻 $t \leq s$ 的解映射（流映射）记为 ${\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$；即

<span id="eq-int-target-gt"></span>

$$
\begin{aligned}
    {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    &\coloneqq {\mathbf{x}}_s + \int_s^t {\mathbf{v}}^*({\mathbf{x}}(\tau), \tau) \mathop{}\!\mathrm{d} \tau,
\end{aligned}
\tag{10.1.2}
$$

其中积分给出 PF-ODE 的解。直观地说，${\boldsymbol{\Psi}}_{s\to t}$ 将噪声 ${\mathbf{x}}_s$ 从时刻 $s$ 输运到时刻 $t$ 噪声更少的状态（最终到达 $t=0$ 时的数据）。

从扩散模型采样，相当于计算 ${\boldsymbol{\Psi}}_{T\to 0}({\mathbf{x}}_T)$，其中 ${\mathbf{x}}_T \sim p_{\mathrm{prior}}$。通常利用速度场 ${\mathbf{v}}$，通过迭代数值求解器近似这一积分（见[第 9 章](../09-numerical-solvers/index.md#ch-numerical-solvers)），但这需要很多步（例如，即使用 DPM-Solver 也至少需要 $10$—$20$ 次 NFE），使其采样比 GAN 等经典单步生成模型更慢。这引出了一个自然的问题：

<span id="question-10-1-1"></span>

**问题 10.1.1**

我们能否直接学习解映射 ${\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$？

特别地，学习映射 ${\boldsymbol{\Psi}}_{T\to 0}({\mathbf{x}}_T)$，其中 ${\mathbf{x}}_T \sim p_{\mathrm{prior}}$，便能够实现单步生成。

**轨迹蒸馏。**

轨迹蒸馏旨在训练一个神经生成器，在实例层面近似解映射。由于 PF-ODE 的积分很少具有闭式解，训练时必须进行数值近似。为形式化描述这一过程，我们引入通用的求解器记号

<span id="eq-solver"></span>

$$
\begin{aligned}
\mathtt{Solver}_{s\to t}({\mathbf{x}}_s;\,\boldsymbol{\phi}^\times)
    \quad \text{or simply} \quad
    \mathtt{Solver}_{s\to t}({\mathbf{x}}_s),
\end{aligned}
\tag{10.1.3}
$$

表示对经验 PF-ODE 从 $s$ 到 $t$ 进行数值积分，起点为 ${\mathbf{x}}_s$，使用教师参数 $\boldsymbol{\phi}^\times$（上下文明确时省略）。

**早期方法：直接知识蒸馏。** 为实现少步甚至单步生成，一种直接方法是训练生成器 ${\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T, T,0)$，模仿数值求解器沿完整轨迹计算所得的输出：

$$
{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T, T,0)\approx \mathtt{Solver}_{T\to 0}({\mathbf{x}}_T),
\qquad {\mathbf{x}}_T \sim p_{\mathrm{prior}}.
$$

这一思想构成了最早的轨迹蒸馏方法之一 *Knowledge Distillation*（[Luhman and Luhman, 2021](../../references.md#ref-luhman2021knowledge)）的基础，其使用如下回归损失：

$$
\mathcal{L}_{\mathrm{KD}}({\boldsymbol{\theta}})
:=
{\mathbb{E}}_{{\mathbf{x}}_T \sim p_{\mathrm{prior}}}
\left\|
{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T, T,0)
-
\mathtt{Solver}_{T\to 0}({\mathbf{x}}_T)
\right\|_2^2.
$$

尽管这一方法能直接获得预训练教师的监督，却无法利用原始训练数据所提供的强监督。此外，如果在训练循环中调用 ODE 积分，计算成本会很高，因为每次参数更新都需要求解 ODE 来构造目标。最后，由于生成器只学习从 $T$ 到 $0$ 的全局映射，可能丧失从中间状态引导生成过程的可控性。因此，[第 8 章](../08-guidance/index.md#ch-guidance)介绍的大多数可控生成技术都无法直接应用。

**Progressive Distillation 导引。** *Progressive Distillation*（PD）（[Salimans and Ho, 2021](../../references.md#ref-salimans2021progressive)）训练一个以时间为条件的 $\mathtt{Student}$，使用 $\mathtt{Teacher}$ 轨迹片段提供的*局部*监督。设 $t_0=T>t_1>\cdots>t_N=0$ 为固定时间网格。$\mathtt{Teacher}$ 提供时间步进映射 $\mathtt{Teacher}_{t_k\to t_{k+1}}$，其中 $k=0,\ldots,N-1$。

PD 不只监督单次跨越 $T\to 0$，而是训练 $\mathtt{Student}$ 的两步跨越映射，使其匹配两个连续的 $\mathtt{Teacher}$ 步骤：

$$
\mathtt{Student}_{t_k\to t_{k+2}}
\;\approx\;
\mathtt{Teacher}_{t_{k+1}\to t_{k+2}}
\circ
\mathtt{Teacher}_{t_{k}\to t_{k+1}},
$$

其中 $k = 0, 2, 4, \dots$。匹配使用简单的回归损失（例如均方误差）。

在局部配对的片段上训练后，$\mathtt{Student}$ 不再沿原始网格的每个时间区间逐一推进，而是隔一个时间点前进一步：

$$
t_0 \to t_2 \to t_4 \to \cdots \to t_N,
$$

这意味着每个 $\mathtt{Student}$ 步骤实际上覆盖两个连续的 $\mathtt{Teacher}$ 步骤。因此，$\mathtt{Student}$ 走完相同的整体时间跨度 $[0,T]$ 仅需 $N/2$ 次转移。

完成这一阶段后，训练好的 $\mathtt{Student}$ 取代 $\mathtt{Teacher}$，成为新的参考模型。随后在更粗的网格上重复整个过程，时间步长加倍 $(N \to N/2 \to N/4 \to \cdots)$，逐步将轨迹蒸馏为越来越少的步骤，直到达到所需的推理步数。这种迭代减半保持整体时间范围不变，同时不断压缩生成过程的时间分辨率。

**流映射学习的统一视角。** KD 和 PD 等多种方法都可在统一的损失框架下表述：

<span id="eq-general-cm-loss"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{oracle}}({\boldsymbol{\theta}})
  := {\mathbb{E}}_{s,t} {\mathbb{E}}_{{\mathbf{x}}_s \sim p_s}
  \left[  w(s,t)  d \big({\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s, s,t),  {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)\big)\right],
\end{aligned}
\tag{10.1.4}
$$

其中 ${\boldsymbol{\Psi}}_{s\to t}$ 是 oracle 流映射，$w(s,t) \geq 0$ 指定不同时间对 $(s,t)$ 的权重，$d(\cdot,\cdot)$ 是差异度量，例如 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_2^2$ 或 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_1$，而 $p_s$ 表示时刻 $s$ 前向加噪后的边缘分布。由于 ${\boldsymbol{\Psi}}_{s\to t}$ 没有可用的闭式表达式，必须依赖近似，通常使用预训练扩散模型（教师）或其他可计算的替代。

KD 可看作[公式 (10.1.4)](01-prologue.md#eq-general-cm-loss)的一个简单实例，此时只考虑端点对 $(s,t)=(T,0)$。等价地，可以取退化的权重 $w(s,t)=\delta(s{-}T)\delta(t)$。假设 $p_T=p_{\mathrm{prior}}$[^2]，oracle 目标变为

$$
{\mathbb{E}}_{{\mathbf{x}}_T\sim p_T}
\big\|{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T,T,0)-{\boldsymbol{\Psi}}_{T\to 0}({\mathbf{x}}_T)\big\|_2^2
\approx \mathcal{L}_{\mathrm{KD}}({\boldsymbol{\theta}}),
$$

其中，实际的 KD 目标用 $\mathtt{Solver}_{T\to 0}\approx{\boldsymbol{\Psi}}_{T\to 0}$ 替代 oracle 流映射。[第 D.5 节](../../appendices/d-supplementary-proofs/05-fast-generators.md#app-sec-flow-map)给出了这一表述的另一种视角。

PD 也符合这一模板，但它并非仅用单个极端时间对 $(T,0)$ 监督，而是使用许多*相邻*的时间对，并施加一条简单的*局部一致性*规则：一个短步之后再接一个短步，应与直接跨越两步相匹配。我们将在[公式 (10.3.3)](03-progressive-distillation.md#eq-pd-local)中再次讨论这一点。

实践中的主要挑战在于 oracle 流映射 ${\boldsymbol{\Psi}}_{s\to t}$ 通常没有闭式表达式，因此无法直接监督。已有一系列方法用于高效近似这一目标，但其效果往往取决于教师模型的质量。我们将再次讨论[公式 (10.1.4)](01-prologue.md#eq-general-cm-loss)，在[第 11 章](../11-fast-generators/index.md#ch-fast-scratch)介绍一个具有理论依据的从头训练方法框架，使学习循环不再需要教师。

[^1]: 从时间上看，流映射层面的蒸馏出现得更早，包括 2021 年的 *Knowledge Distillation*（KD）（[Luhman and Luhman, 2021](../../references.md#ref-luhman2021knowledge)）与 2022 年的 *Progressive Distillation*（PD）（[Salimans and Ho, 2021](../../references.md#ref-salimans2021progressive)），早于约 2023 年出现的分布层面蒸馏方法族。不过，为使论述更流畅，并与下一章衔接，我们先介绍分布层面的蒸馏。

[^2]: 当噪声调度在 $T$ 时恰好达到先验时，这一等式成立；否则，通常只有 $p_T\approx p_{\mathrm{prior}}$，其中 $T$ 要足够大。

# 8.2 分类器引导 {#sec-cg}

## 8.2.1 分类器引导的基础 {#foundation-of-classifier-guidance}

令 ${\mathbf{c}}$ 表示从分布 $p({\mathbf{c}})$ 中抽取的条件变量，如类别标签、图像描述或其他辅助信息。目标是从 $p_0({\mathbf{x}} | {\mathbf{c}})$ 中采样。在基于扩散的条件生成中，我们通过运行时间边缘分布为 $p_t(\cdot | {\mathbf{c}})$ 的逆时间动力学来实现这一目标。其漂移依赖于条件 score

$$
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t | {\mathbf{c}}), \quad t \in [0,T].
$$

因此，一条标准且有效的途径[^1]就是估计这一量。

根据 Bayes 法则，一个基本认识是，条件 score 可以分解为：

<span id="eq-cg-bayes"></span>

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t | {\mathbf{c}})
&= \nabla_{{\mathbf{x}}_t} \log \left( \frac{p_t({\mathbf{x}}_t)   p_t({\mathbf{c}} | {\mathbf{x}}_t)}{p({\mathbf{c}})} \right)  \\
&= \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t) + \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{c}} | {\mathbf{x}}_t) - \nabla_{{\mathbf{x}}_t} \log p({\mathbf{c}})  \\
&= \underbrace{\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)}_{\text{unconditional score}}
+ \underbrace{\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{c}} | {\mathbf{x}}_t)}_{\text{classifier gradient}},
\end{aligned}
\tag{8.2.1}
$$

其中，$p_t({\mathbf{c}} | {\mathbf{x}}_t)$ 表示 $\mathbf{c}$ 在给定 $\mathbf{x}_{t}$ 时的概率，即预测条件 ${\mathbf{c}}$，所依据的是带噪输入 ${\mathbf{x}}_t$，输入对应时刻 $t$。

这一分解[^2]启发了 [Dhariwal and Nichol (2021)](../../references.md#ref-dhariwal2021diffusion) 提出的*分类器引导*（CG）方法。为区分总体恒等式与实践中学习到的分类器，先考虑 oracle 量。对于引导尺度 $\omega\geq0$，在每个固定时刻 $t$ 定义

<span id="eq-cg-density-omega"></span>

$$
\begin{aligned}
\widetilde p_t^{\mathrm{CG},\omega}({\mathbf{x}}_t|{\mathbf{c}})
\propto
p_t({\mathbf{x}}_t)p_t({\mathbf{c}}|{\mathbf{x}}_t)^\omega.
\end{aligned}
\tag{8.2.2}
$$

其 score 为

<span id="eq-cg-omega"></span>

$$
\begin{aligned}
{\mathbf{s}}^{\mathrm{CG}}({\mathbf{x}}_t,t,{\mathbf{c}};\omega)
&:=
\nabla_{{\mathbf{x}}_t}
\log\widetilde p_t^{\mathrm{CG},\omega}({\mathbf{x}}_t|{\mathbf{c}})
\\
&=
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)
+
\omega\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{c}}|{\mathbf{x}}_t).
\end{aligned}
\tag{8.2.3}
$$

因此，$\omega$ 对分类器引导方向进行缩放。当 $\omega=1$ 时，Bayes 法则给出 $\widetilde p_t^{\mathrm{CG},1}({\mathbf{x}}_t|{\mathbf{c}})=p_t({\mathbf{x}}_t|{\mathbf{c}})$，而[式 (8.2.3)](02-classifier-guidance.md#eq-cg-omega)恢复精确条件 score。对于一般 $\omega$，$\widetilde p_t^{\mathrm{CG},\omega}$ 给出引导场在固定时刻的温度缩放解释；它是否也构成引导采样器的边缘路径，将在[第 8.4.2 节](04-training-free-guidance.md#subsec-guided-marginal-path)讨论。

标量 $\omega \ge 0$ 调节分类器的影响：

-   $\omega = 1$：恢复真实条件 score $\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t |{\mathbf{c}})$。

-   $\omega > 1$：放大分类器信号，通常提高条件保真度（往往以多样性为代价）。

-   $0 \le \omega < 1$：降低分类器信号的权重，通常提高样本多样性，但削弱条件作用。

**CG 的实际近似。** 实践中，CG 是一种引导预训练无条件扩散模型的方法，相对于扩散模型本身而言无需训练，

$$
{\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t ,t) \approx \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t).
$$

CG 只在采样时应用，不修改扩散模型本身。为此，需要单独训练随时间变化的分类器 $p_{{\boldsymbol{\psi}}}({\mathbf{c}} |{\mathbf{x}}_t,t)$，使其根据带噪输入预测条件 ${\mathbf{c}}$，输入 ${\mathbf{x}}_t$ 来自不同噪声水平 $t$。分类器以标准方式训练，即最小化交叉熵损失：

<span id="eq-cg-classifier-training"></span>

$$
\begin{aligned}
\mathbb{E}_{t \sim \mathcal{U}[0,T], ({\mathbf{x}}, {\mathbf{c}}) \sim p_{\text{data}},  \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}
\Big[ -\log p_{{\boldsymbol{\psi}}}({\mathbf{c}} | {\mathbf{x}}_t,t) \Big],
\end{aligned}
\tag{8.2.4}
$$

其中，$({\mathbf{x}},{\mathbf{c}})\sim p_{\text{data}}$ 表示成对的带标签数据，${\mathbf{x}}_t = \alpha_t {\mathbf{x}} + \sigma_t \boldsymbol{\epsilon}$ 是时刻 $t$ 的带噪输入。分类器必须跨噪声水平训练，通常将 $t$ 或等价的噪声水平表示作为输入，以便在整条扩散路径上近似 $p_t({\mathbf{c}}|{\mathbf{x}}_t)$。

训练完成后，分类器提供 score，作为真实似然梯度的代理：

$$
\nabla_{{\mathbf{x}}_t} \log p_{{\boldsymbol{\psi}}^\times}({\mathbf{c}} |{\mathbf{x}}_t,t) \approx \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{c}} |{\mathbf{x}}_t).
$$

## 8.2.2 使用 CG 推断 {#inference-with-cg}

推断时，将学习到的分类器梯度 $\nabla_{{\mathbf{x}}_t}\log p_{{\boldsymbol{\psi}}^\times}({\mathbf{c}}|{\mathbf{x}}_t,t)$ 加到无条件 score 估计中，并乘以 $\omega$，从而得到[式 (8.2.3)](02-classifier-guidance.md#eq-cg-omega)中 oracle 场的学习近似：

$$
\begin{aligned}
{\mathbf{s}}^{\mathrm{CG}}_{\mathrm{model}}({\mathbf{x}}_t,t,{\mathbf{c}};\omega)
:=
\underbrace{{\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)}_{\text{uncond. direction}}
+
\omega
\underbrace{\nabla_{{\mathbf{x}}_t}\log
p_{{\boldsymbol{\psi}}^\times}({\mathbf{c}}|{\mathbf{x}}_t,t)}_{\text{guidance direction}}
\approx
{\mathbf{s}}^{\mathrm{CG}}({\mathbf{x}}_t,t,{\mathbf{c}};\omega).
\end{aligned}
$$

随后，将逆时间 SDE 或 PF-ODE 中的无条件 score 估计替换为 ${\mathbf{s}}^{\mathrm{CG}}_{\mathrm{model}}({\mathbf{x}}_t,t,{\mathbf{c}};\omega)$，使生成轨迹趋向符合条件 ${\mathbf{c}}$ 的样本。

## 8.2.3 优点与局限 {#advantages-and-limitations}

CG 为条件生成提供了简单、灵活的机制，可以通过 $\omega$ 显式控制条件作用的强度。它可用于任何预训练无条件扩散模型，只需额外增加一个用于条件控制的分类器。

不过，该方法存在明显局限：

-   **训练代价：** 分类器必须经过训练，能够在所有噪声水平下工作，计算开销较大。

-   **稳健性：** 分类器必须对受到严重扰动的输入 ${\mathbf{x}}_t$ 具有良好泛化能力，尤其在 $t$ 较大时，这可能很困难。

-   **独立训练：** 由于分类器独立于扩散模型训练，它可能无法与学习到的数据分布完全一致。

[^1]: 原则上，可以通过拒绝采样或重要性采样，从无条件生成器得到 $p_0({\mathbf{x}} | {\mathbf{c}})$，前提是 $p({\mathbf{c}} | {\mathbf{x}})$ 可获得且校准良好。但对于高维条件或罕见条件，这通常不实用。

[^2]: 最后一个恒等式中，$\nabla_{{\mathbf{x}}_t}\log p({\mathbf{c}})=0$，因为 $p({\mathbf{c}})$ 不依赖 ${\mathbf{x}}_t$。

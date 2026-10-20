# 3.3 去噪 score 匹配 {#sec-dsm}

## 3.3.1 动机 {#motivation}

尽管[式 (3.2.2)](02-score-based-models.md#eq-sm-alternative)中的替代目标

$$
\begin{aligned}
\widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol{\phi}) = \mathbb{E}_{{\mathbf{x}} \sim p_{\mathrm{data}}} \left[\mathrm{Tr}\big(\nabla_{{\mathbf{x}}}{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\big) + \frac{1}{2} \|{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\|_2^2 \right]
\end{aligned}
$$

更易计算，它仍需要计算 Jacobian 矩阵的迹 $\mathrm{Tr}(\nabla_{{\mathbf{x}}}{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}))$。虽然不必构造完整的 Jacobian 矩阵，但精确求值通常会随着输入维数 $D$ 增大而更加昂贵，因为需要汇总各输入坐标上的导数信息。因此，这在高维空间中可能需要较高的计算成本。

为解决这一问题，切片 score 匹配（[Song et al., 2020](../../references.md#ref-song2020sliced)）用基于随机投影的随机估计替换迹项。下面简要介绍这一思想。

**切片 score 匹配与 Hutchinson 估计器。** 切片 score 匹配通过沿随机“切片”的方向导数取平均，替换 score 匹配中的迹。令 ${\mathbf{u}}\in{\mathbb{R}}^D$ 为*各向同性*随机向量（例如 Rademacher 或标准 Gaussian 随机向量），满足 ${\mathbb{E}}[{\mathbf{u}}]=0$ 和 ${\mathbb{E}}[{\mathbf{u}}{\mathbf{u}}^\top]=\mathbf I$。由 Hutchinson 恒等式

$$
\operatorname{Tr}({\mathbf{A}})={\mathbb{E}}_{{\mathbf{u}}}[{\mathbf{u}}^\top {\mathbf{A}} {\mathbf{u}}],\quad\text{and}\quad{\mathbb{E}}_{{\mathbf{u}}}[({\mathbf{u}}^\top{\mathbf{s}}_{{\boldsymbol{\phi}}}({\mathbf{x}}))^2]=\|{\mathbf{s}}_{{\boldsymbol{\phi}}}({\mathbf{x}})\|_2^2,
$$

可得精确形式

$$
\widetilde{\mathcal{L}}_{\mathrm{SM}}({\boldsymbol{\phi}})
={\mathbb{E}}_{{\mathbf{x}},{\mathbf{u}}} \Big[{\mathbf{u}}^\top\big(\nabla_{{\mathbf{x}}}{\mathbf{s}}_{{\boldsymbol{\phi}}}({\mathbf{x}})\big){\mathbf{u}}+\tfrac12({\mathbf{u}}^\top{\mathbf{s}}_{{\boldsymbol{\phi}}}({\mathbf{x}}))^2\Big].
$$

这一目标可通过自动微分高效求值，使用 Jacobian–向量积与向量–Jacobian 积运算（JVP/VJP），无须显式计算大型 Jacobian 或 Hessian 矩阵。对 $K$ 个随机探测向量取平均，可得到方差为 $\mathcal{O}(1/K)$ 的无偏估计器；方向项 ${\mathbf{u}}^\top(\nabla_{{\mathbf{x}}}{\mathbf{s}}_{{\boldsymbol{\phi}}}){\mathbf{u}}$ 也可通过 JVP/VJP 例程高效计算，无须显式构造 Jacobian 矩阵。直观上，这意味着我们只沿随机方向检查模型的行为：推动投影后的 score 与更高数据密度的区域对齐，使数据点在期望意义下保持平稳。

**从切片 score 匹配到去噪 score 匹配。** 切片 score 匹配绕过了 Jacobian 矩阵，但仍依赖原始数据分布。这使其较为脆弱：对于位于低维流形上的图像数据，score $\nabla_{{\mathbf{x}}}\log p_{\mathrm{data}}({\mathbf{x}})$ 可能没有定义或不稳定，而且该方法只约束观测点处的向量场，对其邻域的控制较弱。此外，它还受到探测向量引入的方差以及重复 JVP/VJP 计算成本的影响。

我们在此重点讨论一种更稳健的替代方法：*去噪 score 匹配*（Denoising Score Matching，DSM）（[Vincent, 2011](../../references.md#ref-vincent2011connection)），它提供了具有理论依据且可扩展的解决方案。

## 3.3.2 训练 {#training}

重新考察[式 (3.2.1)](02-score-based-models.md#eq-sm)中的 SM 损失：

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}) =  \frac{1}{2} \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}) - \nabla_{{\mathbf{x}}} \log p_{\text{data}}({\mathbf{x}})\right\rVert_2^2 \right],
\end{aligned}
$$

其中的困难来自不可计算的项 $\nabla_{{\mathbf{x}}} \log p_{\text{data}}({\mathbf{x}})$。

**[Vincent (2011)](../../references.md#ref-vincent2011connection) 通过条件化给出的解决方案。** 为克服 $\nabla_{{\mathbf{x}}} \log p_{\text{data}}({\mathbf{x}})$ 的计算困难，[Vincent (2011)](../../references.md#ref-vincent2011connection) 提出通过已知条件分布向数据 ${\mathbf{x}} \sim p_{\text{data}}$ 注入噪声，该条件分布为 $p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}})$，尺度为 $\sigma$。神经网络 ${\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}, \sigma)$ 通过最小化以下损失，学习近似扰动后边缘分布的 score；该边缘分布为

$$
p_\sigma(\tilde{{\mathbf{x}}}) = \int p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}}
$$

相应的最小化损失为

<span id="eq-sm-noise-loss"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma) := \frac{1}{2} \mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}, \sigma) - \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}})\right\rVert_2^2 \right].
\end{aligned}
\tag{3.3.1}
$$

尽管 $\nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}})$ 通常不可计算，[Vincent (2011)](../../references.md#ref-vincent2011connection) 表明，以 ${\mathbf{x}} \sim p_{\text{data}}$ 为*条件*可得到一个等价、可计算的目标，即*去噪 score 匹配*（DSM）损失：

<span id="eq-dsm-loss"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma) := \frac{1}{2} \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}, \tilde{{\mathbf{x}}} \sim p_{\sigma}(\cdot|{\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}, \sigma) - \nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}})\right\rVert_2^2 \right].
\end{aligned}
\tag{3.3.2}
$$

[式 (3.3.2)](03-denoising-score-matching.md#eq-dsm-loss)的最优解 ${\mathbf{s}}^*$ 满足

$$
{\mathbf{s}}^*(\tilde{{\mathbf{x}}}, \sigma) = \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}),
$$

它同时也是[式 (3.3.1)](03-denoising-score-matching.md#eq-sm-noise-loss)的最优解。

例如，当 $p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}})$ 是方差为 $\sigma^2$ 的 Gaussian 噪声时，

$$
p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}}) = \mathcal{N}(\tilde{{\mathbf{x}}}; {\mathbf{x}}, \sigma^2 \mathbf{I}),
$$

梯度 $\nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}})$ 具有闭式表达式（见[式 (3.3.4)](03-denoising-score-matching.md#eq-gaussian-dsm)），从而使回归目标显式且在计算上可行。

此外，当 $\sigma \approx 0$ 时，$p_\sigma(\tilde{{\mathbf{x}}}) \approx p_{\text{data}}({\mathbf{x}})$，且

$$
{\mathbf{s}}^*(\tilde{{\mathbf{x}}}, \sigma) = \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}) \approx \nabla_{{\mathbf{x}}} \log p_{\text{data}}({\mathbf{x}}),
$$

这表明学得的 score 近似原始数据 score，因此可用于生成。

下面的定理形式化地表述了上述关于 $\mathcal{L}_{\text{SM}}$ 与 $\mathcal{L}_{\text{DSM}}$ 梯度等价的讨论：

<span id="thm-sm-dsm"></span>

**定理 3.3.1：$\mathcal{L}_{\text{SM}}$ 与 $\mathcal{L}_{\text{DSM}}$ 的等价性**

对任意固定的噪声尺度 $\sigma > 0$，下式成立：

<span id="eq-score-matching"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma) = \mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma) + C,
\end{aligned}
\tag{3.3.3}
$$

其中 $C$ 是与参数 $\boldsymbol{\phi}$ 无关的常数。此外，两个损失的最优解 ${\mathbf{s}}^*(\cdot, \sigma)$ 满足

$$
{\mathbf{s}}^*(\tilde{{\mathbf{x}}}, \sigma) = \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}), \quad \text{for almost every } \tilde{{\mathbf{x}}}.
$$

**定理证明。**

等价性可由直接计算得到：展开 $\mathcal{L}_{\text{SM}}$ 与 $\mathcal{L}_{\text{DSM}}$ 中的 MSE 后，所有依赖 $\boldsymbol{\phi}$ 的项都相互抵消，只留下一个与 $\boldsymbol{\phi}$ 无关的常数差。<br>
最优解的推导与[命题 4.3.1](../04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer)采用相同论证。详细推导放在[第 D.2.2 节](../../appendices/d-supplementary-proofs/02-score-based-perspective.md#app-sec-sm-dsm)。

与 DDPM 中的[定理 2.2.1](../02-variational-perspective/02-ddpm.md#thm-equiv-marginal-kl)一样，这一定理体现了一个关键的共同原则：

<span id="insight-3-3-1"></span>

**启示 3.3.1：条件化技巧**

条件化技巧也出现在 DDPM 的扩散模型变分视角中（见[定理 2.2.1](../02-variational-perspective/02-ddpm.md#thm-equiv-marginal-kl)）：以数据点 ${\mathbf{x}}$ 为条件，将不可计算的损失转化为可通过 Monte Carlo 估计计算的损失。类似思想也出现在基于流的视角中，例如流匹配（[Lipman et al., 2022](../../references.md#ref-lipman2022flow)），我们将在[第 5.2 节](../05-flow-based-perspective/02-flow-matching-framework.md#sec-flow-matching-framework)看到这一点。

<span id="fig-dsm-graph"></span>

![Gaussian 扰动给出可计算的条件 score 目标](../../../en/assets/chapter-03/denoising-score-matching.png)

**图 3.5：** **通过条件化技巧实现 DSM 的示意图。** 对数据分布 $p_{\mathrm{data}}$ 施加小幅加性 Gaussian 噪声 $\mathcal{N}(\mathbf{0}, \sigma^2 {\mathbf{I}})$ 扰动，所得条件分布 $p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}}) = \mathcal{N}(\tilde{{\mathbf{x}}}; {\mathbf{x}}, \sigma^2 {\mathbf{I}})$ 具有闭式 score 函数。

来源：作者绘制。

**特殊情形：加性 Gaussian 噪声。** 现在考虑常见的情形：将 Gaussian 噪声 $\mathcal{N}(\mathbf{0}, \sigma^2 {\mathbf{I}})$（方差为 $\sigma^2$）加到每个数据点 ${\mathbf{x}} \sim p_{\text{data}}$ 上：

$$
\tilde{{\mathbf{x}}} = {\mathbf{x}} + \sigma \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, {\mathbf{I}}),
$$

于是，扰动后的数据 $\tilde{{\mathbf{x}}}$ 服从

$$
p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}}) = \mathcal{N}(\tilde{{\mathbf{x}}}; {\mathbf{x}}, \sigma^2 {\mathbf{I}}).
$$

此时，条件 score 的解析形式为

$$
\nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}}) = \frac{{\mathbf{x}} - \tilde{{\mathbf{x}}}}{\sigma^2}.
$$

因此，DSM 损失简化为：

<span id="eq-gaussian-dsm"></span>

$$
\begin{aligned}
    \mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma) &= \frac{1}{2} \mathbb{E}_{{\mathbf{x}}, \tilde{{\mathbf{x}}}|{\mathbf{x}}} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}, \sigma) - \frac{{\mathbf{x}} - \tilde{{\mathbf{x}}}}{\sigma^2}\right\rVert_2^2 \right] \\
    &= \frac{1}{2} \mathbb{E}_{{\mathbf{x}}, \boldsymbol{\epsilon}} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}} + \sigma \boldsymbol{\epsilon}, \sigma) + \frac{\boldsymbol{\epsilon}}{\sigma}\right\rVert_2^2 \right],
\end{aligned}
\tag{3.3.4}
$$

其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, {\mathbf{I}})$。这一目标构成了基于 score 的扩散模型的核心。

在固定噪声水平 $\sigma>0$ 下，DSM 学习的是经过 Gaussian 平滑的分布

$$
p_\sigma
=
p_{\mathrm{data}} * \mathcal{N}(\boldsymbol{0},\sigma^2{\mathbf{I}}),
$$

的 score，而非 $p_{\mathrm{data}}$ 本身的 score。当 $\sigma$ 较小时，扰动较轻，因此 $p_\sigma$ 仍较好地保留数据分布的局部结构。它的 score 因而为将含噪样本移回可能出现数据的区域提供了有用方向。随着 $\sigma$ 增大，Gaussian 平滑逐渐去除细节，并合并相近的模式。此时，对应的 score 主要捕捉较粗粒度、更全局的结构。这种行为具有实际用途：[第 3.4 节](04-noise-conditional-score-networks.md#sec-smld)的多噪声构造有意结合许多噪声水平，以大噪声进行粗粒度探索，以小噪声进行精细调整。

为更清楚地说明这一目标为什么自然对应“去噪”过程，我们将在[第 3.3.4 节](03-denoising-score-matching.md#subsec-why-dsm-denoising-tweedie)与[第 3.3.5 节](03-denoising-score-matching.md#subsec-why-dsm-denoising-sure)进一步讨论。

## 3.3.3 采样 {#sampling}

获得训练好的 score 模型 ${\mathbf{s}}_{\boldsymbol{\phi}^\times}(\tilde{{\mathbf{x}}}, \sigma)$（对应噪声水平 $\sigma$）后，我们使用 Langevin 动力学生成样本，以学得的模型替换真实 score。更新规则为：

<span id="eq-score-langevin-emp-dsm"></span>

$$
\begin{aligned}
\tilde{{\mathbf{x}}}_{n+1} = \tilde{{\mathbf{x}}}_n + \eta \underbrace{{\mathbf{s}}_{\boldsymbol{\phi}^\times}(\tilde{{\mathbf{x}}}_n, \sigma)}_{\approx \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}_n)} + \sqrt{2\eta}    \boldsymbol{\epsilon}_n, \quad \boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, {\mathbf{I}}),
\end{aligned}
\tag{3.3.5}
$$

其中 $n = 0, 1, 2, \dots$，从初值 $\tilde{{\mathbf{x}}}_0$ 开始。若 $\sigma$ 足够小，则经过足够多次迭代后，$\tilde{{\mathbf{x}}}_n$ 近似为来自 $p_{\text{data}}$ 的样本。

**注入噪声的优势。** 我们还指出，与[式 (3.2.1)](02-score-based-models.md#eq-sm)中的基本 score 匹配相比，注入 Gaussian 噪声以形成 $p_\sigma$（例如[式 (3.3.4)](03-denoising-score-matching.md#eq-gaussian-dsm)）具有两个关键优势（[Song and Ermon, 2019](../../references.md#ref-song2019generative)）：

-   **定义良好的梯度。** 噪声将数据扰动到其低维流形之外，得到的分布 $p_\sigma$ 在 $\mathbb{R}^D$ 中具有全支撑。因此，score 函数 $\nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}})$ 处处都有良好定义。

-   **更好的覆盖。** 噪声平滑了模式之间的稀疏区域，提高训练信号质量，并使 Langevin 动力学能够更有效地穿越低密度区域。

## 3.3.4 为什么 DSM 是去噪：Tweedie 公式 {#subsec-why-dsm-denoising-tweedie}

我们从 *Tweedie 公式*（[Efron, 2011](../../references.md#ref-efron2011tweedie)）开始，它为仅根据含噪观测进行去噪提供了理论依据。具体而言，它表明：给定单个 Gaussian 扰动观测 $\tilde{{\mathbf{x}}}\sim\mathcal N(\,\cdot\,;\alpha{\mathbf{x}},\sigma^2\mathrm{I})$，其来自未知的 ${\mathbf{x}} \sim p_{\mathrm{data}}$，要得到去噪估计（给定 $\tilde{{\mathbf{x}}}$ 时，所有可能的干净信号的平均值），只需将 $\tilde{{\mathbf{x}}}$ 移动步长 $\sigma^2$，方向取为其含噪边缘分布的 score $\nabla_{\tilde{{\mathbf{x}}}}\log p_\sigma(\tilde{{\mathbf{x}}})$；该含噪边缘分布定义为：

$$
p_\sigma(\tilde{{\mathbf{x}}}):= \int \mathcal{N}(\tilde{{\mathbf{x}}};\alpha {\mathbf{x}},\sigma^2\mathrm{I})p_{\mathrm{data}}({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

下面正式陈述这一命题。

<span id="tweedie"></span>

**引理 3.3.2：Tweedie 公式**

假设 ${\mathbf{x}}\sim p_{\mathrm{data}}$，且在给定 ${\mathbf{x}}$ 的条件下，$\tilde{{\mathbf{x}}}\sim\mathcal N(\,\cdot\,;\alpha{\mathbf{x}},\sigma^2\mathrm{I})$，其中 $\alpha\neq0$。则 Tweedie 公式给出

<span id="eq-tweedie"></span>

$$
\begin{aligned}
\alpha   \mathbb{E}_{{\mathbf{x}} \sim p({\mathbf{x}} |\tilde{\mathbf{x}})}\big[{\mathbf{x}}  \big|  \tilde{\mathbf{x}}\big] = \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}),
\end{aligned}
\tag{3.3.6}
$$

其中，期望对后验分布 $p({\mathbf{x}} |\tilde{\mathbf{x}})$ 求取，即 ${\mathbf{x}}$ 在给定 $\tilde{\mathbf{x}}$ 时的后验分布。

**引理证明。**

证明从计算以下边缘分布的 score 开始：

$$
p(\tilde{\mathbf{x}})=\int p(\tilde{\mathbf{x}}|{\mathbf{x}})p_{\mathrm{data}}({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

在积分号内求导，并利用条件密度的 Gaussian 形式，可直接得到一个表达式，整理后即为连接 score 与后验均值的所需恒等式。详见[第 D.2.3 节](../../appendices/d-supplementary-proofs/02-score-based-perspective.md#app-sec-tweedie)。

Tweedie 公式在扩散模型中起着核心作用，这些模型像 DDPM 一样引入多层噪声。它使我们能够通过 score 函数从含噪观测估计干净样本，从而建立 score 预测与去噪器之间的基本联系：

$$
\underbrace{\mathbb{E} \left[{\mathbf{x}}|\tilde{\mathbf{x}}\right]}_{\substack{\text{denoiser}\\\text{estimated from }\tilde{\mathbf{x}}}}
= \frac{1}{\alpha}\left(\tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})\right).
$$

特别地，对含噪对数似然执行一次步长为 $\sigma^2$ 的梯度上升，就能得到去噪估计（干净信号的条件平均值）。这使 DSM 训练与去噪紧密相关：若通过 DSM 训练得到 ${\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}})\approx \nabla_{\tilde{{\mathbf{x}}}}\log p_\sigma(\tilde{{\mathbf{x}}})$，则

$$
\frac{1}{\alpha}\left(\tilde{{\mathbf{x}}}+\sigma^2 {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}})\right)
$$

就是去噪器。

**（选读）高阶 Tweedie 公式。** 经典 Tweedie 公式将后验均值 ${\mathbb{E}}[{\mathbf{x}}_0|\tilde{{\mathbf{x}}}]$ 表示为梯度 $\nabla_{\tilde{{\mathbf{x}}}}\log p(\tilde{{\mathbf{x}}})$ 的函数。高阶扩展（[Meng et al., 2021](../../references.md#ref-meng2021estimating)）通过 $\log p(\tilde{{\mathbf{x}}})$ 的高阶导数表示后验协方差及更高阶累积量。

**具有对数归一化项 $\lambda(\tilde{{\mathbf{x}}})$ 的指数族设定。** 假设 $\tilde{{\mathbf{x}}}$ 在给定潜在自然参数 $\boldsymbol\eta\in{\mathbb{R}}^D$ 时的条件分布属于自然指数族，写作

$$
q_\sigma(\tilde{{\mathbf{x}}}|\boldsymbol\eta)
=\exp\!\big(\boldsymbol\eta^\top \tilde{{\mathbf{x}}}-\psi(\boldsymbol\eta)\big)\,q_0(\tilde{{\mathbf{x}}}).
$$

这里 $q_0(\tilde{{\mathbf{x}}})$ 是*基测度*，即不依赖于 $\boldsymbol\eta$ 的部分；对于方差为 $\sigma^2{\mathbf{I}}$ 的加性 Gaussian 噪声，它等于 $(2\pi\sigma^2)^{-D/2}\exp(-\|\tilde{{\mathbf{x}}}\|^2/2\sigma^2)$。令 $p(\boldsymbol\eta)$ 为潜在自然参数的预设分布，可视为重参数化后的干净数据分布（对于 Gaussian 位置模型，$\boldsymbol\eta={\mathbf{x}}/\sigma^2$）。观测到的含噪边缘分布为

$$
p_\sigma(\tilde{{\mathbf{x}}})=\int q_\sigma(\tilde{{\mathbf{x}}}|\boldsymbol\eta)\,p(\boldsymbol\eta)\mathop{}\!\mathrm{d}\boldsymbol\eta.
$$

将关于 $\tilde{{\mathbf{x}}}$ 的*对数配分函数*（对数归一化项）定义为

$$
\lambda(\tilde{{\mathbf{x}}}) := \log p_\sigma(\tilde{{\mathbf{x}}})-\log q_0(\tilde{{\mathbf{x}}}).
$$

于是，$\boldsymbol\eta$ 在给定 $\tilde{{\mathbf{x}}}$ 时的后验分布为

$$
p(\boldsymbol\eta|\tilde{{\mathbf{x}}})
\ \propto\
\exp\!\big(\boldsymbol\eta^\top \tilde{{\mathbf{x}}}-\psi(\boldsymbol\eta)-\lambda(\tilde{{\mathbf{x}}})\big)\,p(\boldsymbol\eta),
$$

这表明，作为 $\tilde{{\mathbf{x}}}$ 的函数，后验分布具有指数族形式，其自然参数为 $\tilde{{\mathbf{x}}}$，充分统计量为 $\boldsymbol\eta$，对数配分函数为 $\lambda(\tilde{{\mathbf{x}}})$。

**$\lambda$ 的导数给出后验累积量。** 这里用到两条简单规则。第一是归一化：对每个 $\tilde{{\mathbf{x}}}$，

$$
\int \exp\!\big(\boldsymbol\eta^\top \tilde{{\mathbf{x}}}-\psi(\boldsymbol\eta)-\lambda(\tilde{{\mathbf{x}}})\big)\,p(\boldsymbol\eta)\mathop{}\!\mathrm{d}\boldsymbol\eta
=1.
$$

对这一恒等式关于 $\tilde{{\mathbf{x}}}$ 求导，会从指数项中得到 $\boldsymbol\eta$ 的幂以及 $\lambda(\tilde{{\mathbf{x}}})$ 的导数；令结果为零，就得到 $\lambda$ 的导数与 $\boldsymbol\eta$ 的后验矩之间的等式。第二是指数族的一条标准性质：对数配分函数是充分统计量的累积量生成函数。因此，

$$
\nabla_{\tilde{{\mathbf{x}}}}\lambda(\tilde{{\mathbf{x}}})={\mathbb{E}}[\boldsymbol\eta|\tilde{{\mathbf{x}}}],\quad
\nabla_{\tilde{{\mathbf{x}}}}^{2}\lambda(\tilde{{\mathbf{x}}})=\mathrm{Cov}[\boldsymbol\eta|\tilde{{\mathbf{x}}}],\quad
\nabla_{\tilde{{\mathbf{x}}}}^{(k)}\lambda(\tilde{{\mathbf{x}}})=\kappa_k(\boldsymbol\eta|\tilde{{\mathbf{x}}})\quad(k\ge 3),
$$

其中，$\kappa_k$ 是 $k$ 阶*条件累积量*，对应随机向量 $\boldsymbol\eta$ 在给定 $\tilde{{\mathbf{x}}}$ 时的条件分布，可由标准的矩与累积量关系得到。

这些就是高阶 Tweedie 公式。对于满足 $\boldsymbol\eta={\mathbf{x}}/\sigma^2$ 的 Gaussian 位置模型，代入后得到以 $\log p_\sigma(\tilde{{\mathbf{x}}})$ 的导数表示的熟悉形式：

$$
{\mathbb{E}}[{\mathbf{x}}|\tilde{{\mathbf{x}}}]
=\tilde{{\mathbf{x}}}+\sigma^2\nabla_{\tilde{{\mathbf{x}}}}\log p_\sigma(\tilde{{\mathbf{x}}}),
\qquad
\mathrm{Cov}[{\mathbf{x}}|\tilde{{\mathbf{x}}}]
=\sigma^2\mathrm I+\sigma^4\nabla_{\tilde{{\mathbf{x}}}}^{2}\log p_\sigma(\tilde{{\mathbf{x}}}),
$$

更高阶累积量则随 $\log p_\sigma(\tilde{{\mathbf{x}}})$ 的更高阶导数按相应尺度变化。

一些研究探索了通过训练神经网络来估计高阶 score（[Meng et al., 2021](../../references.md#ref-meng2021estimating)；[Lu et al., 2022](../../references.md#ref-lu2022maximum)；[Lai et al., 2023](../../references.md#ref-lai2023fp)）。我们在此旨在阐明它们与统计量之间的关系，具体方法请读者参阅这些工作。

## 3.3.5 （选读）为什么 DSM 是去噪：SURE {#subsec-why-dsm-denoising-sure}

**SURE（Stein 无偏风险估计器）。** 概括而言，Stein 无偏风险估计器（SURE）使我们能够在*不知道干净信号*的情况下，估计去噪器 ${\mathbf{D}}$ 的均方误差（MSE）。换言之，当只有含噪数据时，SURE 提供了一种选择或训练去噪器的方法。

为清楚起见，考虑加性 Gaussian 噪声的情形：

$$
\tilde{{\mathbf{x}}} = {\mathbf{x}} + \sigma \boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},{\mathbf{I}}),
$$

其中 ${\mathbf{x}} \in {\mathbb{R}}^D$ 是未知的干净信号，$\tilde{{\mathbf{x}}}$ 是观测到的含噪版本。去噪器是任意弱可微映射 ${\mathbf{D}}:{\mathbb{R}}^D \to {\mathbb{R}}^D$，它输出估计 ${\mathbf{D}}(\tilde{{\mathbf{x}}})$，以估计 ${\mathbf{x}}$。

自然的质量度量是条件 MSE

$$
R({\mathbf{D}};{\mathbf{x}}) := {\mathbb{E}}_{\tilde{{\mathbf{x}}}|{\mathbf{x}}}\left[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-{\mathbf{x}}\|_2^2 \,\big|\, {\mathbf{x}}\right].
$$

这个量依赖未知真值 ${\mathbf{x}}$，因此无法直接计算。然而，Stein 恒等式（见[第 D.2.4 节](../../appendices/d-supplementary-proofs/02-score-based-perspective.md#app-sec-stein-identity)）给出了以下*可观测*的替代量：

<span id="eq-sure-obj"></span>

$$
\begin{aligned}
\mathrm{SURE}({\mathbf{D}};\tilde{{\mathbf{x}}})
= \|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|_2^2
+ 2\sigma^2\,\nabla_{\tilde{{\mathbf{x}}}}\cdot {\mathbf{D}}(\tilde{{\mathbf{x}}})
- D\sigma^2,
\end{aligned}
\tag{3.3.7}
$$

其中 $\nabla_{\tilde{{\mathbf{x}}}}\cdot {\mathbf{D}}(\tilde{{\mathbf{x}}})$ 表示 ${\mathbf{D}}$ 的散度。我们强调，$\mathrm{SURE}({\mathbf{D}};\tilde{{\mathbf{x}}})$ 只需要含噪观测 $\tilde{\mathbf{x}}$，无须干净的 ${\mathbf{x}}$。

直观上，$\mathrm{SURE}$ 由两个互补的部分组成。项 $\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|^2$ 衡量去噪器输出与含噪输入的距离；单靠这一项会低估真实误差，因为 $\tilde{{\mathbf{x}}}$ 已被扰动。散度项起到修正作用：它捕捉去噪器对输入微小扰动的敏感程度，从而有效计入噪声引入的方差。

对于任意固定但未知的 ${\mathbf{x}}$，有

$$
{\mathbb{E}}_{\tilde{{\mathbf{x}}}|{\mathbf{x}}}\left[\mathrm{SURE}({\mathbf{D}};{\mathbf{x}}+\sigma{\boldsymbol{\epsilon}}) \,\big|\, {\mathbf{x}}\right]
= R({\mathbf{D}};{\mathbf{x}}),
$$

其中期望对 Gaussian 噪声 ${\boldsymbol{\epsilon}}\sim\mathcal{N}(\boldsymbol{0},{\mathbf{I}})$ 求取。因此，最小化 SURE（在期望意义下或通过经验估计）等价于最小化真实 MSE，同时只依赖含噪数据。实践中，将 $\mathrm{SURE}$ 同时对 ${\mathbf{x}}\sim p_{\mathrm{data}}$ 和扰动噪声 ${\boldsymbol{\epsilon}}$ 取平均，可得到全局 MSE 风险的无偏估计。

**与 Tweedie 公式及 Bayes 最优性的联系。** 令 $p_\sigma(\tilde{{\mathbf{x}}}) = \big(p_{\mathrm{data}} * \mathcal{N}(\mathbf{0}, \sigma^2 {\mathbf{I}})\big)(\tilde{{\mathbf{x}}})$ 表示本节考虑的含噪边缘分布。

在给定 ${\mathbf{x}}$ 的条件下，SURE 是对噪声取期望的均方误差的无偏估计器：

$$
\mathbb{E}_{\tilde{{\mathbf{x}}}| {\mathbf{x}}}\!\big[\mathrm{SURE}({\mathbf{D}}; \tilde{{\mathbf{x}}})\big]
=
\mathbb{E}_{\tilde{{\mathbf{x}}}| {\mathbf{x}}}\!\big[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-{\mathbf{x}}\|^2\big].
$$

因此，由全期望公式（塔式性质），最小化期望 SURE 等同于最小化 Bayes 风险 <span class="book-inline-equation" tabindex="0">$\mathbb{E}_{({\mathbf{x}}, \tilde{{\mathbf{x}}})}\big[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-{\mathbf{x}}\|^2\big] =\mathbb{E}_{\tilde{{\mathbf{x}}}}\!\big[\mathbb{E}_{{\mathbf{x}}| \tilde{{\mathbf{x}}}}\!\big[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-{\mathbf{x}}\|^2\big]\big]$</span>。这一分解得到逐点优化：对几乎处处的 $\tilde{{\mathbf{x}}}$，

$$
{\mathbf{D}}^*(\tilde{{\mathbf{x}}})
=\arg\min_{\mathbf z}\ \mathbb{E}_{{\mathbf{x}}| \tilde{{\mathbf{x}}}}\!\big[\|\mathbf z-{\mathbf{x}}\|^2\big]
=\mathbb{E}[{\mathbf{x}}| \tilde{{\mathbf{x}}}].
$$

因此，使 SURE 最优的去噪器与[第 3.3.4 节](03-denoising-score-matching.md#subsec-why-dsm-denoising-tweedie)中的 Bayes 估计器一致，并且由 Tweedie 恒等式，

<span id="eq-sure-tweedie-denoiser"></span>

$$
\begin{aligned}
{\mathbf{D}}^*(\tilde{{\mathbf{x}}})=\mathbb{E}[{\mathbf{x}}| \tilde{{\mathbf{x}}}]=\tilde{{\mathbf{x}}}+\sigma^2\nabla_{\tilde{{\mathbf{x}}}}\log p_\sigma(\tilde{{\mathbf{x}}}).
\end{aligned}
\tag{3.3.8}
$$

**SURE 与 score 匹配的关系。** [式 (3.3.8)](03-denoising-score-matching.md#eq-sure-tweedie-denoiser)中的恒等式启发我们通过 score 场来参数化去噪器 ${\mathbf{D}}$：

$$
{\mathbf{D}}(\tilde{{\mathbf{x}}})=\tilde{{\mathbf{x}}}+\sigma^2{\mathbf{s}}_{\boldsymbol\phi}(\tilde{{\mathbf{x}}}, \sigma),
$$

其中 ${\mathbf{s}}_{\boldsymbol\phi}(\cdot, \sigma)$ 用于近似含噪 score $\nabla_{\tilde{{\mathbf{x}}}}\log p_\sigma(\cdot)$。将 ${\mathbf{D}}(\tilde{{\mathbf{x}}})=\tilde{{\mathbf{x}}}+\sigma^2 {\mathbf{s}}_{\boldsymbol\phi}(\tilde{{\mathbf{x}}}, \sigma)$ 代入[式 (3.3.7)](03-denoising-score-matching.md#eq-sure-obj)，得到

$$
\frac{1}{2\sigma^4}\,\mathrm{SURE}({\mathbf{D}};\tilde{{\mathbf{x}}})
=
\operatorname{Tr}\!\big(\nabla_{\tilde{{\mathbf{x}}}} s_{\boldsymbol\phi}(\tilde{{\mathbf{x}}}, \sigma)\big)
+\tfrac12\|{\mathbf{s}}_{\boldsymbol\phi}(\tilde{{\mathbf{x}}}, \sigma)\|_2^2
+\text{const}(\sigma).
$$

因此，对 $\tilde{{\mathbf{x}}}\sim p_\sigma$ 取期望时，最小化 SURE 等价于最小化噪声水平 $\sigma$ 下 Hyvärinen 的替代 score 匹配目标（相差一个加性常数），其中期望在 $p_\sigma$ 下求取（见[式 (3.2.2)](02-score-based-models.md#eq-sm-alternative)）。因此，两个目标具有相同的最优解，即[式 (3.3.8)](03-denoising-score-matching.md#eq-sure-tweedie-denoiser)中的去噪器。

## 3.3.6 （选读）广义 score 匹配 {#generalized-score-matching}

**动机。** 经典 score 匹配、去噪 score 匹配及其高阶变体的目标都是

$$
\frac{\mathcal L p({\mathbf{x}})}{p({\mathbf{x}})},\quad\text{for some density } p
$$

其中线性算子 $\mathcal L$ 作用于密度。在经典情形 $\mathcal L=\nabla_{{\mathbf{x}}}$ 下，得到

$$
\nabla_{{\mathbf{x}}}\log p({\mathbf{x}}) = \frac{\nabla_{{\mathbf{x}}} p({\mathbf{x}})}{p({\mathbf{x}})}
$$

$\frac{\mathcal L p}{p}$ 这一结构允许通过分部积分去除归一化常数，从而得到只依赖 $p$ 的样本和学得的场 ${\mathbf{s}}_{\boldsymbol{\phi}}$ 的可计算目标。这一视角引出了广义 score 匹配框架。

**广义 Fisher 散度。** 令 $p$ 为数据分布，$q$ 为任意模型分布。对于线性算子 $\mathcal L$，它作用于 ${\mathbf{x}}$ 的标量函数，定义广义 Fisher 散度为

$$
\mathcal D_{\mathcal L}(p \,\|\, q)
:= \int p({\mathbf{x}})
\left\|
\frac{\mathcal L p({\mathbf{x}})}{p({\mathbf{x}})}
-
\frac{\mathcal L q({\mathbf{x}})}{q({\mathbf{x}})}
\right\|_2^2 \mathop{}\!\mathrm{d} {\mathbf{x}}.
$$

若 $\mathcal L$ 是*完备的*，即

$$
\frac{\mathcal L p_1}{p_1}=\frac{\mathcal L p_2}{p_2} \text{ a.e.}\quad \text{implies}\quad p_1=p_2 \text{ a.e.,}
$$

则 $\mathcal D_{\mathcal L}(p \,\|\, q)=0$ 能确定 $q=p$。当 $\mathcal L=\nabla_{\tilde{\mathbf{x}}}$ 时，这就回到经典 Fisher 散度（见[式 (1.1.3)](../01-deep-generative-modeling/01-what-is-deep-generative-modeling.md#eq-fisher)）。

**score 参数化。** 实践中，我们不对归一化密度 $q$ 建模，而是直接参数化向量场 ${\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})$ 来近似广义 score $\frac{\mathcal L p({\mathbf{x}})}{p({\mathbf{x}})}$。考虑

$$
\mathcal D_{\mathcal L}\!\left(p \,\|\, {\mathbf{s}}_{\boldsymbol{\phi}}\right)
:= \mathbb E_{{\mathbf{x}}\sim p}\!\left[\left\|\,{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}) - \frac{\mathcal L p({\mathbf{x}})}{p({\mathbf{x}})}\right\|_2^2\right].
$$

尽管 $\frac{\mathcal L p({\mathbf{x}})}{p({\mathbf{x}})}$ 未知，“分部积分”可使损失只依赖于 ${\mathbf{s}}_{\boldsymbol{\phi}}$。令 $\mathcal L^\dagger$ 为 $\mathcal L$ 的伴随算子，定义为

$$
\int \big(\mathcal L f\big)^\top g
= \int f\,(\mathcal L^\dagger g)
\quad\text{for all test functions } f,g,
$$

当边界项消失时，它在形式上将 $\mathcal L$ “移到”积分的另一侧。展开平方并应用这一恒等式，得到可计算的目标

$$
\mathcal L_{\text{GSM}}({\boldsymbol{\phi}})
= \mathbb E_{{\mathbf{x}}\sim p}\!\left[
\frac{1}{2}\,\|{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\|_2^2
-
\big(\mathcal L^\dagger {\mathbf{s}}_{\boldsymbol{\phi}}\big)({\mathbf{x}})
\right] + \text{const},
$$

其中常数与 ${\boldsymbol{\phi}}$ 无关。我们仅通过期望使用 $p$，因此，广义 score 匹配损失可以像经典 score 匹配一样，通过训练数据进行经验估计。

对于 $\mathcal L=\nabla$，有 $\mathcal L^\dagger=-\nabla\!\,\cdot$，这就还原了[式 (3.2.2)](02-score-based-models.md#eq-sm-alternative)中 Hyvärinen 的 score 匹配目标 $\mathbb E_p\!\big[\tfrac{1}{2}\|{\mathbf{s}}_{\boldsymbol{\phi}}\|_2^2 + \nabla\!\cdot {\mathbf{s}}_{\boldsymbol{\phi}}\big]$。

**算子示例。**

-   **经典 score 匹配。** 考虑 $\mathcal L = \nabla_{{\mathbf{x}}}$。此时，广义 score 退化为经典 score 函数

    $$
    \frac{\mathcal L p({\mathbf{x}})}{p({\mathbf{x}})}
    = \nabla_{{\mathbf{x}}}\log p({\mathbf{x}}).
    $$

-   **去噪 score 匹配。** 对于加性 Gaussian 噪声，定义算子

    $$
    (\mathcal L f)(\tilde{{\mathbf{x}}})
    = \tilde{{\mathbf{x}}}\,f(\tilde{{\mathbf{x}}}) + \sigma^2\nabla_{\tilde{{\mathbf{x}}}} f(\tilde{{\mathbf{x}}}).
    $$

    则

    $$
    \frac{\mathcal L p_\sigma(\tilde{{\mathbf{x}}})}{p_\sigma(\tilde{{\mathbf{x}}})}
    = \tilde{{\mathbf{x}}} + \sigma^2\nabla_{\tilde{{\mathbf{x}}}}\log p_\sigma(\tilde{{\mathbf{x}}})
    = \mathbb E[{\mathbf{x}}_0|\tilde{{\mathbf{x}}}],
    $$

    其中

    $$
    p_\sigma(\tilde{\mathbf{x}})
    :=
    \int
    \mathcal{N}(\tilde{\mathbf{x}};{\mathbf{x}}_0,\sigma^2{\mathbf{I}})
    \,p_{\mathrm{data}}({\mathbf{x}}_0)\,\mathop{}\!\mathrm{d}{\mathbf{x}}_0,
    \qquad
    \tilde{\mathbf{x}}={\mathbf{x}}_0+\sigma{\boldsymbol{\epsilon}}.
    $$

    这恰好是加性 Gaussian 噪声下的 Tweedie 恒等式。因此，使用这一算子最小化 $\mathcal{L}_{\mathrm{GSM}}$，就是训练 ${\mathbf{s}}_{{\boldsymbol{\phi}}}$ 来近似后验均值去噪器。若改用缩放后的 Gaussian 位置模型 $\tilde{\mathbf{x}}=\alpha{\mathbf{x}}_0+\sigma{\boldsymbol{\epsilon}}$，对应恒等式会包含[式 (3.3.6)](03-denoising-score-matching.md#eq-tweedie)中的因子 $\alpha$。

-   **高阶目标。** 在 $\mathcal L$ 中堆叠导数，可以得到 $\nabla^2\log p$ 及更高阶导数，它们对应后验协方差与更高阶累积量。

**扩展与应用。** 广义 score 匹配从连续变量扩展到了离散情形，包括语言建模（[Meng et al., 2022](../../references.md#ref-meng2022concrete)；[Lou et al., 2024](../../references.md#ref-lou2024discrete)）。它还启发了基于 score 思想的训练，得到具有去噪形式的目标。这种算子视角统一了一系列目标，允许从数据进行经验估计，并通过恰当地选择 $\mathcal L$，为设计损失函数提供一般原则。
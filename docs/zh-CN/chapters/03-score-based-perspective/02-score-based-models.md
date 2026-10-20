# 3.2 从基于能量的生成模型到基于 score 的生成模型 {#sec-ebm-to-score}

EBM 表明，生成过程只依赖于指向更高概率区域的 score，无须完整的归一化密度。虽然 score 匹配避免了配分函数的计算，但通过能量进行训练仍然需要计算代价高昂的二阶导数。关键思路在于，既然使用 Langevin 动力学采样只需要 score，我们就可以用神经网络直接学习它。从对能量建模转向对 score 建模，构成了基于 score 的生成模型的基础。

<span id="fig-sm-illustration"></span>

![神经网络近似真实的 score 向量场](../../../en/assets/chapter-03/score-matching.png)

**图 3.4：** **score 匹配示意图。** 通过均方误差（MSE）损失训练神经网络 score ${\color{#5D94D1}{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})}$，使其匹配真实的 score ${\mathbf{s}}({\mathbf{x}})$。两者均表示为向量场。

来源：由作者绘制。

## 3.2.1 使用 score 匹配进行训练 {#subsec-training-sm}

**score 匹配（Score Matching，SM）。** 为了近似 score 函数 ${\mathbf{s}}({\mathbf{x}}) = \nabla_{{\mathbf{x}}} \log p_{\mathrm{data}}({\mathbf{x}})$，我们使用 $p_{\mathrm{data}}$ 的样本，直接以由神经网络参数化的向量场 ${\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})$ 来近似它（见[图 3.4](02-score-based-models.md#fig-sm-illustration)）：

$$
{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}) \approx {\mathbf{s}}({\mathbf{x}}).
$$

*score 匹配*通过最小化真实 score 与估计 score 之间的均方误差（MSE）来拟合这个向量场：

<span id="eq-sm"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{SM}}(\boldsymbol{\phi})
    := \frac{1}{2}  \mathbb{E}_{{\mathbf{x}} \sim p_{\mathrm{data}}}
    \Big[\| {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}) - {\mathbf{s}}({\mathbf{x}})\|_2^2\Big].
\end{aligned}
\tag{3.2.1}
$$

**可计算的 score 匹配。** 乍看之下，这一目标似乎无法实现，因为作为回归目标的真实 score ${\mathbf{s}}({\mathbf{x}})$ 是未知的。幸运的是，[Hyvärinen and Dayan (2005)](../../references.md#ref-hyvarinen2005estimation) 表明，通过分部积分可以得到一个等价目标，它仅依赖于模型 ${\mathbf{s}}_{\boldsymbol{\phi}}$ 和数据样本，无须知道真实 score。我们在下面的命题中陈述这一关键结果：

<span id="sm-trace"></span>

**命题 3.2.1：Hyvärinen 给出的 SM 可计算形式**

我们可以将等式写为：

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}) = \widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol{\phi}) + C.
\end{aligned}
$$

其中

<span id="eq-sm-alternative"></span>

$$
\begin{aligned}
\widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol{\phi})
            := \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[
            \operatorname{Tr}\left(\nabla_{{\mathbf{x}}}{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right)
            + \frac{1}{2} \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right\rVert_2^2
            \right].
\end{aligned}
\tag{3.2.2}
$$

且 $C$ 是与 $\boldsymbol{\phi}$ 无关的常数。使目标最小化的 ${\mathbf{s}}^*$ 为：${\mathbf{s}}^*(\cdot) = \nabla_{{\mathbf{x}}} \log p(\cdot)$。

**命题的证明。**

展开 $\mathcal{L}_{\text{SM}}$ 中的 MSE 并应用分部积分，即可得到该结果。证明见[第 D.2.1 节](../../appendices/d-supplementary-proofs/02-score-based-perspective.md#app-sec-sm-trace)。

使用[式（3.2.2）](02-score-based-models.md#eq-sm-alternative)中的等价目标，训练 score 模型 ${\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})$ 仅需要来自 $p_{\mathrm{data}}$ 的观测样本，无须真实的 score 函数。

**[式（3.2.2）](02-score-based-models.md#eq-sm-alternative)的直观解释。** 可以从其中的两项来理解 score 匹配的另一种目标 $\widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol\phi)$。范数项 $\tfrac12\|{\mathbf{s}}_{\boldsymbol\phi}({\mathbf{x}})\|^2$ 惩罚较大的 score 值，尤其是在 $p_{\mathrm{data}}$ 较大的区域。散度项 $\operatorname{Tr}(\nabla_{{\mathbf{x}}} {\mathbf{s}}_{\boldsymbol\phi}({\mathbf{x}}))$ 在损失中带有正号，因此最小化这一目标也倾向于产生局部收缩行为。这两项共同给出总体最优的 score 场 ${\mathbf{s}}^*({\mathbf{x}})=\nabla_{{\mathbf{x}}}\log p_{\mathrm{data}}({\mathbf{x}})$。下面进一步说明其几何直觉。

**幅值项的作用。** 由于 $\widetilde{\mathcal{L}}_{\mathrm{SM}}(\boldsymbol\phi)$ 中的期望是在 $p_{\mathrm{data}}$ 下计算的，因此 $p_{\mathrm{data}}({\mathbf{x}})$ 较大的区域（数据密度较高的区域）对损失的贡献更大。所以，幅值项 $\tfrac12\|{\mathbf{s}}_{\boldsymbol\phi}({\mathbf{x}})\|^2$ 会对这些区域中较大的 score 值施加更强的惩罚。不过，仅凭这一项，并不能推断出每个高密度点都满足 ${\mathbf{s}}_{\boldsymbol\phi}({\mathbf{x}})=\mathbf{0}$；在总体最优解处，

$$
{\mathbf{s}}^*({\mathbf{x}})=\nabla_{{\mathbf{x}}}\log p_{\mathrm{data}}({\mathbf{x}}),
$$

它只在对数密度的驻点处为零。

**向量场（近似）为梯度场时的凹性。** 由于散度项 $\operatorname{Tr}(\nabla_{{\mathbf{x}}} {\mathbf{s}}_{\boldsymbol\phi}({\mathbf{x}}))$ 以正号进入损失，最小化目标会倾向于使数据分布所强调的区域具有负散度。负散度表示局部净收缩，但仅凭这一点并不能保证某个驻点是汇点。为了准确说明这一点，假设 ${\mathbf{s}}_{\boldsymbol\phi}=\nabla_{{\mathbf{x}}}u$，其中 $u:{\mathbb{R}}^D\to{\mathbb{R}}$ 是一个标量函数；在匹配对数密度时，这一假设是自然的。此时，$\nabla_{{\mathbf{x}}}{\mathbf{s}}_{\boldsymbol\phi}=\nabla_{{\mathbf{x}}}^2 u$（Hessian 矩阵），且 $\nabla \cdot {\mathbf{s}}_{\boldsymbol\phi}({\mathbf{x}})=\operatorname{Tr}(\nabla_{{\mathbf{x}}}^2 u({\mathbf{x}}))$（散度）。

在驻点 ${\mathbf{x}}_\star$ 处，有 ${\mathbf{s}}_{\boldsymbol\phi}({\mathbf{x}}_\star)=\nabla_{{\mathbf{x}}}u({\mathbf{x}}_\star)=\mathbf{0}$，二阶 Taylor 展开给出

$$
u({\mathbf{x}})
= u({\mathbf{x}}_\star)
+ \tfrac12({\mathbf{x}}-{\mathbf{x}}_\star)^\top \nabla_{{\mathbf{x}}}^2 u({\mathbf{x}}_\star)({\mathbf{x}}-{\mathbf{x}}_\star)
+ o(\|{\mathbf{x}}-{\mathbf{x}}_\star\|^2).
$$

如果 Hessian 矩阵 $\nabla_{{\mathbf{x}}}^2 u({\mathbf{x}}_\star)$ 负定，那么 $u$ 在 ${\mathbf{x}}_\star$ 处局部为凹函数，对数密度在此取得严格局部极大值[^1]。由于 Hessian 矩阵的所有特征值均为负，其迹也为负：$\operatorname{Tr}(\nabla_{{\mathbf{x}}}^2 u({\mathbf{x}}_\star))<0$。因此，在这种情况下，该驻点在 score 场作用下具有局部吸引性：小扰动会被拉回 ${\mathbf{x}}_\star$。

## 3.2.2 使用 Langevin 动力学采样 {#sampling-with-langevin-dynamics}

通过最小化[式（3.2.2）](02-score-based-models.md#eq-sm-alternative)完成训练后，score 模型 ${\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}})$ 就可以替代 Langevin 动力学中的 oracle score 来进行采样：

<span id="eq-score-langevin-emp"></span>

$$
\begin{aligned}
\mathbf{x}_{n+1} = \mathbf{x}_n + \eta   {\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_n) + \sqrt{2\eta}    \boldsymbol{\epsilon}_n, \quad \boldsymbol{\epsilon}_n \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}),
\end{aligned}
\tag{3.2.3}
$$

其中 $n=0,1,2,\dots$，初始状态为 ${\mathbf{x}}_0$。与 EBM 情形中的[式（3.1.6）](01-energy-based-models.md#eq-continuous-langevin)一样，这一递推恰好是连续时间 Langevin SDE 的 Euler–Maruyama 离散化：

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}}(t) = {\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t)) \mathop{}\!\mathrm{d} t + \sqrt{2} \mathop{}\!\mathrm{d} \mathbf{w}(t),
\end{aligned}
$$

其初始状态为 ${\mathbf{x}}(0)$。因此，在步长趋于零的极限下，离散形式与连续形式一致。在实践中，既可以运行离散采样器，也可以直接模拟该 SDE。

## 3.2.3 引子：基于 score 的生成模型 {#prologue-score-based-generative-models}

在本章的其余部分，我们将考察 score 函数在现代扩散模型中的基础作用。score 函数最初被引入是为了高效训练 EBM，如今已成为新一代生成模型的关键组成部分。在此基础上，我们将探讨 score 函数如何指导*基于 score 的扩散模型*的理论表述和实际实现，从而为通过随机过程生成数据提供一个具有理论依据的框架。

[^1]: 这里指出，严格凹性（从而使对数密度取得严格局部极大值）要求整个 Hessian 矩阵 $\nabla_{{\mathbf{x}}}^2 u$ 负定，仅有负迹是不够的。负迹保证特征值之和为负，但其中一些特征值仍可能为正，从而产生鞍点，而非极大值点。

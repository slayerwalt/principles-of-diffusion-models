# 6.1 条件化技巧：扩散模型的秘诀 {#sec-conditional-trick}

到目前为止，我们从三个看似不同的起点探索了扩散模型：变分、基于 score 与基于流的视角。每个视角最初的动机不同，并导出各自的训练目标（固定 $t$）：

-   **变分视角：** 学习参数化密度 $p_{\boldsymbol{\phi}}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)$，近似 oracle 逆向转移 $p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)$，最小化目标为：

    $$
    \mathcal{J}_{\text{KL}}(\boldsymbol{\phi}) := \mathbb{E}_{p_t(\mathbf{x}_t)}\left[
            \mathcal{D}_{\mathrm{KL}}\big(p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t) \| p_{\boldsymbol{\phi}}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)\big)
        \right];
    $$

-   **基于 score 的视角：** 学习 score 模型 ${\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$，近似边缘 score $\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)$，目标为：

    $$
    \mathcal{J}_{\text{SM}}(\boldsymbol{\phi}) := \mathbb{E}_{p_t({\mathbf{x}}_t)}\left[
           \left\| {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \right\|_2^2
        \right];
    $$

-   **基于流的视角：** 学习速度模型 ${\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$，匹配 oracle 速度 ${\mathbf{v}}_t({\mathbf{x}}_t)$（例如由[式 (5.2.9)](../05-flow-based-perspective/02-flow-matching-framework.md#eq-marginal-oracle-velocity)定义），最小化目标为：

    $$
    \mathcal{J}_{\text{FM}}(\boldsymbol{\phi}) := \mathbb{E}_{p_t({\mathbf{x}}_t)} \left[
            \left\| {\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - {\mathbf{v}}_t({\mathbf{x}}_t) \right\|_2^2
        \right].
    $$

乍看之下，这些目标似乎完全无法计算，因为它们都需要访问一般情况下根本无法获知的 oracle 量。但有趣的是，各种方法分别找到了同一个简洁解法：*以数据 ${\mathbf{x}}_0$ 为条件*。这一技巧将每个不可计算的训练目标都转化为可计算目标。

这种“条件化技巧”将目标改写为对已知 Gaussian 条件分布 $p_t({\mathbf{x}}_t | {\mathbf{x}}_0)$ 的期望，得到可计算的条件目标和训练目标，并在梯度意义下与相应边缘目标等价：

-   **变分视角**（[式 (2.2.3)](../02-variational-perspective/02-ddpm.md#eq-kl-matching)）**：**

    $$
    \mathcal{J}_{\text{KL}}(\boldsymbol{\phi}) = \underbrace{\mathbb{E}_{{\mathbf{x}}_0} \mathbb{E}_{p_t({\mathbf{x}}_t|{\mathbf{x}}_0)} \left[
            \mathcal{D}_{\mathrm{KL}}\big(p({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t, {\mathbf{x}}_0) \| p_{\boldsymbol{\phi}}({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t)\big)
        \right]}_{\mathcal{J}_{\text{CKL}}(\boldsymbol{\phi})} + C;
    $$

-   **基于 score 的视角**（[式 (3.3.3)](../03-score-based-perspective/03-denoising-score-matching.md#eq-score-matching)）**：**

    $$
    \mathcal{J}_{\text{SM}}(\boldsymbol{\phi}) = \underbrace{\mathbb{E}_{{\mathbf{x}}_0} \mathbb{E}_{p_t({\mathbf{x}}_t|{\mathbf{x}}_0)} \left[
            \left\| {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t | {\mathbf{x}}_0) \right\|_2^2
        \right]}_{\mathcal{J}_{\text{DSM}}(\boldsymbol{\phi})} + C;
    $$

-   **基于流的视角**（[式 (5.2.10)](../05-flow-based-perspective/02-flow-matching-framework.md#eq-fm-cfm)）**：**

    $$
    \mathcal{J}_{\text{FM}}(\boldsymbol{\phi}) = \underbrace{\mathbb{E}_{{\mathbf{x}}_0} \mathbb{E}_{p_t({\mathbf{x}}_t|{\mathbf{x}}_0)} \left[
            \left\| {\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - {\mathbf{v}}_t({\mathbf{x}}_t | {\mathbf{x}}_0) \right\|^2
        \right]}_{\mathcal{J}_{\text{CFM}}(\boldsymbol{\phi})} + C.
    $$

为建立统一视角，下面系统地重新考察条件 KL、score 与速度目标。这些目标不仅可计算，而且与原始形式只相差一个常数的竖直平移。条件版本（$\mathcal{J}_{\text{CKL}}$、$\mathcal{J}_{\text{DSM}}$、$\mathcal{J}_{\text{CFM}}$）与原始版本（$\mathcal{J}_{\text{KL}}$、$\mathcal{J}_{\text{SM}}$、$\mathcal{J}_{\text{FM}}$）之间，只相差一个与模型参数无关的常数。因此，在同一模型类中，每个条件目标都与对应原始目标具有相同的最优解。

为理解这些最优解在理想情况下恢复的是什么，假设模型表达能力足以表示精确 oracle。尽管目标不同，三个视角都遵循同一概念模式：

$$
\underbrace{\text{marginal oracle}}_{\text{generally intractable}}
=
\mathbb{E}_{{\mathbf{x}}_0\sim p(\cdot|{\mathbf{x}}_t)}
\left[
\underbrace{\text{conditional oracle}}_{\text{tractable given }{\mathbf{x}}_0}
\right].
$$

也就是说，以干净样本 ${\mathbf{x}}_0$ 为条件可得到可计算目标，而对所有与 ${\mathbf{x}}_t$ 相容的可能干净样本上的这些条件目标取平均，就恢复了所需的边缘 oracle。具体而言，

<span id="eq-three-minimizer-oracle"></span>

$$
\begin{aligned}
        p^*(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t) &= \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot|\mathbf{x}_t)} \big[p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t, \mathbf{x}_0)\big] &&= p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t), \\
    \mathbf{s}^*(\mathbf{x}_t, t) &= \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot|\mathbf{x}_t)} \big[\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0)\big] &&= \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t), \\
    \mathbf{v}^*(\mathbf{x}_t, t) &= \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot|\mathbf{x}_t)} \big[\mathbf{v}_t(\mathbf{x}_t|\mathbf{x}_0)\big] &&= \mathbf{v}_t(\mathbf{x}_t).
\end{aligned}
\tag{6.1.1}
$$

对于变分视角，第一个恒等式来自对条件逆向转移关于 ${\mathbf{x}}_0$ 的边缘化，所得边缘逆向转移最小化 KL 目标。对于 score 与速度匹配，平方误差目标是最小二乘回归，其总体最优解为对应的条件期望。这些恒等式描述不受限制的总体 oracle；受约束的神经模型未必能精确表示它们。下面给出有限数据点情形的具体示例。

**示例：有限数据集上的闭式 oracle 目标**

为使[式 (6.1.1)](01-conditioning-tricks.md#eq-three-minimizer-oracle)更具体，假设数据分布是经验测度

$$
\hat p_0=\frac{1}{N}\sum_{i=1}^N \delta_{{\mathbf{x}}_0^{(i)}},
$$

其中 $\{{\mathbf{x}}_0^{(i)}\}_{i=1}^N$ 表示有限训练集。

在[式 (6.1.1)](01-conditioning-tricks.md#eq-three-minimizer-oracle)中，记号

$$
{\mathbf{x}}_0 \sim p(\cdot|{\mathbf{x}}_t)
$$

表示在给定含噪观测 ${\mathbf{x}}_t$ 时，对干净样本的后验分布取平均。等价地，对于任意量 ${\mathbf{h}}({\mathbf{x}}_0)$，

$$
\mathbb{E}_{{\mathbf{x}}_0 \sim p(\cdot|{\mathbf{x}}_t)}[{\mathbf{h}}({\mathbf{x}}_0)]
=
\int {\mathbf{h}}({\mathbf{x}}_0)\, p({\mathbf{x}}_0|{\mathbf{x}}_t)\mathop{}\!\mathrm{d}{\mathbf{x}}_0.
$$

由 Bayes 法则，

$$
p({\mathbf{x}}_0|{\mathbf{x}}_t)
=
\frac{p_t({\mathbf{x}}_t|{\mathbf{x}}_0)\,p_{\text{data}}({\mathbf{x}}_0)}
{\int p_t({\mathbf{x}}_t|\tilde{{\mathbf{x}}}_0)\,p_{\text{data}}(\tilde{{\mathbf{x}}}_0)\mathop{}\!\mathrm{d}\tilde{{\mathbf{x}}}_0}.
$$

在标准 Gaussian 扩散前向过程中，

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)})
=
\mathcal{N}\left({\mathbf{x}}_t;\alpha_t {\mathbf{x}}_0^{(i)}, \sigma_t^2 {\mathbf{I}}\right),
$$

因此，条件似然是显式的。然而，在总体层面，后验 $p({\mathbf{x}}_0|{\mathbf{x}}_t)$ 仍需要对完整数据分布 $p_{\text{data}}$ 取平均。

若现在将 $p_{\text{data}}$ 替换为经验测度

$$
\hat p_0=\frac{1}{N}\sum_{i=1}^N \delta_{{\mathbf{x}}_0^{(i)}},
$$

则后验 $p({\mathbf{x}}_0|{\mathbf{x}}_t)$ 仅支撑在训练样例上，其在 ${\mathbf{x}}_0^{(i)}$ 处的概率质量为

$$
p({\mathbf{x}}_0={\mathbf{x}}_0^{(i)}|{\mathbf{x}}_t)
=
\frac{p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)})}
{\sum_{j=1}^N p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(j)})}
=: w_i({\mathbf{x}}_t,t),
\qquad
\sum_{i=1}^N w_i({\mathbf{x}}_t,t)=1.
$$

因此，上面的后验期望化为有限求和

$$
\mathbb{E}_{{\mathbf{x}}_0 \sim p(\cdot|{\mathbf{x}}_t)}[{\mathbf{h}}({\mathbf{x}}_0)]
=
\sum_{i=1}^N w_i({\mathbf{x}}_t,t){\mathbf{h}}({\mathbf{x}}_0^{(i)}).
$$

于是，在有限数据集情形下，[式 (6.1.1)](01-conditioning-tricks.md#eq-three-minimizer-oracle)中抽象的后验平均，成为训练集上的具体加权平均，权重为 $w_i({\mathbf{x}}_t,t)$。下面明确写出这三种情形。

**变分视角。** oracle 逆向转移成为

$$
p^*({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t)
=
\sum_{i=1}^N
w_i({\mathbf{x}}_t,t)
p({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t,{\mathbf{x}}_0^{(i)}).
$$

这里每个条件逆向核

$$
p({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t,{\mathbf{x}}_0^{(i)})
$$

在 Gaussian 扩散下本身都有闭式形式；事实上，如[引理 2.2.2](../02-variational-perspective/02-ddpm.md#ddpm-reverse-kernel)所示，它是 Gaussian 分布。因此，真实逆向核只是显式 Gaussian 桥按后验加权的混合。

**基于 score 的视角。** 类似地，oracle score 成为

$$
\mathbf{s}^*({\mathbf{x}}_t,t)
=
\sum_{i=1}^N
w_i({\mathbf{x}}_t,t)
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)}).
$$

条件 score 是显式的：

$$
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)})
=
-\frac{{\mathbf{x}}_t-\alpha_t{\mathbf{x}}_0^{(i)}}{\sigma_t^2}.
$$

因此，

$$
\mathbf{s}^*({\mathbf{x}}_t,t)
=
-\frac{1}{\sigma_t^2}
\left(
{\mathbf{x}}_t-\alpha_t\sum_{i=1}^N w_i({\mathbf{x}}_t,t) {\mathbf{x}}_0^{(i)}
\right).
$$

所以，边缘 score 从含噪样本指向训练样例的后验加权平均。

**基于流的视角。** 流匹配中也出现同样现象：

$$
\mathbf{v}^*({\mathbf{x}}_t,t)
=
\sum_{i=1}^N
w_i({\mathbf{x}}_t,t)
\mathbf{v}_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)}).
$$

同样，组成部分都是显式的。对应条件速度场具有闭式形式：

$$
\mathbf{v}_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)})
=
{\alpha}'_t {\mathbf{x}}_0^{(i)}
+
\frac{{\sigma}'_t}{\sigma_t}\bigl({\mathbf{x}}_t-\alpha_t{\mathbf{x}}_0^{(i)}\bigr).
$$

因此，oracle 速度同样是数据集上的显式后验加权平均。

这些公式共同表明，在有限数据集上，三个 oracle 目标都化为闭式条件量的显式后验加权平均。

**关于记忆的说明。** 上述公式刻画经验分布 $\hat p_0$ 所诱导的精确 oracle 目标，而非未知总体分布 $p_{\mathrm{data}}$ 的目标。若模型在每个噪声水平都完美学得这些经验目标，且逆向过程计算没有误差，生成就会精确重现经验分布 $\hat p_0$。由于 $\hat p_0$ 仅对训练样例赋予概率，生成样本会直接来自训练集本身。在这一理想化意义下，完美拟合经验 oracle 就等于记忆。

然而，实际训练的扩散模型未必精确实现这一 oracle：模型容量与优化过程有限，生成分布还受模型近似误差和数值采样误差影响。因此，有限数据集计算说明，记忆是一个精确的经验解，但不意味着每个实际训练的模型都必然记住训练集。

这种共同的条件化改写并非巧合：它在使训练可计算的同时，揭示了深层统一性。变分扩散、基于 score 的 SDE 与流匹配，都是同一原理的不同侧面。三个视角通过一个共同认识自然地联系起来。本章余下部分将继续探索它们的等价性。
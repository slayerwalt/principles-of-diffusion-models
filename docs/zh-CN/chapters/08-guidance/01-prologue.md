# 8.1 引言 {#sec-prologue-guidance}

<span id="fig-guidance"></span>

![引导使逆时间扩散采样从噪声走向所需的日本绘画风格](../../../en/assets/chapter-08/steered-diffusion-sampling.png)

**图 8.1：** **受引导的扩散采样示意图。** 逆时间 PF-ODE 采样从右侧的纯噪声（$t = T$）开始，逐渐演化为左侧的干净样本（$t = 0$）。在此过程中，由 $w_t$ 加权的时变引导方向修改无条件 score 场，将轨迹引向所需属性（日本绘画风格），同时样本的细节由粗到细逐步完善。

来源：作者绘制。

扩散模型的生成过程由粗到细进行，为可控生成提供了灵活的框架。每一步去除少量噪声，使样本更加清晰，逐渐显现更多结构和细节。这一性质使我们能够控制生成过程：在学习到的时变速度场中加入引导项，就能使生成轨迹体现用户意图。

扩散模型中基于引导的采样，其理论基础是条件 score 的 Bayes 分解。对于每个噪声水平 $t$，

<span id="eq-guidance-bayes"></span>

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t |{\mathbf{c}})
=
\underbrace{\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)}_{\text{unconditional direction}}
+
\underbrace{\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{c}} |{\mathbf{x}}_t)}_{\text{guidance direction}}.
\end{aligned}
\tag{8.1.1}
$$

这一恒等式说明，在无条件 score 上加入引导项 $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{c}}|{\mathbf{x}}_t)$，即可得到精确的条件逆向动力学。后文将看到，分类器引导（[Dhariwal and Nichol, 2021](../../references.md#ref-dhariwal2021diffusion)）直接估计这一项，无分类器引导（[Ho and Salimans, 2021](../../references.md#ref-ho2021classifier)）通过联合学习的条件与无条件 score 之差间接估计它，而更一般的免训练方法则可能使用针对任务的代理引导方向。

使用精确条件 score 时，PF-ODE 变为

<span id="eq-condition-prob-ode"></span>

$$
\begin{aligned}
    \frac{\mathop{}\!\mathrm{d} \mathbf{x}(t)}{\mathop{}\!\mathrm{d} t} &= f(t)\mathbf{x}(t) - \frac{1}{2} g^2(t) \underbrace{\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}(t) |{\mathbf{c}})}_{\text{conditional score}} \\
    &= f(t)\mathbf{x}(t) - \frac{1}{2} g^2(t) \Big[ \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}(t)) + \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{c}} |{\mathbf{x}}(t)) \Big].
\end{aligned}
\tag{8.1.2}
$$

在 score 精确且积分精确的条件下，从相应终端边缘分布初始化时，这些动力学会恢复 $p_0({\mathbf{x}}_0|{\mathbf{c}})$，其中初始化分布为 $p_T({\mathbf{x}}_T|{\mathbf{c}})$。在标准扩散构造中，终端加噪的选择使得 $p_T({\mathbf{x}}_T|{\mathbf{c}})$ 近似匹配通常采用的、与条件无关的先验。

为清晰起见，下文采用 score 预测进行表述。标准的 ${\mathbf{x}}$ 预测、${\boldsymbol{\epsilon}}$ 预测、score 预测及 ${\mathbf{v}}$ 预测参数化，在转换有定义的地方，通过随时间变化的线性变换相互联系。因此，同一引导修正也可以利用[式 (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)中的关系，用其他参数化形式表达。

**引导方向的具体形式。**

1.  **分类器引导（CG）。** 在[第 8.2 节](02-classifier-guidance.md#sec-cg)中，分类器引导（CG）训练以时间为条件的分类器 $p_{{\boldsymbol{\psi}}}({\mathbf{c}} | {\mathbf{x}}_t, t)$，训练数据为加噪数据 ${\mathbf{x}}_t$，这些数据由干净的带标签样本在噪声水平 $t$ 下扰动得到。在采样时，分类器相对于输入的梯度提供引导项：

    $$
    \nabla_{{\mathbf{x}}_t}\log p_{{\boldsymbol{\psi}}}({\mathbf{c}} | {\mathbf{x}}_t, t)
     \approx  \nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{c}} | {\mathbf{x}}_t),
    $$

    然后将其加入无条件 score（[Dhariwal and Nichol, 2021](../../references.md#ref-dhariwal2021diffusion)）。

2.  **无分类器引导（CFG）。** 在[第 8.3 节](03-classifier-free-guidance.md#sec-cfg)中，CFG 直接训练单个条件模型

    $$
    {\mathbf{s}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t,{\mathbf{c}}) \approx \nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{c}}),
    $$

    在一部分训练步骤中，随机将条件替换为特殊空标记，从而联合学习无条件模型。

3.  **免训练（代理）引导。** 在标准前向加噪构造中，先采样 $({\mathbf{x}}_0,{\mathbf{c}})\sim p_{\mathrm{data}}$，再采样 ${\mathbf{x}}_t\sim p_t(\cdot|{\mathbf{x}}_0)$，其中前向核不额外依赖 ${\mathbf{c}}$。因此 ${\mathbf{c}}\perp{\mathbf{x}}_t\mid{\mathbf{x}}_0$[^1]。于是，

    $$
    p_t({\mathbf{c}}|{\mathbf{x}}_t)
    =
    \int
    p({\mathbf{c}}|{\mathbf{x}}_0)\,
    p({\mathbf{x}}_0|{\mathbf{x}}_t)\,
    \mathrm d{\mathbf{x}}_0.
    $$

    这一后验平均通常难以计算，因此免训练方法转而构造针对任务的代理引导方向。

    在[第 8.4.1 节](04-training-free-guidance.md#subsec-tfg)中，免训练（基于损失的）引导避免直接计算条件似然 $p_t({\mathbf{c}}|{\mathbf{x}}_t)$。它从现成的损失 $\ell({\mathbf{x}}_t,{\mathbf{c}};t)$ 出发，度量当前样本满足所需条件的程度；损失越小，一致程度越高。根据这一损失，定义正值的*引导势函数*

    $$
    {\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})
    :=
    \exp\big(-\tau\,\ell({\mathbf{x}}_t,{\mathbf{c}};t)\big),
    \qquad \tau>0.
    $$

    因此，损失较低的样本具有较大的势函数值。将相应的*引导方向*记为

    $$
    \mathcal{G}_t({\mathbf{x}}_t,{\mathbf{c}})
    :=
    \nabla_{{\mathbf{x}}_t}\log {\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})
    =
    -\tau\nabla_{{\mathbf{x}}_t}\ell({\mathbf{x}}_t,{\mathbf{c}};t).
    $$

    分类器引导自然符合这一视角。在总体层面，其引导势函数为

    $$
    {\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})
    =
    p_t({\mathbf{c}}|{\mathbf{x}}_t),
    $$

    实践中通过学习到的分类器 $p_{{\boldsymbol{\psi}}^\times}({\mathbf{c}}|{\mathbf{x}}_t,t)$ 近似。等价地，学习到的分类器对应于损失

    $$
    \ell({\mathbf{x}}_t,{\mathbf{c}};t)
    =
    -\log p_{{\boldsymbol{\psi}}^\times}({\mathbf{c}}|{\mathbf{x}}_t,t),
    \qquad \tau=1.
    $$

在总体层面，上述三种构造共享同一个加性引导形式。给定正值势函数 ${\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})$ 及引导强度 $w_t\geq0$，定义引导后的 score 场

<span id="eq-generic-guided-score"></span>

$$
\begin{aligned}
{\mathbf{s}}^{\mathrm{guide}}({\mathbf{x}}_t,t,{\mathbf{c}};w_t)
:=
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)
+
w_t\mathcal{G}_t({\mathbf{x}}_t,{\mathbf{c}}).
\end{aligned}
\tag{8.1.3}
$$

对于基于损失的引导，$\mathcal{G}_t({\mathbf{x}}_t,{\mathbf{c}}) =-\tau\nabla_{{\mathbf{x}}_t}\ell({\mathbf{x}}_t,{\mathbf{c}};t)$。对于精确分类器引导，$\mathcal{G}_t({\mathbf{x}}_t,{\mathbf{c}}) =\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{c}}|{\mathbf{x}}_t)$；无分类器引导则通过条件与无条件 score 之差，表达同一个总体引导方向。

在每个固定时刻 $t$，只要可以归一化，[式 (8.1.3)](01-prologue.md#eq-generic-guided-score) 就是如下倾斜密度的 score：

$$
\widetilde p_t^{\mathrm{tilt}}({\mathbf{x}}_t|{\mathbf{c}})
\propto
p_t({\mathbf{x}}_t){\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})^{w_t}.
$$

这提供了有用的能量解释：基础密度偏好扩散模型所支持的样本，势函数则对它们重新加权，使其趋向所需条件。

实践中，将[式 (8.1.3)](01-prologue.md#eq-generic-guided-score)相应的学习版本或代理版本代入逆时间动力学。需要注意，上述倾斜密度只是在固定时刻的解释：密度族 $\{\widetilde p_t^{\mathrm{tilt}}(\cdot|{\mathbf{c}})\}_t$ 不必与引导采样器生成的边缘分布一致。因此，对于一般损失或引导调度，从 $T$ 到 $0$ 积分引导后的 PF-ODE，一般不会生成服从 $\widetilde p_0^{\mathrm{tilt}}(\cdot|{\mathbf{c}})$ 的样本。要使这一解释成立，还需要额外的跨时间一致性，[第 8.4.2 节](04-training-free-guidance.md#subsec-guided-marginal-path)将对此讨论。

[图 8.1](01-prologue.md#fig-guidance)展示了引导对采样轨迹的影响。同一原理也可以用于已经带条件的模型，使额外控制信号修改其采样动力学。

**从控制到通过直接偏好优化改善对齐。**

较强的控制可以满足条件，却不符合偏好：样本可能满足条件信号（如提示词），但偏离人类实际偏好的结果。我们通过在 $t=0$ 处使用偏好评分倾斜条件目标，来形式化这一点：

$$
\widetilde p_0^{\mathrm{tilt}}({\mathbf{x}}_0|{\mathbf{c}})
\propto
p_0({\mathbf{x}}_0|{\mathbf{c}})
\exp\bigl(
\lambda_{\mathrm{pref}} r({\mathbf{x}}_0,{\mathbf{c}})
\bigr),
\qquad
\lambda_{\mathrm{pref}}>0,
$$

其中，$r({\mathbf{x}}_0,{\mathbf{c}})$ 是干净样本 ${\mathbf{x}}_0$ 在条件 ${\mathbf{c}}$ 下的标量对齐评分，数值越大表示偏好越强；$\lambda_{\mathrm{pref}}$ 控制偏好重加权的强度。在后文介绍的 KL 正则化表述中，最优解正具有这一形式，其基础分布为参考策略，且 $\lambda_{\mathrm{pref}}=1/\beta$。

实现这种可引导性的现有方法通常收集人类对模型生成结果相对质量的标注，再微调条件扩散模型，使其与这些偏好对齐，常用方式是基于人类反馈的强化学习（RLHF）。然而，RLHF 过程复杂且往往不稳定：它先拟合奖励模型来捕捉人类偏好，再通过强化学习微调条件扩散模型，在约束策略偏离原模型程度的同时，最大化估计的奖励。

这自然引出问题：*能否完全去掉奖励模型训练阶段？* 我们将通过 Diffusion-DPO（[Wallace et al., 2024](../../references.md#ref-wallace2024diffusion)）回答，它是对原先为大语言模型开发的直接偏好优化（Direct Preference Optimization，DPO）（[Rafailov et al., 2023](../../references.md#ref-rafailov2023direct)）的改编。如[第 8.5 节](05-preference-alignment.md#sec-rlhf-dpo)所述，Diffusion-DPO 直接从成对选择中学习偏好倾斜，从而在没有独立奖励模型的情况下，微调条件扩散模型以对齐偏好。

[^1]: 确实，

    $$
    p_t({\mathbf{c}}|{\mathbf{x}}_t)
    =
    \int
    p({\mathbf{c}},{\mathbf{x}}_0|{\mathbf{x}}_t)\,\mathrm d{\mathbf{x}}_0
    =
    \int
    p({\mathbf{c}}|{\mathbf{x}}_0,{\mathbf{x}}_t)\,
    p({\mathbf{x}}_0|{\mathbf{x}}_t)\,\mathrm d{\mathbf{x}}_0.
    $$

    由于标准前向核满足 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0,{\mathbf{c}})=p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$，有 $p({\mathbf{c}}|{\mathbf{x}}_0,{\mathbf{x}}_t)=p({\mathbf{c}}|{\mathbf{x}}_0)$，从而得到下面的表达式。

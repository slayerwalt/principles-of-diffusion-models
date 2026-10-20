# 8. 引导与可控生成 {#ch-guidance}

扩散模型是强大的生成框架。在无条件设定下，目标是学习 $p_{\text{data}}({\mathbf{x}})$，并在没有外部输入的情况下生成样本。

然而，许多应用需要*条件生成*，使输出满足用户指定的要求。这可以通过引导无条件模型实现，也可以直接学习条件分布 $p_0({\mathbf{x}} | {\mathbf{c}})$，利用条件 ${\mathbf{c}}$（如标签、文本描述或草图）引导生成过程。

本章以条件 score 的理论视角为基础：它可以分解为*无条件方向*和*引导方向*，后者在保持真实感的同时，将样本推向给定条件。我们解释引导为何重要，说明条件 score 如何成为统一的控制接口，并介绍近似或替代引导项的方法。随后区分*控制*（满足条件）与*对齐*（在条件下满足人类偏好），说明如何将偏好纳入同一框架。最后讨论无需额外奖励模型的直接偏好优化；奖励模型是指经过学习、为更符合人类偏好的输出赋予更高分值的评分器。

**读完本章后，你将能够……**

-   理解条件信息如何修改扩散采样中使用的速度场；

-   推导分类器引导与无分类器引导背后的关键思想，并理解引导强度如何控制条件作用；

-   理解免训练引导如何利用任务特定的损失或测量模型来引导预训练扩散模型，包括用于逆问题的情形；

-   区分精确条件采样、由代理信号引导的动力学，以及后者在固定时刻的倾斜密度解释；

-   理解从 RLHF 到 DPO 和 Diffusion-DPO 的基于偏好的对齐，如何进一步塑造条件生成模型，使其倾向于受偏好的输出。

**阅读路线。** 本章假定读者已了解[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)和[第 6 章](../06-unified-perspective/index.md#ch-unified-perspective)中的 Score SDE／PF-ODE 视角。主要关注可控生成的读者，可以在第二遍阅读时再看后面的偏好对齐内容。

继续阅读下面各节及[参考文献](../../references.md#references)。

[8.1 引言](01-prologue.md#sec-prologue-guidance)

[8.2 分类器引导](02-classifier-guidance.md#sec-cg)

[8.3 无分类器引导](03-classifier-free-guidance.md#sec-cfg)

[8.4 免训练引导](04-training-free-guidance.md#sec-training-free-guidance)

[8.5 从强化学习到用于模型对齐的直接偏好优化](05-preference-alignment.md#sec-rlhf-dpo)

[8.6 结语](06-closing-remarks.md#sec-ch8-cr)

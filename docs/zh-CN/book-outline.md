---
title: 全书目录
description: 中文网页版的目录。
---

# 全书目录 {#book-contents}

本目录遵循原书顺序，包括书前材料、第 1—12 章、第二部分无编号的开篇内容、各部分的概览图，以及附录 A—D。

## 书前材料 {#front-matter}

- [作者](front-matter/authors.md)
- [摘要](front-matter/abstract.md)
- [致谢](front-matter/acknowledgements.md)
- [前言与阅读路线](front-matter/preface.md)
- [符号说明](front-matter/notation.md)

## 第一部分 · 深度生成建模导论 {#part-i-introduction-to-deep-generative-modeling}

1. [深度生成建模](chapters/01-deep-generative-modeling/index.md)

    - [1.1 什么是深度生成建模？](chapters/01-deep-generative-modeling/01-what-is-deep-generative-modeling.md)
    - [1.2 主要的深度生成模型](chapters/01-deep-generative-modeling/02-examples.md)
    - [1.3 建模方法的分类](chapters/01-deep-generative-modeling/03-taxonomy.md)
    - [1.4 结语](chapters/01-deep-generative-modeling/04-summary.md)

## 第二部分 · 扩散模型的基础与视角 {#part-ii-foundations-and-perspectives-of-diffusion-models}

[本部分概览图](foundations/index.md)

[从操作流程初步理解扩散模型的训练与采样](foundations/operational-view.md)是本部分无编号的开篇内容。

2. [变分视角：从 VAE 到 DDPM](chapters/02-variational-perspective/index.md)

    - [2.1 变分自编码器](chapters/02-variational-perspective/01-variational-autoencoder.md)
    - [2.2 变分视角：DDPM](chapters/02-variational-perspective/02-ddpm.md)
    - [2.3 结语](chapters/02-variational-perspective/03-closing-remarks.md)

3. [基于 score 的视角：从 EBM 到 NCSN](chapters/03-score-based-perspective/index.md)

    - [3.1 基于能量的模型](chapters/03-score-based-perspective/01-energy-based-models.md)
    - [3.2 从基于能量的生成模型到基于 score 的生成模型](chapters/03-score-based-perspective/02-score-based-models.md)
    - [3.3 去噪 score 匹配](chapters/03-score-based-perspective/03-denoising-score-matching.md)
    - [3.4 多噪声水平下的去噪 score 匹配（NCSN）](chapters/03-score-based-perspective/04-noise-conditional-score-networks.md)
    - [3.5 总结：NCSN 与 DDPM 的比较](chapters/03-score-based-perspective/05-comparison.md)
    - [3.6 结语](chapters/03-score-based-perspective/06-closing-remarks.md)

4. [当今的扩散模型：Score SDE 框架](chapters/04-score-sde-framework/index.md)

    - [4.1 Score SDE 的原理](chapters/04-score-sde-framework/01-score-sde-principles.md)
    - [4.2 前向与逆时间 SDE 及 PF-ODE 中的边缘分布匹配](chapters/04-score-sde-framework/02-matching-marginals.md)
    - [4.3 Score SDE 的训练与采样](chapters/04-score-sde-framework/03-training-and-sampling.md)
    - [4.4 SDE 的具体形式](chapters/04-score-sde-framework/04-sde-instantiations.md)
    - [4.5 通过累积扰动核的直接视角](chapters/04-score-sde-framework/05-accumulated-perturbation-kernels.md)
    - [4.6 （选读）通过边缘化与 Bayes 法则推导 Fokker–Planck 方程和逆时间 SDE](chapters/04-score-sde-framework/06-fokker-planck-and-reverse-sde.md)
    - [4.7 结语](chapters/04-score-sde-framework/07-closing-remarks.md)

5. [基于流的视角：从 NF 到流匹配](chapters/05-flow-based-perspective/index.md)

    - [5.1 基于流的模型：归一化流与神经 ODE](chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md)
    - [5.2 流匹配框架](chapters/05-flow-based-perspective/02-flow-matching-framework.md)
    - [5.3 构造分布之间的概率路径与速度](chapters/05-flow-based-perspective/03-probability-paths-and-velocities.md)
    - [5.4 （选读）标准仿射流的性质](chapters/05-flow-based-perspective/04-canonical-affine-flow.md)
    - [5.5 结语](chapters/05-flow-based-perspective/05-closing-remarks.md)

6. [扩散模型的统一与系统视角](chapters/06-unified-perspective/index.md)

    - [6.1 条件化技巧：扩散模型的秘诀](chapters/06-unified-perspective/01-conditioning-tricks.md)
    - [6.2 理解扩散模型训练损失的路线图](chapters/06-unified-perspective/02-training-objectives.md)
    - [6.3 扩散模型中的等价性](chapters/06-unified-perspective/03-equivalent-parameterizations.md)
    - [6.4 这一切的基础：Fokker–Planck 方程](chapters/06-unified-perspective/04-fokker-planck-connections.md)
    - [6.5 结语](chapters/06-unified-perspective/05-closing-remarks.md)

7. [（选读）扩散模型与最优传输](chapters/07-optimal-transport/index.md)

    - [7.1 分布间转换引言](chapters/07-optimal-transport/01-distribution-translation.md)
    - [7.2 问题设定的分类](chapters/07-optimal-transport/02-transport-problem-setups.md)
    - [7.3 不同最优传输表述之间的关系](chapters/07-optimal-transport/03-relationships-between-formulations.md)
    - [7.4 扩散模型的 SDE 是 SB 问题的最优解吗？](chapters/07-optimal-transport/04-diffusion-sdes-and-schrodinger-bridges.md)
    - [7.5 扩散模型的 ODE 是 OT 问题的最优映射吗？](chapters/07-optimal-transport/05-probability-flow-odes-and-optimal-transport.md)
    - [7.6 结语](chapters/07-optimal-transport/06-closing-remarks.md)

## 第三部分 · 扩散模型的采样 {#part-iii-sampling-of-diffusion-models}

[本部分概览图](sampling/index.md)

8. [引导与可控生成](chapters/08-guidance/index.md)

    - [8.1 引言](chapters/08-guidance/01-prologue.md)
    - [8.2 分类器引导](chapters/08-guidance/02-classifier-guidance.md)
    - [8.3 无分类器引导](chapters/08-guidance/03-classifier-free-guidance.md)
    - [8.4 无需训练的引导](chapters/08-guidance/04-training-free-guidance.md)
    - [8.5 模型对齐：从强化学习到直接偏好优化](chapters/08-guidance/05-preference-alignment.md)
    - [8.6 结语](chapters/08-guidance/06-closing-remarks.md)

9. [用于快速扩散采样的数值求解器](chapters/09-numerical-solvers/index.md)

    - [9.1 引言](chapters/09-numerical-solvers/01-prologue.md)
    - [9.2 DDIM](chapters/09-numerical-solvers/02-ddim.md)
    - [9.3 DEIS](chapters/09-numerical-solvers/03-deis.md)
    - [9.4 DPM-Solver](chapters/09-numerical-solvers/04-dpm-solver.md)
    - [9.5 DPM-Solver++](chapters/09-numerical-solvers/05-dpm-solver-plus-plus.md)
    - [9.6 （选读）ParaDiGMs](chapters/09-numerical-solvers/06-paradigms.md)
    - [9.7 结语](chapters/09-numerical-solvers/07-closing-remarks.md)

## 第四部分 · 学习快速扩散生成器 {#part-iv-toward-learning-fast-diffusion-based-generators}

10. [用于快速采样的蒸馏方法](chapters/10-distillation/index.md)

    - [10.1 引言](chapters/10-distillation/01-prologue.md)
    - [10.2 基于分布的蒸馏](chapters/10-distillation/02-distribution-based-distillation.md)
    - [10.3 渐进式蒸馏](chapters/10-distillation/03-progressive-distillation.md)
    - [10.4 结语](chapters/10-distillation/04-closing-remarks.md)

11. [从头学习快速生成器](chapters/11-fast-generators/index.md)

    - [11.1 引言](chapters/11-fast-generators/01-prologue.md)
    - [11.2 特殊流映射：离散时间的 Consistency Model](chapters/11-fast-generators/02-discrete-consistency-models.md)
    - [11.3 特殊流映射：连续时间的 Consistency Model](chapters/11-fast-generators/03-continuous-consistency-models.md)
    - [11.4 一般流映射：Consistency Trajectory Model](chapters/11-fast-generators/04-consistency-trajectory-models.md)
    - [11.5 一般流映射：Mean Flow](chapters/11-fast-generators/05-mean-flow.md)
    - [11.6 结语](chapters/11-fast-generators/06-closing-remarks.md)

## 结语与展望 {#epilogue-and-outlook}

12. [统一原理与未来之路](chapters/12-epilogue/index.md)

    - [12.1 密度传输这一共同基础](chapters/12-epilogue/01-density-transport-backbone.md)
    - [12.2 超越连续状态：离散空间中的概率演化](chapters/12-epilogue/02-discrete-spaces.md)
    - [12.3 全书结语](chapters/12-epilogue/03-closing-remarks.md)

## 附录 {#appendices}

- **A.** [微分方程速成](appendices/a-differential-equations/index.md)
    - [A.1 常微分方程基础](appendices/a-differential-equations/01-ordinary-differential-equations.md)
    - [A.2 随机微分方程基础](appendices/a-differential-equations/02-stochastic-differential-equations.md)
- **B.** [密度演化：从变量替换到 Fokker–Planck 方程](appendices/b-density-evolution/index.md)
    - [B.1 变量替换公式：从确定性映射到随机流](appendices/b-density-evolution/01-change-of-variable.md)
    - [B.2 连续性方程的直观理解](appendices/b-density-evolution/02-continuity-equation-intuition.md)
    - [B.3 （选读）将 Wasserstein 梯度流视为分布层面的训练](appendices/b-density-evolution/03-wasserstein-gradient-flows.md)
- **C.** [扩散模型的幕后：Itô 微积分与 Girsanov 定理](appendices/c-ito-calculus/index.md)
    - [C.1 Itô 公式：随机过程的链式法则](appendices/c-ito-calculus/01-ito-formula.md)
    - [C.2 测度变换：扩散模型中的 Girsanov 定理](appendices/c-ito-calculus/02-girsanov-theorem.md)
- **D.** [补充材料与证明](appendices/d-supplementary-proofs/index.md)
    - [D.1 变分视角](appendices/d-supplementary-proofs/01-variational-perspective.md)
    - [D.2 基于 score 的视角](appendices/d-supplementary-proofs/02-score-based-perspective.md)
    - [D.3 基于流的视角](appendices/d-supplementary-proofs/03-flow-based-perspective.md)
    - [D.4 理论补充：扩散模型的统一与系统视角](appendices/d-supplementary-proofs/04-unified-perspective.md)
    - [D.5 理论补充：学习快速扩散生成器](appendices/d-supplementary-proofs/05-fast-generators.md)
    - [D.6 （选读）Elucidating Diffusion Model（EDM）](appendices/d-supplementary-proofs/06-edm.md)

## 参考文献 {#references}

[前言、各章及附录引用的参考文献](references.md)

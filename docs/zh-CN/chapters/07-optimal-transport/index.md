# 7. （选读）扩散模型与最优传输 {#ch-ot-eot}

将一个分布映射为另一个分布（生成是其中的特例）是一项核心挑战。流匹配通过学习随时间变化的速度场，将质量从源分布传输到目标分布，从而处理这一问题。这自然地联系到传输理论：经典最优传输寻找分布之间代价最小的路径，而其熵正则化形式，即 Schrödinger 桥，则相对于 Brownian 运动等参考过程，选择最可能的受控扩散。

本章回顾最优传输、熵正则化最优传输和 Schrödinger 桥的基础，它们都是分布到分布问题的数学表述。这引出了一个核心问题：扩散模型在多大程度上实现了这样的最优传输？扩散模型有两种视角：由前向和逆向 SDE 定义的*随机过程*，以及由概率流 ODE（PF-ODE）给出的*确定性过程*。随机视角直接对应熵正则化最优传输，而 PF-ODE 一般并不对应任何已知的传输目标。这一差距留下了一个开放问题：在什么条件下，可以认为扩散模型求解了最优传输问题？

**读完本章后，你将能够……**

-   区分传输耦合、确定性传输映射、动态传输流和随机 Schrödinger 桥，并从概念上将它们与深度生成建模联系起来；

-   理解经典 OT、熵正则化 OT 和 Schrödinger 桥问题各自优化什么，以及这些表述之间的关系；

-   解释标准扩散模型的前向／逆向 SDE 为什么一般不能求解给定两个端点的 Schrödinger 桥问题；

-   解释扩散模型的 PF-ODE 为什么定义了确定性传输映射，却一般不等于二次代价下的最优传输映射。

**阅读路线。** 对于理解扩散模型的主线，本章是选读内容。如果你想精确理解“扩散在何种意义上属于最优传输，或者是否属于最优传输”，可以阅读本章。

继续阅读下面各节及[参考文献](../../references.md#references)。

[7.1 分布到分布转换引言](01-distribution-translation.md#sec-distribution-translation)

[7.2 问题设定的分类](02-transport-problem-setups.md#sec-ot-intro)

[7.3 不同最优传输表述之间的关系](03-relationships-between-formulations.md#sec-relation-ot)

[7.4 扩散模型的 SDE 是 SB 问题的最优解吗？](04-diffusion-sdes-and-schrodinger-bridges.md#sec-dm-and-sb)

[7.5 扩散模型的 ODE 是 OT 问题的最优映射吗？](05-probability-flow-odes-and-optimal-transport.md#sec-pfode-ot)

[7.6 结语](06-closing-remarks.md#sec-ch7-cr)

<span id="sec-ot-intro"></span>

[第 7.2 节](02-transport-problem-setups.md#sec-ot-intro)

<span id="eq-monge-formulation"></span>

[式 (7.2.2)](02-transport-problem-setups.md#eq-monge-formulation)

<span id="eq-monge-ampere"></span>

[式 (7.2.4)](02-transport-problem-setups.md#eq-monge-ampere)

<span id="subsec-ot-linear-reflow"></span>

[第 7.5.2 节](05-probability-flow-odes-and-optimal-transport.md#subsec-ot-linear-reflow)

<span id="eq-reflow-loss-gradient"></span>

[式 (7.5.2)](05-probability-flow-odes-and-optimal-transport.md#eq-reflow-loss-gradient)

<span id="ot-conditional-flow"></span>

[命题 7.5.1](05-probability-flow-odes-and-optimal-transport.md#ot-conditional-flow)

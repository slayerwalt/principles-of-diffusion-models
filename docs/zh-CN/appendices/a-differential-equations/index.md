# A. 微分方程速成 {#app-de}

微分方程（DE）是动态系统建模的基本工具，大体可分为*常微分方程*（ODE）、*随机微分方程*（SDE）和*偏微分方程*（PDE）。

ODE 描述系统状态如何按精确规则随时间变化，因此只要知道起点，就能精确确定未来路径。SDE 在这一演化中加入随机性，描述噪声或不确定性如何影响系统行为，使结果具有概率性而非固定不变。PDE 描述依赖多个变量（例如时间和空间）的函数如何共同演化，刻画热扩散、波传播，以及*随机系统中概率密度随时间的演化*等现象（提前揭晓：Fokker–Planck 方程）。这些微分方程构成基本语言，用于理解系统在确定性与随机因素影响下如何随时间和空间演化。

本章提供微分方程方面的必要预备知识。

**学完本章后，你将能够……**

-   从速度场、解轨迹及存在唯一性的作用理解 ODE；

-   利用积分因子揭示半线性 ODE 的线性结构，并理解这一结构为何有助于扩散 ODE 求解器；

-   区分单步、多步和时间并行的数值视角，并认识 Euler、Runge–Kutta、Adams–Bashforth 和 Picard 迭代这些代表性例子；

-   理解 SDE 如何通过 Brownian 噪声扩展确定性动力学，其微分记号如何与 Itô 积分相联系，以及 Euler–Maruyama 方法如何离散化所得随机动力学。

**阅读路线。** 本附录提供本书使用的微分方程背景。重点关注动力学、轨迹与数值近似之间的联系；后续扩散章节将这些思想具体应用于概率流 ODE 和随机扩散过程。

请继续阅读以下各节及[参考文献](../../references.md#references)。

[A.1 常微分方程基础](01-ordinary-differential-equations.md#app-ode-intro)

[A.2 随机微分方程基础](02-stochastic-differential-equations.md#app-sde-intro)

## 专题选读 {#selected-topics}

<span id="app-exp-int-factor"></span>

[第 A.1.3 节](01-ordinary-differential-equations.md#app-exp-int-factor)

<span id="app-de-solver"></span>

[第 A.1.4 节](01-ordinary-differential-equations.md#app-de-solver)

<span id="eq-classic-solvers"></span>

[公式 (A.1.6)](01-ordinary-differential-equations.md#eq-classic-solvers)

<span id="app-de-solver-examples"></span>

[第 A.1.5 节](01-ordinary-differential-equations.md#app-de-solver-examples)

<span id="app-sde-intro"></span>

[第 A.2 节](02-stochastic-differential-equations.md#app-sde-intro)

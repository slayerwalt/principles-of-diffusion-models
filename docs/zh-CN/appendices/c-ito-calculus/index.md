# C. 扩散模型背后的数学：Itô 微积分与 Girsanov 定理 {#app-ito}

（基于 score 的）扩散模型建立在 SDE 之上：漂移项推动状态变化，Brownian 项使状态产生随机扰动。与 ODE 路径不同，Brownian 路径处处不可微，因此普通链式法则不再适用。本附录介绍两种使这些数学表述精确成立的基本工具：

-   **Itô 公式**是随机轨迹所适用的链式法则。它说明函数 ${\mathbf{h}}({\mathbf{x}}_t,t)$ 在 ${\mathbf{x}}_t$ 遵循 SDE 时如何演化。借助它可以推导 Fokker–Planck 方程、矩的动力学、Itô 乘积法则，以及基于 score 的训练中使用的恒等式。

-   **Girsanov 定理**是关于*路径概率*的测度变换结果。它量化了噪声保持不变、漂移发生改变时，似然如何变化。这将似然加权的 score 匹配与路径空间上的 KL散度联系起来，并通过变分界进一步联系到数据似然。

有了这些工具，扩散模型中的标准推导，包括 Fokker–Planck 方程、逆时间 SDE、训练目标和似然关系，都可以明确地给出，无须依赖含糊的直觉论证。

**读完本附录后，你将能够……**

-   理解为什么普通微积分不足以处理 SDE，以及 Itô 公式如何给出随机轨迹适用的链式法则；

-   看出 Itô 公式如何支撑扩散模型中的重要推导，包括 Fokker–Planck 方程和线性 SDE；

-   将 Girsanov 定理理解为一种比较随机路径概率的方法：两种过程的漂移不同，但扩散相同；

-   区分逐时刻匹配分布与匹配完整随机路径，并理解似然加权的 score 匹配为何与路径空间 KL散度及数据似然相关。

**阅读路线。** 本附录介绍扩散模型背后的一些数学工具。Itô 微积分使随机微分的运算具有精确定义，Girsanov 定理则解释改变漂移会如何改变整条随机路径的概率。这些工具支撑着扩散模型中的多种标准推导，它们常被直接使用，却未被明确展示。

继续阅读以下各节及[参考文献](../../references.md#references)。

[C.1 Itô 公式：随机过程的链式法则](01-ito-formula.md#sec-ito-formula)

[C.2 测度的变量替换：扩散模型中的 Girsanov 定理](02-girsanov-theorem.md#sec-girsanov-theorem)

<span id="subsec-rigorous-proof-fpe"></span>

[第 C.1.4 节](01-ito-formula.md#subsec-rigorous-proof-fpe)

<span id="subsec-rigorous-proof-linear-sde"></span>

[第 C.1.5 节](01-ito-formula.md#subsec-rigorous-proof-linear-sde)

<span id="subsec-girsanov"></span>

[第 C.2.1 节](02-girsanov-theorem.md#subsec-girsanov)

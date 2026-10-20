# 11.6 结语 {#sec-ch11-cr}

最后这一章使我们的探索首尾相接：从缓慢的迭代扩散采样器，走向快速的少步生成器（从头学习得到）。本章各方法背后的共同对象，是[公式 (10.1.4)](01-prologue.md#eq-general-cm-loss-rep)引入的 oracle 流映射：

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
{\mathbf{x}}_s+\int_s^t {\mathbf{v}}^*({\mathbf{x}}_\tau,\tau)\,\mathop{}\!\mathrm{d} \tau.
$$

学习这一映射，便可用一次或少量学得的跳跃替代许多细小的数值更新。

此时，可以暂时跳出各个算法来整体审视。Knowledge Distillation（[第 10.1 节](../10-distillation/01-prologue.md#sec-distillation-prologue)）、Progressive Distillation（[第 10.3 节](../10-distillation/03-progressive-distillation.md#sec-pd)）、CM（[第 11.2 节](02-discrete-consistency-models.md#sec-discrete-cm)）、CTM（[第 11.4 节](04-consistency-trajectory-models.md#sec-ctm)）及 MF（[第 11.5 节](05-mean-flow.md#sec-mf)）通过不同的训练构造提出，但都显式或隐式地利用了同一个 ODE 解映射结构。

流映射本身是微分方程和动力系统中的经典对象，与连续介质力学及流体力学中基于粒子和基于场的描述密切相关；参见[公式 (4.2.2)](../04-score-sde-framework/02-matching-marginals.md#eq-flow-map-def)及附近的历史讨论。这里，我们用同一个 ODE 流的三种经典描述，即*半群*、*Lagrangian* 和 *Eulerian* 视角，来组织上文介绍的现代学习方法[^1]。

**半群视角：拆分行程。** 如[公式 (11.2.1)](02-discrete-consistency-models.md#eq-semi-group)所介绍，有限步恒等式为

$$
{\boldsymbol{\Psi}}_{u\to t}\big({\boldsymbol{\Psi}}_{s\to u}({\mathbf{x}}_s)\big)
=
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s).
$$

如[图 11.3](02-discrete-consistency-models.md#fig-semigroup)所示，其直观含义与沿固定路线行进相同：从 $s$ 到 $u$，再从 $u$ 到 $t$，与直接从 $s$ 到 $t$ 会到达同一点。在这一视角下，KD 学习完整转移 ${\boldsymbol{\Psi}}_{T\to0}$，PD 将教师两个相邻步骤压缩为学生的一步，CM 学习任意时刻到干净数据的特殊映射 ${\boldsymbol{\Psi}}_{s\to0}$，CTM 则将同一复合原则扩展到一般映射 ${\boldsymbol{\Psi}}_{s\to t}$。

<span id="fig-lagrangian-consistency"></span>

![Lagrangian 一致性沿固定 PF-ODE 轨迹移动终止时刻](../../../en/assets/chapter-11/lagrangian-consistency.png)

<span id="fig-lagrangian-base"></span>

**(a)** 原始终止时刻 $t$。

<span id="fig-lagrangian-moved"></span>

**(b)** 移动后的终止时刻 $t'$。

**图 11.7：** **流映射的 Lagrangian 一致性。** 起始状态与时间对 $({\mathbf{x}}_s,s)$ 固定，终止时刻从 $t$ 移到 $t'$。流映射的输出沿同一 PF-ODE 轨迹移动，其无穷小变化等于 ODE 速度：$\partial_t {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s) = {\mathbf{v}}^*({\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s),t)$。

来源：改编自 [Dieleman (2026)](../../references.md#ref-dieleman2026flowmaps)。

**Lagrangian 视角：移动终点。** 固定起始状态与时间对 $({\mathbf{x}}_s,s)$，改变终止时刻 $t$，终点便沿同一轨迹移动：

$$
\partial_t {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
{\mathbf{v}}^*({\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s),t).
$$

“Lagrangian”这一名称沿用经典的粒子视角：随着终止时刻改变，我们沿轨迹跟踪终点。在无穷小尺度上，终点以该点处的局部 ODE 速度运动。

**Eulerian 视角：移动起点。** 固定终止时刻 $t$，让起点沿同一轨迹滑动，时刻 $t$ 的终点保持不变：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)=0.
$$

由于 $\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{x}}_s={\mathbf{v}}^*({\mathbf{x}}_s,s)$，由链式法则可得

$$
\partial_s{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
+
\big(\partial_{{\mathbf{x}}}{\boldsymbol{\Psi}}_{s\to t}\big)({\mathbf{x}}_s){\mathbf{v}}^*({\mathbf{x}}_s,s)
=
0.
$$

“Eulerian”这一术语借用了经典的场视角：这里终止时刻 $t$ 固定，而起始状态与时间对 $({\mathbf{x}}_s,s)$ 沿同一 ODE 轨迹移动，因此时刻 $t$ 的终点保持不变。对于 CM，终止时刻固定为 $t=0$，所以终点就是干净样本：

$$
{\mathbf{f}}^*({\mathbf{x}}_s,s)={\boldsymbol{\Psi}}_{s\to0}({\mathbf{x}}_s).
$$

因此，[公式 (11.3.1)](03-continuous-consistency-models.md#eq-conti-ct-motivation)是[第 11.2 节](02-discrete-consistency-models.md#sec-discrete-cm)和[第 11.3 节](03-continuous-consistency-models.md#sec-conti-cm)所研究的特殊流映射关于起始时间的一致性条件。

这一视角也概括了[第 11.5.3 节](05-mean-flow.md#subsec-equivalence-ctm-mf)中 CTM 与 MF 的关系。二者近似同一个路径积分

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
{\mathbf{x}}_s+\int_s^t{\mathbf{v}}^*({\mathbf{x}}_\tau,\tau)\,\mathop{}\!\mathrm{d}\tau,
$$

但提供不同的可训练替代量。CTM 通过位移形式的参数化表示终点，MF 则通过平均漂移表示同一个终点：

$$
{\mathbf{h}}^*({\mathbf{x}}_s,s,t)
=
\frac{1}{t-s}\int_s^t{\mathbf{v}}^*({\mathbf{x}}_\tau,\tau)\,\mathop{}\!\mathrm{d}\tau.
$$

将

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)={\mathbf{x}}_s+(t-s){\mathbf{h}}^*({\mathbf{x}}_s,s,t)
$$

代入 Eulerian 起始时间恒等式，得到

$$
{\mathbf{h}}^*({\mathbf{x}}_s,s,t)
=
{\mathbf{v}}^*({\mathbf{x}}_s,s)
-
(s-t)
\left[
\partial_s{\mathbf{h}}^*({\mathbf{x}}_s,s,t)
+
(\partial_{{\mathbf{x}}}{\mathbf{h}}^*)({\mathbf{x}}_s,s,t){\mathbf{v}}^*({\mathbf{x}}_s,s)
\right].
$$

这就是基本的 Mean Flow 恒等式：长区间上的平均漂移，等于起点的瞬时漂移，再根据起点沿同一轨迹滑动时平均漂移的变化加以修正。

<span id="fig-eulerian-consistency"></span>

![Eulerian 一致性改变起始状态和时间，同时固定终点](../../../en/assets/chapter-11/eulerian-consistency.png)

<span id="fig-eulerian-base"></span>

**(a)** 原始起始时刻 $s$。

<span id="fig-eulerian-moved"></span>

**(b)** 移动后的起始时刻 $s'$。

**图 11.8：** **流映射的 Eulerian 一致性。** 终止时刻 $t$ 固定，起始时刻从 $s$ 移到 $s'$。由于相应起点位于同一 PF-ODE 轨迹上，终点保持不变：${\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)={\boldsymbol{\Psi}}_{s'\to t}({\mathbf{x}}_{s'})$。在无穷小尺度上，这给出 $\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)=0$，等价地也可写为 $\partial_s{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s) +(\partial_{{\mathbf{x}}}{\boldsymbol{\Psi}}_{s\to t})({\mathbf{x}}_s){\mathbf{v}}^*({\mathbf{x}}_s,s)=0$。

来源：改编自 [Dieleman (2026)](../../references.md#ref-dieleman2026flowmaps)。

**三种视角的数学等价性。** 对于精确的 ODE 流，半群、Lagrangian 和 Eulerian 恒等式都表达同一条简单原则：在标准的存在性、唯一性与光滑性假设下，ODE 确定经过每个状态与时间对的唯一轨迹。也就是说，一旦固定某个时刻的状态，就只有一条可沿循的路径。半群视角将路径拆成连续片段，Lagrangian 视角随着终止时刻变化跟踪终点，Eulerian 视角则让起点沿同一路径移动，同时保持终点不变。因此，这三个恒等式是同一个 ODE 流的等价描述（[Arnold, 1988](../../references.md#ref-arnold1988geometrical)；[Chorin and Marsden, 1993](../../references.md#ref-chorin1993mathematical)）。将半群恒等式视为有限区间的表述，将 Lagrangian 和 Eulerian 恒等式视为分别通过缩短终止时刻或起始时刻附近的短区间得到的两种无穷小形式，两者的联系就尤为清楚。

半群恒等式给出了这一原则的有限区间形式：

$$
{\boldsymbol{\Psi}}_{u\to t}\big({\boldsymbol{\Psi}}_{s\to u}({\mathbf{x}}_s)\big)
    =
    {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s),
    \qquad
    {\boldsymbol{\Psi}}_{s\to s}={\mathbf{I}} .
$$

将两个区间中的一个缩短，便得到两种微分视角。

**终止时间极限：Lagrangian 恒等式。** 固定起始状态与时间对 $({\mathbf{x}}_s,s)$，缩短行程的最后一段。对于 $\Delta t>0$，

$$
{\boldsymbol{\Psi}}_{t\to t-\Delta t}
    \big({\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)\big)
    =
    {\boldsymbol{\Psi}}_{s\to t-\Delta t}({\mathbf{x}}_s).
$$

令 ${\mathbf{x}}_t:={\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$。对短时间流作一阶展开，得到

$$
{\boldsymbol{\Psi}}_{t\to t-\Delta t}({\mathbf{x}}_t)
    =
    {\mathbf{x}}_t-\Delta t\,{\mathbf{v}}^*({\mathbf{x}}_t,t)
    +\mathcal{O}\big((\Delta t)^2\big).
$$

因此，

$$
\frac{
    {\boldsymbol{\Psi}}_{s\to t-\Delta t}({\mathbf{x}}_s)
    -
    {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)}
    {-\Delta t}
    =
    {\mathbf{v}}^*({\mathbf{x}}_t,t)
    +\mathcal{O}(\Delta t).
$$

令 $\Delta t\to0$，得到 Lagrangian 恒等式

$$
\partial_t{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    =
    {\mathbf{v}}^*\big({\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s),t\big).
$$

因此，当起始状态与时间对 $({\mathbf{x}}_s,s)$ 固定、终止时刻变化时，终点以该点处的 ODE 速度移动。

**起始时间极限：Eulerian 恒等式。** 现在固定终止时刻 $t$，将起始状态与时间对沿同一轨迹向后移动一个短区间 $\Delta s>0$：

$$
{\mathbf{x}}_{s-\Delta s}
    :=
    {\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)
    =
    {\mathbf{x}}_s-\Delta s\,{\mathbf{v}}^*({\mathbf{x}}_s,s)
    +\mathcal{O}\big((\Delta s)^2\big).
$$

由半群恒等式可得

$$
{\boldsymbol{\Psi}}_{s-\Delta s\to t}({\mathbf{x}}_{s-\Delta s})
    =
    {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s).
$$

对左侧同时关于起始状态与起始时间展开，得到

$$
{\boldsymbol{\Psi}}_{s-\Delta s\to t}({\mathbf{x}}_{s-\Delta s})
    =
    {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    -
    \Delta s\,
    \big(\partial_{{\mathbf{x}}}{\boldsymbol{\Psi}}_{s\to t}\big)({\mathbf{x}}_s){\mathbf{v}}^*({\mathbf{x}}_s,s)
    -
    \Delta s\,
    \partial_s{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    +
    \mathcal{O}\big((\Delta s)^2\big).
$$

与半群恒等式比较，并除以 $-\Delta s$，得到

$$
\partial_s{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    +
    \big(\partial_{{\mathbf{x}}}{\boldsymbol{\Psi}}_{s\to t}\big)({\mathbf{x}}_s){\mathbf{v}}^*({\mathbf{x}}_s,s)
    =
    \mathcal{O}(\Delta s).
$$

令 $\Delta s\to0$，得到 Eulerian 恒等式

$$
\partial_s{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    +
    \big(\partial_{{\mathbf{x}}}{\boldsymbol{\Psi}}_{s\to t}\big)({\mathbf{x}}_s){\mathbf{v}}^*({\mathbf{x}}_s,s)
    =
    0.
$$

这里，终止时刻 $t$ 固定，起始状态与时间对沿同一轨迹移动。因此，终点保持不变。

**由任一微分恒等式恢复流映射。** 反过来，任一微分恒等式结合边界条件 ${\boldsymbol{\Psi}}_{s\to s}={\mathbf{I}}$，都能恢复同一流映射。Lagrangian 恒等式使终点从 ${\boldsymbol{\Psi}}_{s\to s}({\mathbf{x}}_s)={\mathbf{x}}_s$ 演化。Eulerian 恒等式说明，${\boldsymbol{\Psi}}_{\tau\to t}({\mathbf{x}}_\tau)$ 沿特征线 ${\mathbf{x}}_\tau={\boldsymbol{\Psi}}_{s\to \tau}({\mathbf{x}}_s)$ 保持常量，因此

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    =
    {\boldsymbol{\Psi}}_{t\to t}({\mathbf{x}}_t)
    =
    {\mathbf{x}}_t.
$$

尽管这些视角在 oracle 层面等价，但引入近似后会得到不同的实际算法：必须选择施加哪条恒等式、在何处设置停止梯度目标、如何采样时间对，以及如何参数化映射。

**结束语。** 本章讨论的核心转变，是从学习局部速度场转向学习其 ODE 解映射的有限片段。传统扩散采样器反复计算局部场并进行数值积分。流映射模型则在训练阶段摊销部分积分工作，使一次或少量网络计算就能跨越较长区间。

这为快速生成提供了直接途径，但也将难点从采样转移到了学习。模型必须近似一个更全局的对象，其效果取决于映射参数化、可计算目标的构造、时间对的采样与加权，以及优化的稳定化。

从这一角度看，流映射学习是围绕经典动力学概念构建的现代学习策略。它保留扩散模型基于轨迹的基础，同时扩大高效生成的设计空间。目标不再仅是在推理时更高效地积分生成 ODE，还包括直接学习其解映射中有用的有限时间转移。

[^1]: 在现代生成建模中，[Boffi et al. (2025)](../../references.md#ref-boffi2024flow) 汇集了 PD、CM、CTM 和 MF 中已有的结构思想，用共同语言描述扩散 PF-ODE 流映射的学习。因此，主要组成部分大多来自这些已有研究方向。本节的论述与插图也参考了 [Dieleman (2026)](../../references.md#ref-dieleman2026flowmaps) 的阐释。

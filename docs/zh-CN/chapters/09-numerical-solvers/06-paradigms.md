# 9.6 （选读）ParaDiGMs {#sec-picard}

<span id="fig-computation-graphs"></span>

![顺序时间步进与 Picard 迭代的计算图](../../../en/assets/chapter-09/computation-graphs.png)

**图 9.3：** **两种计算图的比较。** 左：常规时间步进 ODE 求解，解按时间顺序传播。右：Picard 迭代使用前一次迭代的结果同时更新所有时间节点，从而实现并行计算，避免时间步进严格顺序执行的限制。

来源：改编自 [Shih et al. (2023)](../../references.md#ref-shih2023parallel)。

<span id="fig-chaingraph"></span>

**(a)** 在生成过程中通过时间步进估计进行顺序采样。

<span id="fig-connectedgraph"></span>

**(b)** 具有跨步依赖的 Picard 迭代。

## 9.6.1 从时间步进到时间并行求解器 {#time-parallel-solver}

前面各节关注时间步进方法，它从先验时刻 $T$ 向任意 $t\in[0,T]$ 演化，以估计轨迹。令

$$
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}({\mathbf{x}},t):={\mathbf{f}}({\mathbf{x}}, t) - \frac{1}{2}g^2(t) {\mathbf{s}}_{{\boldsymbol{\phi}}^\times}({\mathbf{x}},t)
$$

表示预训练扩散模型给出的经验 PF-ODE 漂移。从 $T$ 到任意中间时刻 $t$ 的精确演化为：

<span id="eq-picard-t-to-t"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{T\to t}\big({\mathbf{x}}(T)\big)
    =
    {\mathbf{x}}(T)
    + \int_{T}^{t}{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}\big({\mathbf{x}}(\tau),\tau\big)\mathop{}\!\mathrm{d}\tau,
    \quad {\mathbf{x}}(T)\sim p_{\mathrm{prior}}.
\tag{9.6.1}
$$

时间步进格式通过基于过去时间步的离散更新来近似这一积分。

本节转向*时间并行*方法，以 *ParaDiGMS* 为例。它基于经典 Picard 迭代，实现跨时间的并行积分。ParaDiGMS 的关键思想是*用计算资源换取更快的模拟速度*。

## 9.6.2 ParaDiGMS 的方法 {#methodology}

**从轨迹到作为不动点更新的 Picard 迭代。** 固定逆时间采样的初始状态，

$$
{\mathbf{x}}_T\sim p_{\mathrm{prior}}.
$$

[式 (9.6.1)](06-paradigms.md#eq-picard-t-to-t)中的积分表达式可以看作一个映射：输入整条候选轨迹，输出更新后的轨迹。形式上，给定任意路径 $\{{\mathbf{y}}(\tau)\}_{\tau\in[0,T]}$，定义

$$
(\mathcal{L}[{\mathbf{y}}(\cdot)])(t)
=
{\mathbf{x}}_T
+
\int_T^t
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}\big({\mathbf{y}}(\tau),\tau\big)\mathop{}\!\mathrm{d}\tau,
\qquad
t\in[0,T].
$$

也就是说，$\mathcal{L}$ 保持初始状态 ${\mathbf{x}}_T$ 固定，并使用沿候选路径计算的速度，更新轨迹其余部分。

经验 PF-ODE 的解轨迹 ${\mathbf{x}}^*(\cdot)$，满足 ${\mathbf{x}}^*(T)={\mathbf{x}}_T$，恰好就是这一映射的*不动点*：

$$
{\mathbf{x}}^*(t)
=
\mathcal{L}[{\mathbf{x}}^*(\cdot)](t)
=
{\mathbf{x}}_T
+
\int_T^t
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}\big({\mathbf{x}}^*(\tau),\tau\big)\mathop{}\!\mathrm{d}\tau.
$$

因此，可以寻找一条与这一积分方程一致的完整轨迹，不必严格逐个时间步构造。

从初始猜测 ${\mathbf{x}}^{(0)}(\cdot)$ 出发，要求满足 ${\mathbf{x}}^{(0)}(T)={\mathbf{x}}_T$，例如常数路径 ${\mathbf{x}}^{(0)}(t)\equiv{\mathbf{x}}_T$，Picard 迭代反复应用这一映射：

<span id="eq-picard-iter-t-to-t"></span>

$$
\begin{aligned}
{\mathbf{x}}^{(k+1)}(t)
=
\mathcal{L}[{\mathbf{x}}^{(k)}(\cdot)](t)
=
{\mathbf{x}}_T
+
\int_T^t
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}
\big({\mathbf{x}}^{(k)}(\tau),\tau\big)\mathop{}\!\mathrm{d}\tau,
\end{aligned}
\tag{9.6.2}
$$

其中 $k=0,1,2,\ldots$。每次迭代沿前一条轨迹计算速度，再用这些结果更新所有时间点的轨迹。

**从 $T$ 到 $0$ 网格上的离散 Picard。** 为将[式 (9.6.2)](06-paradigms.md#eq-picard-iter-t-to-t)转为实际算法，在 $[0,T]$ 上放置均匀递减网格：选择步数 $M$，令 $\Delta t := T/M$，并定义

$$
t_j := T - j \Delta t,
\quad j=0,1,\ldots,M,
$$

因此 $t_0=T$ 且 $t_M=0$。将采样节点上的迭代值记为

$$
{\mathbf{x}}^{(k)}_j := {\mathbf{x}}^{(k)}(t_j).
$$

由于网格沿时间反向排列，从 $T$ 到 $t_j$ 的积分具有负方向。在划分 $\{[t_{i+1},t_i]\}_{i=0}^{j-1}$ 上使用左端点近似，得到

$$
\int_{T}^{t_j}{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}({\mathbf{x}}^{(k)}(\tau), \tau) \mathop{}\!\mathrm{d}\tau
 \approx
- \Delta t \sum_{i=0}^{j-1} {\mathbf{v}}_{{\boldsymbol{\phi}}^\times}\big({\mathbf{x}}^{(k)}_{i}, t_i\big),
$$

因为每个 $[t_{i+1},t_i]$ 上的小积分等于 $- \int_{t_i}^{t_{i+1}} \cdot\mathop{}\!\mathrm{d}\tau$。将此近似代入[式 (9.6.2)](06-paradigms.md#eq-picard-iter-t-to-t)，得到离散 Picard 更新

<span id="eq-discrete-picard-decreasing"></span>

$$
{\mathbf{x}}^{(k+1)}_j
=
{\mathbf{x}}_T
-
\Delta t
\underbrace{
\sum_{i=0}^{j-1}
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}
\big({\mathbf{x}}^{(k)}_i,t_i\big)
}_{\substack{\text{cumulative sum}\\\text{of drifts}}},
\qquad
j=1,\ldots,M.
\tag{9.6.3}
$$

这一格式简单且便于并行：每个漂移计算 ${\mathbf{v}}_{{\boldsymbol{\phi}}^\times} \big({\mathbf{x}}^{(k)}_i,t_i\big)$ 只依赖同一时间节点 $t_i$ 处的前次迭代值，因此网格上的全部 $i=0,\dots,j - 1$ 次计算可以独立进行。随后通过累加恢复积分，累加可串行执行，也可使用并行*前缀和（扫描／滑动窗口）*。

**滑动窗口与并行计算。**

<span id="fig-picard-1"></span>

![在包含四步的滑动窗口内并行计算漂移](../../../en/assets/chapter-09/paradigms-parallel-drift.png)

**图 9.4：** 并行计算 ${\mathbf{x}}^{(k)}_{\ell:\ell+p}$ 的漂移，批处理窗口大小为 $p=4$。

<span id="fig-picard-2"></span>

![累积漂移更新当前窗口内的状态](../../../en/assets/chapter-09/paradigms-cumulative-update.png)

**图 9.5：** 利用窗口内各点的累积漂移，将值更新为 ${\mathbf{x}}^{(k+1)}_{\ell:\ell+p}$。

<span id="fig-picard"></span>

<span id="fig-picard-3"></span>

![误差控制计算窗口向前移动的距离](../../../en/assets/chapter-09/paradigms-sliding-window.png)

**图 9.6：** 根据误差 $\lVert {\mathbf{x}}^{(k+1)}_i - {\mathbf{x}}^{(k)}_i \rVert^2$ 确定窗口向前滑动的距离。

来源：改编自 [Shih et al. (2023)](../../references.md#ref-shih2023parallel)。

离散 Picard 更新[式 (9.6.3)](06-paradigms.md#eq-discrete-picard-decreasing)将每个 ${\mathbf{x}}^{(k+1)}_j$ 表示为固定初值 ${\mathbf{x}}_T$ 减去漂移的累积和。为限制内存使用并利用并行硬件，可以在短的滑动索引块上局部应用同一思想。

固定窗口长度 $p$ 及左索引 $\ell$；窗口覆盖 $j=\ell,\ldots,\ell+p$，其中 $t_\ell>t_{\ell+1}>\cdots>t_{\ell+p}$。在迭代 $k$ 中：

**步骤 1. 窗口内并行计算漂移。** 只使用前次迭代值，并行计算

$$
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times} \big({\mathbf{x}}^{(k)}_{\ell+i}, t_{\ell+i}\big),
\quad i=0,1,\ldots,p-1.
$$

这些是所需的 $p$ 个局部增量，用于从左边界 $t_\ell$ 跨越窗口推进。

**步骤 2. 以左边界为锚点的累积更新。** 以 $j=\ell$ 为锚点，累加各子区间的漂移，构造窗口更新：

<span id="eq-window-update"></span>

$$
{\mathbf{x}}^{(k+1)}_{\ell+j+1}
 =
{\mathbf{x}}^{(k)}_{\ell}
 - \Delta t \sum_{i=0}^{j}
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times} \big({\mathbf{x}}^{(k)}_{\ell+i}, t_{\ell+i}\big),
\quad j=0,1,\ldots,p-1.
\tag{9.6.4}
$$

这是[式 (9.6.3)](06-paradigms.md#eq-discrete-picard-decreasing)的局部窗口版本，当前左边界值 ${\mathbf{x}}^{(k)}_\ell$ 作为窗口锚点。负号反映时间递减方向。内部求和是窗口内漂移的前缀和，因此所有部分和都可以在并行硬件上高效产生。

**步骤 3. 进度控制与窗口前移。** 在当前窗口完成以左边界为锚点的累积更新（步骤 2）后，需要决定窗口滑动多远。使用逐点 Picard 变化量度量局部收敛，

$$
\mathtt{error}_j
:=
\big\|{\mathbf{x}}^{(k+1)}_{\ell+j}-{\mathbf{x}}^{(k)}_{\ell+j}\big\|^2,
\quad j=1,\ldots,p-1,
$$

并与给定容差 $\mathtt{tol}_{\ell+j}$ 比较。也就是说，$\mathtt{error}_j$ 度量节点 $\ell+j$ 的迭代值在最近一次 Picard 更新中变化了多少。若这一数值较小，说明相邻两次近似局部一致，因而该节点的不动点迭代局部收敛。若较大，则该节点尚未稳定，需要更多 Picard 平滑更新。

*跨步长度*选为窗口内第一个未通过检验的索引（如果全部通过，则取完整窗口长度 $p$）：

$$
\mathtt{stride}
:=
\min\Big(\,\{\,j\ge 1:\ \mathtt{error}_j>\mathtt{tol}_{\ell+j}\,\}\ \cup\ \{p\}\Big).
$$

随后令 $\ell \leftarrow \ell+\mathtt{stride}$ 来滑动窗口。换言之，接受从左边界开始、直到第一个未收敛节点之前的所有节点（不包含该未收敛节点）；如果全部节点收敛，就接受整个窗口。然后按接受的节点数 $\ell \leftarrow \ell+\mathtt{stride}$ 滑动窗口，并继续。这最多前进一个窗口长度 $p$，绝不跳过尚未满足容差的节点。如果滑动会超过网格末端 $M$，就将窗口截短为 $p \leftarrow \min\{p,\,M-\ell\}$ 后继续。

窗口前移会暴露尚无数值的新时间节点。为在这些节点启动 Picard 迭代，只需复制窗口左边界的值作为初始猜测。这种“常数外推”成本低且稳定，并会被后续更新校正。需要时，也可换成更准确的初始猜测，例如根据过去节点进行线性或多项式外推。

至此，ParaDiGMS 的过程已完整给出。[算法 8](06-paradigms.md#alg-paradigms)总结了该算法。

<section class="book-algorithm" id="alg-paradigms" aria-labelledby="alg-paradigms-title" markdown="1">

<div class="book-algorithm__title" id="alg-paradigms-title" markdown="span">
算法 8：使用滑动窗口的 ParaDiGMS
</div>

<div class="book-algorithm__meta" markdown="1">

**输入：** 漂移 ${\mathbf{v}}_{{\boldsymbol{\phi}}^\times}({\mathbf{x}},t)$; $\{t_j\}_{j=0}^{M}$; 窗口长度 $p$; $\{\mathtt{tol}_j\}_{j=1}^{M}$

</div>

<div class="book-algorithm__meta" markdown="1">

**输出：** 近似轨迹 $\{{\mathbf{x}}^{(k)}_j\}_{j=0}^{M}$，其中 ${\mathbf{x}}^{(k)}_M$ 位于 $t=0$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="算法 8 的步骤" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

$k \gets 0$, $\ell \gets 0$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

采样 ${\mathbf{x}}^{(0)}_0 \sim p_{\mathrm{prior}}$；令 ${\mathbf{x}}^{(0)}_j \gets {\mathbf{x}}^{(0)}_0$，其中 $j=1,\ldots,\min(p,M)$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">▷ *常数外推*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

**while** $\ell &lt; M$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

$J \gets \min(p, M-\ell)$  <span class="book-algorithm__comment">▷ *current 窗口长度*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

**步骤 1：并行计算**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

For $i=0,\ldots,J-1$: $g_i \gets {\mathbf{v}}_{{\boldsymbol{\phi}}^\times} \big({\mathbf{x}}^{(k)}_{\ell+i},\, t_{\ell+i}\big)$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">▷ *使用前次迭代的漂移（Picard 冻结）*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">7</div>
<div class="book-algorithm__statement" markdown="1">

计算前缀和 $S_j \gets \sum_{i=0}^{j} g_i$，其中 $j=0,\ldots,J-1$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">▷ *扫描窗口内的漂移*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">8</div>
<div class="book-algorithm__statement" markdown="1">

**步骤 2：累积更新**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">9</div>
<div class="book-algorithm__statement" markdown="1">

For $j=0,\ldots,J-1$: $\displaystyle {\mathbf{x}}^{(k+1)}_{\ell+j+1} \gets {\mathbf{x}}^{(k)}_{\ell} - \Delta t\, S_j$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">▷ *以左边界为锚点的更新；参见[式 (9.6.4)](06-paradigms.md#eq-window-update)*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">10</div>
<div class="book-algorithm__statement" markdown="1">

**步骤 3：进度控制与窗口前移**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">11</div>
<div class="book-algorithm__statement" markdown="1">

For $j=1,\ldots,J-1$: $\mathtt{error}_j \gets \big\Vert {\mathbf{x}}^{(k+1)}_{\ell+j}-{\mathbf{x}}^{(k)}_{\ell+j}\big\Vert ^2$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">▷ *逐点 Picard 变化量*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">12</div>
<div class="book-algorithm__statement" markdown="1">

$\displaystyle \mathtt{stride} \gets \min \Big(\,\{\,j\in\{1,\ldots,J-1\}:~\mathtt{error}_j&gt;\mathtt{tol}_{\ell+j}\,\}\ \cup\ \{J\}\Big)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">13</div>
<div class="book-algorithm__statement" markdown="1">

**初始化新节点**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">14</div>
<div class="book-algorithm__statement" markdown="1">

For $r=1,\ldots,\min\{\mathtt{stride},\,M-\ell-J\}$: ${\mathbf{x}}^{(k+1)}_{\ell+J+r} \gets {\mathbf{x}}^{(k+1)}_{\ell+J}$

</div>
</div>

<div class="book-algorithm__line book-algorithm__line--comment" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

<span class="book-algorithm__comment">▷ *对新暴露的索引作常数外推*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">15</div>
<div class="book-algorithm__statement" markdown="1">

$\ell \gets \ell + \mathtt{stride}$; $k \gets k+1$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">16</div>
<div class="book-algorithm__statement" markdown="1">

**end while**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">17</div>
<div class="book-algorithm__statement" markdown="1">

**return** $\{{\mathbf{x}}^{(k)}_j\}_{j=0}^{M}$

</div>
</div>

</div>
</div>

</section>

## 9.6.3 与时间步进求解器的关系 {#time-stepping-solvers}

**滑动窗口大小的选择。** 为理解窗口大小的作用，先考虑最小选择 $p=1$。此时没有时间并行性，方法退化为其顺序基础更新。对于[式 (9.6.4)](06-paradigms.md#eq-window-update)中的左端点离散化，这就是普通 Euler 步。

更一般地，ParaDiGMS 规定不同时间点的计算如何并行，而各步使用的数值更新可以单独选择。例如，可以将 Euler 更新替换为 DDIM 或 DPM-Solver，得到 ParaDDIM 或 ParaDPMSolver。增大 $p$ 会让更多时间点参与并行计算，而不改变基础时间网格或总步数 $M$。Picard 迭代反复校正这些状态，使其与所选基础求解器一致，收敛由给定局部容差控制。

**与高阶求解器（如 DPM）的兼容性。** Picard 框架决定不同时间点的状态如何并行更新和计算，而用于近似每步的数值规则可以单独选择。

例如，可以用梯形规则替代[式 (9.6.4)](06-paradigms.md#eq-window-update)中的左端点规则：

$$
\begin{aligned}
{\mathbf{x}}^{(k+1)}_{\ell+j+1}
=
{\mathbf{x}}^{(k)}_{\ell}
-\Delta t\Bigg[
\tfrac{1}{2}
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}
\big({\mathbf{x}}^{(k)}_{\ell},t_{\ell}\big)
+
\sum_{i=1}^{j}
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}
\big({\mathbf{x}}^{(k)}_{\ell+i},t_{\ell+i}\big)
+
\tfrac{1}{2}
{\mathbf{v}}_{{\boldsymbol{\phi}}^\times}
\big({\mathbf{x}}^{(k)}_{\ell+j+1},t_{\ell+j+1}\big)
\Bigg],
\end{aligned}
$$

其中 $j=0,\ldots,p-1$。对于 $j=0$，这就是熟悉的、使用两个端点漂移的单步梯形更新。对于更大的 $j$，同一规则累加窗口内的梯形贡献。由于所有漂移均使用前次 Picard 迭代值计算，它们仍可并行计算，而内部贡献可通过前缀和累加。

更一般地，ParaDiGMS 可以将 DDIM 或 DPM-Solver 等现有顺序采样器作为基础离散化。在这种情况下，Picard 更新由相应基础求解器的增量构造，而非采用上述梯形公式。具体形式取决于所选求解器，因为高阶方法可能使用中间阶段或先前模型计算。

对于基于给定顺序求解器的 ParaDiGMS 构造，精确的不动点收敛会恢复相应顺序离散轨迹。使用有限停止容差时，最终结果除了基础求解器的离散化误差，还包含剩余的不动点迭代误差。

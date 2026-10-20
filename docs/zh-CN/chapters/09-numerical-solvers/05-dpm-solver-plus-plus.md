# 9.5 DPM-Solver`++` {#sec-dpm}

## 9.5.1 从 DPM-Solver 到用于引导采样的 DPM-Solver`++` {#guided-sampling}

DPM-Solver`++`（[Lu et al., 2022](../../references.md#ref-lu2022dpm2)）旨在改进强引导下的高阶采样。在这一情形中会出现两个实际困难。第一，较大的引导尺度会放大模型输出及其导数，缩小高阶求解器的收敛区域，因而需要更小的数值步长。第二，引导后的预测可能将最终解推到训练数据所覆盖的范围之外，产生[第 8 章](../08-guidance/index.md#ch-guidance)讨论的训练与测试不匹配。

DPM-Solver`++` 通过两种互补方式处理这些问题。一方面，用数据预测重新表述扩散 ODE，从而可以应用动态阈值处理等数据空间修正；另一方面，构造复用先前模型计算的多步变体，使固定 NFE 预算下能够使用更小的有效步长。下面先推导数据预测表述，再介绍相应的单步和多步二阶求解器。

## 9.5.2 DPM-Solver`++` 的方法 {#methodology}

DPM-Solver`++` 保留 DPM-Solver 引入的对数信噪比坐标 $\lambda$，但通过数据预测而非噪声预测表达经验 PF-ODE。回顾[式 (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)，

$$
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)
=
\frac{
{\mathbf{x}}_t-\alpha_t{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)
}{
\sigma_t
}.
$$

将这一关系代入[式 (9.4.5)](04-dpm-solver.md#eq-analytic-solution)中的噪声预测精确解，得到等价的数据预测形式

<span id="eq-dpmpp-data-exact"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
\frac{\sigma_t}{\sigma_s}{\mathbf{x}}_s
+
\sigma_t
\int_{\lambda_s}^{\lambda_t}
e^\lambda
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\mathop{}\!\mathrm{d}\lambda,
\tag{9.5.1}
$$

其中，沿用[式 (9.4.3)](04-dpm-solver.md#eq-change-of-var-x-epsilon)，将以 $\lambda$ 参数化的数据预测场定义为

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
({\mathbf{x}},\lambda)
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
\bigl(
{\mathbf{x}},
t_\lambda(\lambda)
\bigr).
$$

因此，沿经验 PF-ODE 轨迹，

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
\bigl(
{\mathbf{x}}_{t_\lambda(\lambda)},
t_\lambda(\lambda)
\bigr).
$$

这一结构与 DPM-Solver 恰好对偶：已知调度相关演化解析处理，而剩余积分中唯一与模型有关的量，现在是数据预测。DPM-Solver`++` 为这一精确形式构造了两种二阶实现：

-   **单步求解器 DPM-Solver`++`(2S)。** 在当前步内的中间阶段额外计算一次模型，类似 DPM-Solver-2 的分阶段构造。

-   **多步求解器 DPM-Solver`++`(2M)。** 复用前两个已完成步骤的数据预测。启动后，每步只需一次新的模型计算。

**说明。**

不使用阈值处理时，将模型输出转换为数据预测，只是对同一经验 PF-ODE 进行逐点代数改写。但数值方法仍可能改变，因为有限步近似现在作用于数据预测，而非噪声预测。如果使用阈值处理，它是对模型输出的额外修改，应与下面的数值阶数分析分开看待。

## 9.5.3 通过 Taylor 展开构造单步 DPM-Solver`++` {#subsec-dpm-solver--single}

沿用[第 9.4.2 节](04-dpm-solver.md#subsec-dpm-method-2)中的 DPM-Solver 构造，DPM-Solver`++` 使用对数信噪比时间中的局部 Taylor 展开，近似沿经验轨迹的数据预测。对于 $k\geq0$，将其沿经验 PF-ODE 轨迹的 $k$ 阶全导数记为

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(k)}
(\hat{\mathbf{x}}_\lambda,\lambda)
:=
\frac{\mathop{}\!\mathrm{d}^k}{\mathop{}\!\mathrm{d}\lambda^k}
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda).
$$

对于从 $t_{i-1}$ 到 $t_i$ 的一步，令 $h_i := \lambda_{t_i}-\lambda_{t_{i-1}} >0$。在 $\lambda_{t_{i-1}}$ 附近 Taylor 展开，得到

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
=
\sum_{k=0}^{n-1}
\frac{
(\lambda-\lambda_{t_{i-1}})^k
}{
k!
}
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(k)}
\bigl(
\hat{\mathbf{x}}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\bigr)
+
\mathcal O
\bigl(
(\lambda-\lambda_{t_{i-1}})^n
\bigr).
$$

将此展开代入[式 (9.5.1)](05-dpm-solver-plus-plus.md#eq-dpmpp-data-exact)，得到

<span id="eq-dpmpp-taylor-expansion"></span>

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{t_{i-1}\to t_i}
(\tilde{\mathbf{x}}_{t_{i-1}})
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\sum_{k=0}^{n-1}
h_i^{k+1}
\varphi_{k+1}(-h_i)
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(k)}
\bigl(
\hat{\mathbf{x}}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\bigr)
+
\mathcal O(h_i^{n+1}),
\end{aligned}
\tag{9.5.2}
$$

其中，$\varphi_m$ 定义于[式 (9.4.9)](04-dpm-solver.md#eq-analytic-expansion)。这里使用了

$$
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda
\frac{
(\lambda-\lambda_{t_{i-1}})^k
}{
k!
}
\mathop{}\!\mathrm{d}\lambda
=
e^{\lambda_{t_i}}
h_i^{k+1}
\varphi_{k+1}(-h_i),
$$

以及 $\sigma_{t_i}e^{\lambda_{t_i}}=\alpha_{t_i}$。

当 $n=1$ 时，去掉 $\mathcal O(h_i^2)$ 余项，得到

$$
\tilde{\mathbf{x}}_{t_i}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_{i-1},
\qquad
{\mathbf{D}}_{i-1}
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_{i-1}},t_{i-1}),
$$

这恰好是数据预测形式的确定性 DDIM 更新。

当 $n=2$ 时，通过额外步内阶段近似一阶全导数。对于中点选择，定义

$$
s_i^{\mathrm{mid}}
:=
t_\lambda
\left(
\lambda_{t_{i-1}}+\frac{h_i}{2}
\right).
$$

用一阶 DPM-Solver`++` 步预测中点状态

$$
{\mathbf{u}}_i
=
\frac{
\sigma_{s_i^{\mathrm{mid}}}
}{
\sigma_{t_{i-1}}
}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{s_i^{\mathrm{mid}}}
\left(
1-e^{-h_i/2}
\right)
{\mathbf{D}}_{i-1},
$$

并在该处计算

$$
{\mathbf{D}}_i^{\mathrm{mid}}
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
({\mathbf{u}}_i,s_i^{\mathrm{mid}}).
$$

于是，中点 DPM-Solver`++`(2S) 更新具有特别简单的形式

$$
\tilde{\mathbf{x}}_{t_i}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_i^{\mathrm{mid}}.
$$

与中点 DPM-Solver-2 一样，这一简化的两阶段更新与直接二阶 Taylor／有限差分构造只相差 $\mathcal O(h_i^3)$，因此保留二阶精度。

这就得到单步 DPM-Solver`++` 构造。当 $n=1$ 时，它精确化为数据预测形式的确定性 DDIM 更新。当 $n=2$ 时，通过额外步内模型计算近似一阶 $\lambda$ 全导数，得到 *DPM-Solver`++`(2S)*，即[算法 5](04-dpm-solver.md#alg-dpm-solver-2)中 DPM-Solver-2 的数据预测对应版本。其用中点实现的形式总结于[算法 6](05-dpm-solver-plus-plus.md#alg-dpmpp-2s-mid)。

<section class="book-algorithm" id="alg-dpmpp-2s-mid" aria-labelledby="alg-dpmpp-2s-mid-title" markdown="1">

<div class="book-algorithm__title" id="alg-dpmpp-2s-mid-title" markdown="span">
算法 6：DPM-Solver++(2S)：中点特例。
</div>

<div class="book-algorithm__meta" markdown="1">

**输入：** 初始值 ${\mathbf{x}}_T$, 时间步 $\{t_i\}_{i=0}^M$, 数据预测模型 ${\mathbf{x}}_{{\boldsymbol{\phi}}^\times}$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="算法 6 的步骤" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_0}\gets{\mathbf{x}}_T$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

$\lambda_{t_i}\gets\log(\alpha_{t_i}/\sigma_{t_i})$ for $i=0,\ldots,M$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_0 \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times}(\tilde{\mathbf{x}}_{t_0},t_0)$  <span class="book-algorithm__comment">▷ *缓存起点预测*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

**for** $i\gets1$ **to** $M$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

$h_i \gets \lambda_{t_i}-\lambda_{t_{i-1}}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

$s_i^{\mathrm{mid}} \gets t_\lambda \left( \tfrac{\lambda_{t_{i-1}}+\lambda_{t_i}}{2} \right)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">7</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{u}}_i \gets \tfrac{\sigma_{s_i^{\mathrm{mid}}}}{\sigma_{t_{i-1}}} \tilde{\mathbf{x}}_{t_{i-1}} + \alpha_{s_i^{\mathrm{mid}}} \left( 1-e^{-h_i/2} \right) {\mathbf{D}}_{i-1}$  <span class="book-algorithm__comment">▷ *预测中点阶段*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">8</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_i^{\mathrm{mid}} \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} ({\mathbf{u}}_i,s_i^{\mathrm{mid}})$  <span class="book-algorithm__comment">▷ *在中点计算*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">9</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_i} \gets \tfrac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \tilde{\mathbf{x}}_{t_{i-1}} + \alpha_{t_i} \left( 1-e^{-h_i} \right) {\mathbf{D}}_i^{\mathrm{mid}}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">10</div>
<div class="book-algorithm__statement" markdown="1">

**if** $i&lt;M$ **then**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">11</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_i \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} (\tilde{\mathbf{x}}_{t_i},t_i)$  <span class="book-algorithm__comment">▷ *缓存供下一步使用*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">12</div>
<div class="book-algorithm__statement" markdown="1">

**end if**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">13</div>
<div class="book-algorithm__statement" markdown="1">

**end for**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">14</div>
<div class="book-algorithm__statement" markdown="1">

**return** $\tilde{\mathbf{x}}_{t_M}$

</div>
</div>

</div>
</div>

</section>

## 9.5.4 通过复用历史构造多步 DPM-Solver`++` {#subsec-dpm-solver--multi}

DPM-Solver`++`(2S) 通过在*当前步内*增加模型计算来获取二阶信息。DPM-Solver`++`(2M) 采取另一条途径：复用先前已完成步骤的模型预测，估计相同的局部变化。初始化后，每步只需一次新的模型计算，因此在相同 NFE 预算下可以进行更多、更小的数值步骤。这在强引导下尤其有用，因为大步长可能降低数值稳定性。

思想很简单。回顾[式 (9.5.2)](05-dpm-solver-plus-plus.md#eq-dpmpp-taylor-expansion)中的二阶展开，它同时需要当前数据预测及其关于 $\lambda$ 的一阶全导数。单步方法使用新的步内阶段估计导数，多步方法则根据近期模型预测的历史估计它。

第一步没有这样的历史，因此用一阶数据预测更新初始化

$$
\tilde{\mathbf{x}}_{t_1}
=
\frac{\sigma_{t_1}}{\sigma_{t_0}}
\tilde{\mathbf{x}}_{t_0}
+
\alpha_{t_1}
\left(
1-e^{-h_1}
\right)
{\mathbf{D}}_0,
\qquad
{\mathbf{D}}_0
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_0},t_0).
$$

这恰好是数据预测形式的确定性 DDIM 更新。

一旦获得两个已完成步骤的预测，对于 $i\geq2$，定义

$$
{\mathbf{D}}_{i-1}
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_{i-1}},t_{i-1}),
\qquad
{\mathbf{D}}_{i-2}
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_{i-2}},t_{i-2}).
$$

保留[式 (9.5.2)](05-dpm-solver-plus-plus.md#eq-dpmpp-taylor-expansion)的前两项，得到

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{t_{i-1}\to t_i}
(\tilde{\mathbf{x}}_{t_{i-1}})
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
&+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_{i-1}
\\
&+
\alpha_{t_i}
\left(
h_i-1+e^{-h_i}
\right)
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(1)}
\bigl(
\hat{\mathbf{x}}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\bigr)
+
\mathcal O(h_i^3).
\end{aligned}
$$

因此，二阶更新额外需要的唯一量，就是数据预测的一阶 $\lambda$ 全导数。

DPM-Solver`++`(2M) 不在新的中间阶段计算模型，而是根据前一个已完成步骤估计该导数：

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(1)}
\bigl(
\hat{\mathbf{x}}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\bigr)
\approx
\frac{
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
}{
h_{i-1}
}.
$$

在标准光滑性假设下，这一后向差分的误差为 $\mathcal O(h_{i-1})$。将其代入二阶展开，并合并两个预测项，得到

$$
\tilde{\mathbf{x}}_{t_i}^{\mathrm{poly}}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_i^{\mathrm{poly}},
$$

其中

$$
{\mathbf{D}}_i^{\mathrm{poly}}
:=
{\mathbf{D}}_{i-1}
+
\frac{\beta(h_i)}{h_{i-1}}
\left(
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
\right),
\qquad
\beta(h)
:=
\frac{
h-1+e^{-h}
}{
1-e^{-h}
}.
$$

当相邻步长比保持有界时，这一加权多步更新的单步局部误差为 $\mathcal O(h_i^3)$。

**实际 DPM-Solver`++`(2M) 更新。** 上述表达式已是有效的二阶多步构造。DPM-Solver`++`(2M) 使用略简化的系数，同时保留相同阶数。由于

$$
\beta(h)
=
\frac{h}{2}
+
\frac{h^2}{12}
+
\mathcal O(h^3),
$$

定义步长比

$$
r_i
:=
\frac{h_i}{h_{i-1}}
$$

并将 ${\mathbf{D}}_i^{\mathrm{poly}}$ 替换为

$$
{\mathbf{D}}_i^{\mathrm{sim}}
:=
{\mathbf{D}}_{i-1}
+
\frac{r_i}{2}
\left(
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
\right).
$$

等价地，

$$
{\mathbf{D}}_i^{\mathrm{sim}}
=
\left(
1+\frac{r_i}{2}
\right){\mathbf{D}}_{i-1}
-
\frac{r_i}{2}{\mathbf{D}}_{i-2}.
$$

解释很直观：最近两次模型预测提供数据预测变化的局部方向，${\mathbf{D}}_i^{\mathrm{sim}}$ 将这一趋势外推到当前步。

在相同的光滑性与步长比有界假设下，

$$
{\mathbf{D}}_i^{\mathrm{poly}}
-
{\mathbf{D}}_i^{\mathrm{sim}}
=
\mathcal O(h_i^2).
$$

由于

$$
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
=
\mathcal O(h_i),
$$

这一简化只使状态更新改变 $\mathcal O(h_i^3)$。因此，它保留二阶精度，得到实际 DPM-Solver`++`(2M) 更新

$$
\tilde{\mathbf{x}}_{t_i}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_i^{\mathrm{sim}}.
$$

计算 $\tilde{\mathbf{x}}_{t_i}$ 后，在新状态上计算一次模型，

$$
{\mathbf{D}}_i
=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_i},t_i),
$$

并缓存这一预测供下一步使用。因此，启动后每个 DPM-Solver`++`(2M) 步骤只需要一次新的模型计算。

**说明。**

对于均匀对数信噪比步长，$r_i=1$，所以

$$
{\mathbf{D}}_i^{\mathrm{sim}}
=
\frac{3}{2}{\mathbf{D}}_{i-1}
-
\frac{1}{2}{\mathbf{D}}_{i-2}.
$$

系数 $(\frac32,-\frac12)$ 是用于*模型预测*的熟悉的 AB2 外推权重。但完整 DPM-Solver`++`(2M) 更新并非普通 AB2：已知扩散相关因子 $\alpha_{t_i}(1-e^{-h_i})$ 被解析保留。

完整启动与两步过程总结于[算法 7](05-dpm-solver-plus-plus.md#alg-dpmpp-2m)。它与 DPM-Solver`++`(2S) 的关键区别是二阶信息的来源：2S 从当前步内额外阶段获取，而 2M 复用先前已完成步骤的模型计算。

<section class="book-algorithm" id="alg-dpmpp-2m" aria-labelledby="alg-dpmpp-2m-title" markdown="1">

<div class="book-algorithm__title" id="alg-dpmpp-2m-title" markdown="span">
算法 7：DPM-Solver++(2M)。
</div>

<div class="book-algorithm__meta" markdown="1">

**输入：** 初始值 ${\mathbf{x}}_T$, 时间步 $\{t_i\}_{i=0}^M$, 数据预测模型 ${\mathbf{x}}_{{\boldsymbol{\phi}}^\times}$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="算法 7 的步骤" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_0}\gets{\mathbf{x}}_T$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

$\lambda_{t_i}\gets\log(\alpha_{t_i}/\sigma_{t_i})$ for $i=0,\ldots,M$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_0 \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} (\tilde{\mathbf{x}}_{t_0},t_0)$  <span class="book-algorithm__comment">▷ *缓存起点预测*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

**情况 I. 启动（$i=1$）**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

$h_1 \gets \lambda_{t_1}-\lambda_{t_0}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_1} \gets \tfrac{\sigma_{t_1}}{\sigma_{t_0}} \tilde{\mathbf{x}}_{t_0} + \alpha_{t_1} \left( 1-e^{-h_1} \right) {\mathbf{D}}_0$  <span class="book-algorithm__comment">▷ *数据预测形式的 DDIM*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

**if** $M&gt;1$ **then**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">7</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_1 \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} (\tilde{\mathbf{x}}_{t_1},t_1)$  <span class="book-algorithm__comment">▷ *缓存供多步更新使用*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">8</div>
<div class="book-algorithm__statement" markdown="1">

**end if**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

**情况 II. 两步多步更新（$i\geq2$）**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">9</div>
<div class="book-algorithm__statement" markdown="1">

**for** $i\gets2$ **to** $M$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">10</div>
<div class="book-algorithm__statement" markdown="1">

$h_i \gets \lambda_{t_i}-\lambda_{t_{i-1}}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">11</div>
<div class="book-algorithm__statement" markdown="1">

$r_i \gets h_i/h_{i-1}$  <span class="book-algorithm__comment">▷ *步长比*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">12</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_i^{\mathrm{sim}} \gets \left( 1+\tfrac12 r_i \right) {\mathbf{D}}_{i-1} - \tfrac12 r_i {\mathbf{D}}_{i-2}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">13</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_i} \gets \tfrac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \tilde{\mathbf{x}}_{t_{i-1}} + \alpha_{t_i} \left( 1-e^{-h_i} \right) {\mathbf{D}}_i^{\mathrm{sim}}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">14</div>
<div class="book-algorithm__statement" markdown="1">

**if** $i&lt;M$ **then**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">15</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_i \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} (\tilde{\mathbf{x}}_{t_i},t_i)$  <span class="book-algorithm__comment">▷ *一次新模型调用；缓存供下一步使用*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">16</div>
<div class="book-algorithm__statement" markdown="1">

**end if**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">17</div>
<div class="book-algorithm__statement" markdown="1">

**end for**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">18</div>
<div class="book-algorithm__statement" markdown="1">

**return** $\tilde{\mathbf{x}}_{t_M}$

</div>
</div>

</div>
</div>

</section>

## 9.5.5 DPM-Solver`++`(2M) 与 DEIS 的关系 {#subsec-dpm-deis}

DPM-Solver`++`(2M) 与 DEIS 使用同一个基本多步思想：复用先前已完成步骤的模型预测，近似神经网络预测在下一区间如何变化，同时解析保留已知扩散相关权重。差别在于应用这一思想的表示。前面介绍的 DEIS 使用原始扩散时间下的噪声预测，而 DPM-Solver`++`(2M) 使用对数信噪比时间下的数据预测。

为直接揭示这一联系，可以将 DEIS 多项式外推原理应用于 DPM-Solver`++` 所基于的精确数据预测积分。回顾

$$
\widetilde{\boldsymbol{\Psi}}_{t_{i-1}\to t_i}
(\tilde{\mathbf{x}}_{t_{i-1}})
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\sigma_{t_i}
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\mathop{}\!\mathrm{d}\lambda.
$$

已知权重 $e^\lambda$ 已经解析处理。因此，在该步内唯一需要近似的量就是数据预测

$$
\lambda
\longmapsto
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda).
$$

使用最近缓存的两个预测 ${\mathbf{D}}_{i-2}$ 和 ${\mathbf{D}}_{i-1}$，最简单的非常数近似是通过这两个锚点的唯一仿射多项式。其 Lagrange 形式为

$$
P_1(\lambda)
=
\frac{
\lambda-\lambda_{t_{i-2}}
}{
h_{i-1}
}
{\mathbf{D}}_{i-1}
-
\frac{
\lambda-\lambda_{t_{i-1}}
}{
h_{i-1}
}
{\mathbf{D}}_{i-2}.
$$

在最近锚点附近，可以更清晰地将同一多项式写成

$$
P_1(\lambda)
=
{\mathbf{D}}_{i-1}
+
\frac{
\lambda-\lambda_{t_{i-1}}
}{
h_{i-1}
}
\left(
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
\right).
$$

第一个表达式是 Lagrange 形式，第二个则是*同一*仿射多项式的后向 Newton 形式。因此，这里的 Lagrange 插值与后向差分并非不同数值近似，只是同一个双锚点外推多项式的两种表示。

在 DPM-Solver`++` 积分内应用这一外推，得到

$$
\tilde{\mathbf{x}}_{t_i}^{\mathrm{poly}}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\sigma_{t_i}
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda
P_1(\lambda)
\mathop{}\!\mathrm{d}\lambda.
$$

由于权重 $e^\lambda$ 已知，所需的两个矩都可以精确计算：

$$
\sigma_{t_i}
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda\mathop{}\!\mathrm{d}\lambda
=
\alpha_{t_i}
\left(
1-e^{-h_i}
\right),
$$

以及

$$
\sigma_{t_i}
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda
\left(
\lambda-\lambda_{t_{i-1}}
\right)
\mathop{}\!\mathrm{d}\lambda
=
\alpha_{t_i}
\left(
h_i-1+e^{-h_i}
\right).
$$

代入这些系数，得到

$$
\begin{aligned}
\tilde{\mathbf{x}}_{t_i}^{\mathrm{poly}}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
&+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_{i-1}
\\
&+
\alpha_{t_i}
\left(
h_i-1+e^{-h_i}
\right)
\frac{
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
}{
h_{i-1}
}.
\end{aligned}
$$

等价地，

$$
\tilde{\mathbf{x}}_{t_i}^{\mathrm{poly}}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_i^{\mathrm{poly}},
$$

其中，${\mathbf{D}}_i^{\mathrm{poly}}$ 定义于[第 9.5.4 节](05-dpm-solver-plus-plus.md#subsec-dpm-solver--multi)。

这展示了与 DEIS 的精确联系。在对实际系数进行简化之前，DPM-Solver`++` 的二阶多步更新，恰好等于对数据预测／对数信噪比积分应用一次多项式 DEIS 外推原理所得到的更新。在这一意义上，它是 AB-DEIS-1 在数据预测／对数信噪比表示下的对应版本。

实际 DPM-Solver`++`(2M) 更新进一步使用主导阶展开来简化这一系数

$$
\beta(h_i)
=
\frac{h_i}{2}
+
\mathcal O(h_i^2),
$$

从而得到

$$
\frac{\beta(h_i)}{h_{i-1}}
=
\frac{h_i}{2h_{i-1}}
+
\mathcal O\left(
\frac{h_i^2}{h_{i-1}}
\right).
$$

相应地，DPM-Solver`++`(2M) 将有效预测中的 $\beta(h_i)/h_{i-1}$ 替换为 $h_i/(2h_{i-1})$。由于

$$
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
=
\mathcal O(h_{i-1}),
$$

这使有效预测改变

$$
\mathcal O\left(
\frac{h_i^2}{h_{i-1}}
\right)
\mathcal O(h_{i-1})
=
\mathcal O(h_i^2).
$$

状态更新中乘在有效预测前的系数为 $\alpha_{t_i}(1-e^{-h_i})=\mathcal O(h_i)$，因此状态更新的最终变化仅为 $\mathcal O(h_i^3)$。所以，DPM-Solver`++`(2M) 与一次加权多项式更新通常是不同的有限步公式，但一致到二阶。

这也说明了为什么与 DEIS 的联系特指 DPM-Solver`++`(2M)：2M 通过外推先前已完成步骤的预测获得额外二阶信息，而 DPM-Solver`++`(2S) 从当前步内的新中间阶段获取。

完整 DPM-Solver`++`(2M) 过程，包括一阶启动及每步后缓存一次新数据预测，总结于[算法 7](05-dpm-solver-plus-plus.md#alg-dpmpp-2m)。

## 9.5.6 结语 {#closing-remark}

虽然主要讨论了 DPM-Solver 与 DPM-Solver`++`，但简要了解更广泛的背景也有帮助。DPM-Solver 从噪声预测表述推导，而 DPM-Solver`++` 使用数据预测重新表述求解器，后者在强 CFG 下可能表现更好。DPM-Solver-v3（[Zheng et al., 2023](../../references.md#ref-zheng2023dpm)）进一步发展这一思想，引入更一般的模型表示，利用预先计算的模型统计量选择系数，以降低局部离散化误差。在这一意义上，它使表示本身成为数值求解器设计的一部分。

另一个相关方向是预测器—校正器框架。在 Score SDE 框架中（见[第 4 章](../04-score-sde-framework/index.md#ch-score-sde)），*预测器*通过逆时间动力学的数值步骤推进样本，*校正器*则使用 score 进一步改进样本，通常采用同一噪声水平下的 Langevin 型更新（见[第 3.4 节](../03-score-based-perspective/04-noise-conditional-score-networks.md#sec-smld)）。这种先预测、再改进的模式，也有经典数值对应。例如，Heun 方法先用 Euler 预测端点，再用平均斜率改进数值更新。UniPC 等方法（[Zhao et al., 2023](../../references.md#ref-zhao2023unipc)）专门为扩散采样构造预测器—校正器方法。这里不继续展开，但这些扩展体现了本章同一个更广泛的主题：有效的扩散求解器将经典数值思想与扩散 PF-ODE 特有结构结合起来。

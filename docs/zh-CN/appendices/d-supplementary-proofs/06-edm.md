# D.6（选读）Elucidating Diffusion Model（EDM） {#sec-edm}

本节介绍 ${\mathbf{x}}$-预测模型中神经网络参数化设计的具体准则，源自 Elucidating Diffusion Models（EDM）（[Karras et al., 2022](../../references.md#ref-karras2022elucidating)）。EDM 提供了一套简单的方法，使训练过程更易于优化，也更可靠。${\mathbf{x}}$-预测模型表示为随时间变化的跳跃连接与缩放残差的组合（[式（D.6.1）](06-edm.md#eq-d-parametrization)）。其主要思想是，在所有时刻都将输入和回归目标归一化至单位方差，并调整跳跃分支，使残差误差不会随时间演化而放大。这套方法已成为扩散模型实现中广泛采用的默认方案，也自然延伸到流映射学习，尤其是 Consistency Model 系列方法。

## D.6.1 神经网络 ${\mathbf{x}}_{\boldsymbol{\phi}}$ 的设计准则 {#subsec-d-design-edm}

EDM 考虑 ${\mathbf{x}}$-预测模型的一种参数化，略微复用记号，将其记为 ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t)$[^1]，形式如下：

<span id="eq-d-parametrization"></span>

$$
\begin{aligned}
{\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t) := c_{\mathrm{skip}}(t) {\mathbf{x}} + c_{\mathrm{out}}(t) {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t) {\mathbf{x}}, c_{\mathrm{noise}}(t)\right).
\end{aligned}
\tag{D.6.1}
$$

这里，$c_{\mathrm{skip}}(t)$、$c_{\mathrm{out}}(t)$、$c_{\mathrm{in}}(t)$ 和 $c_{\mathrm{noise}}(t)$ 都是随时间变化的函数。它们基于下文将介绍的实际考虑来选取，目的是提高训练的稳定性和性能。

将其代入[式（6.2.5）](../../chapters/06-unified-perspective/02-training-objectives.md#eq-general-dm-loss-d)，经过直接的代数变换，目标函数变为[^2]：

<span id="eq-loss-f"></span>

$$
\begin{aligned}
\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)c_{\mathrm{out}}^2(t)\left\lVert {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t){\mathbf{x}}_t, c_{\mathrm{noise}}(t)\right) - {\mathbf{x}}_{\mathrm{tgt}}(t)\right\rVert_2^2\right].
\end{aligned}
\tag{D.6.2}
$$

其中，回归目标 ${\mathbf{x}}_{\mathrm{tgt}}(t)$ 为：

$$
{\mathbf{x}}_{\mathrm{tgt}}(t) = \frac{\left(1-c_{\mathrm{skip}}(t)\alpha_t\right){\mathbf{x}}_0-c_{\mathrm{skip}}(t)\sigma_t \boldsymbol{\epsilon}}{c_{\mathrm{out}}(t)}.
$$

引入 $p_{\text{data}}$ 的标准差，记为 $\sigma_{\mathrm{d}}$，EDM 提出了以下网络参数化设计准则，可以称为*单位方差准则*。

**输入具有单位方差。**

$$
\begin{aligned}
&\text{Var}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[ c_{\mathrm{in}}(t){\mathbf{x}}_t\right] = 1
    \\ \Longleftrightarrow &~c_{\mathrm{in}}^2(t) \text{Var}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[ \alpha_t {\mathbf{x}}_0 +\sigma_t \boldsymbol{\epsilon} \right] = 1
    \\ \Longleftrightarrow &~c_{\mathrm{in}}(t)  = \frac{1}{\sqrt{\sigma_{\mathrm{d}}^2 \alpha_t^2 + \sigma_t^2}},
\end{aligned}
$$

取其中的正根。

**训练目标具有单位方差。**

<span id="eq-c-out"></span>

$$
\begin{aligned}
&\text{Var}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[ {\mathbf{x}}_{\mathrm{tgt}}(t) \right] = 1
    \\ \Longleftrightarrow &~c_{\mathrm{out}}^2(t)  = \left(1-c_{\mathrm{skip}}(t)\alpha_t\right)^2 \sigma_{\mathrm{d}}^2 + c_{\mathrm{skip}}^2(t)\sigma_t^2,
\end{aligned}
\tag{D.6.3}
$$

其中数据经过中心化（${\mathbb{E}}[{\mathbf{x}}_0]=\boldsymbol{0}$）。

**最小化从 ${\mathbf{F}}_{\boldsymbol{\phi}}$ 到 ${\mathbf{x}}_{\boldsymbol{\phi}}$ 的误差放大。** EDM 旨在减轻网络学习误差从 ${\mathbf{F}}_{\boldsymbol{\phi}}$ 到 ${\mathbf{x}}_{\boldsymbol{\phi}}$ 的放大效应。为此，选择 $c_{\mathrm{skip}}$ 使 $c_{\mathrm{out}}$ 最小：

$$
c_{\mathrm{skip}}^* \in \operatorname*{arg\,min}_{c_{\mathrm{skip}}}  c_{\mathrm{out}}^2.
$$

采用标准方法，通过求解 $\frac{\partial c_{\mathrm{out}}}{\partial c_{\mathrm{skip}}} = 0$ 得到临界点 $c_{\mathrm{skip}}^*$，可得

$$
\begin{aligned}
c_{\mathrm{skip}}^*(t) = \frac{\alpha_t \sigma_{\mathrm{d}}^2 }{\alpha_t^2 \sigma_{\mathrm{d}}^2  + \sigma_t^2}.
\end{aligned}
$$

将其代入[式（D.6.3）](06-edm.md#eq-c-out)，最优值为

$$
\begin{aligned}
c_{\mathrm{out}}^*(t) = \pm \frac{\sigma_t \sigma_{\mathrm{d}} }{\sqrt{\alpha_t^2 \sigma_{\mathrm{d}}^2  + \sigma_t^2}}.
\end{aligned}
$$

按照惯例，输出尺度取非负分支：

$$
c_{\mathrm{out}}^*(t) = \frac{\sigma_t \sigma_{\text{d}}}
{\sqrt{\alpha_t^2 \sigma_{\text{d}}^2 + \sigma_t^2}}
\;\;\;(\ge 0),
$$

这保证了 $c_{\mathrm{out}}(0)=0$，并且 $c_{\mathrm{out}}(t)\to\sigma_{\text{d}}$ 在 $\sigma_t$ 足够大时成立，从而得到直观的极限 ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,0)\approx {\mathbf{x}}_t$ 和 ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)\approx \sigma_{\text{d}}\,{\mathbf{F}}_{\boldsymbol{\phi}}(\cdot)$。

将这些系数归纳如下：

记 $R_t := \alpha_t^2 \sigma_{\mathrm{d}}^2+ \sigma_t^2$，则选择：

<span id="eq-edm-coeff"></span>

$$
\begin{aligned}
            c_{\mathrm{in}}(t)  = \frac{1}{\sqrt{R_t}}, \quad
            c_{\mathrm{skip}}(t) = \frac{\alpha_t \sigma_{\mathrm{d}}^2 }{R_t}, \quad
            c_{\mathrm{out}}(t) =  \frac{\sigma_t \sigma_{\mathrm{d}} }{\sqrt{R_t}}.
        \end{aligned}
\tag{D.6.4}
$$

利用[式（D.6.4）](06-edm.md#eq-edm-coeff)，回归目标 ${\mathbf{x}}_{\mathrm{tgt}}(t)$ 简化为

$$
{\mathbf{x}}_{\mathrm{tgt}}(t) = \frac{1}{\sigma_{\mathrm{d}}}\frac{\sigma_t {\mathbf{x}}_0 - \alpha_t \sigma_{\mathrm{d}}^2\boldsymbol{\epsilon} }{\sqrt{R_t}}.
$$

此外，将这些表达式代入[式（D.6.1）](06-edm.md#eq-d-parametrization)和[式（D.6.2）](06-edm.md#eq-loss-f)，可以同时简化参数化形式与损失函数，得到：

$$
\begin{aligned}
{\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t)=\frac{\alpha_t \sigma_{\mathrm{d}}^2 }{R_t} {\mathbf{x}} + \frac{\sigma_t \sigma_{\mathrm{d}} }{\sqrt{R_t}}{\mathbf{F}}_{\boldsymbol{\phi}}\left(\frac{1}{\sqrt{R_t}}{\mathbf{x}}, c_{\mathrm{noise}}(t)\right),
\end{aligned}
$$

以及

$$
\begin{aligned}
\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)\frac{\sigma_t^2  }{R_t}\left\lVert \sigma_{\mathrm{d}} {\mathbf{F}}_{\boldsymbol{\phi}}\left(\frac{1}{\sqrt{R_t}}{\mathbf{x}}_t, c_{\mathrm{noise}}(t)\right) - \left(\frac{\sigma_t {\mathbf{x}}_0 - \alpha_t \sigma_{\mathrm{d}}^2\boldsymbol{\epsilon} }{\sqrt{R_t}}\right)\right\rVert_2^2\right].
\end{aligned}
$$

采用这一参数化，并在条件 $\alpha_0 \approx 1$ 和 $\sigma_0 \approx 0$ 下，可以看到

$$
c_{\mathrm{skip}}(0) \approx 1 \quad \text{and} \quad c_{\mathrm{out}}(0) \approx 0.
$$

**$c_{\mathrm{noise}}(t)$ 的选择。** 它为 ${\mathbf{F}}_{\boldsymbol{\phi}}$ 提供噪声水平嵌入；噪声水平 $\sigma_t$ 的任意一一映射（例如 $c_{\mathrm{noise}}(t)=\log \sigma_t$）均可。

## D.6.2 EDM 的一个常见特例：$\alpha_t = 1$、$\sigma_t = t$ {#common-edm-special-case}

考虑 EDM 使用的简化噪声调度，其中 $\alpha_t = 1$ 且 $\sigma_t = t$，这一设定也出现在[第 11.4 节](../../chapters/11-fast-generators/04-consistency-trajectory-models.md#sec-ctm)对 CTM 的讨论中。此时，前向过程变为

$$
{\mathbf{x}}_t = {\mathbf{x}}_0 + t\boldsymbol{\epsilon},
\quad
\text{with }
{\mathbf{x}}_0 \sim p_{\text{data}},
\,\,
\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}),
$$

对应的扰动核为

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
=
\mathcal{N}({\mathbf{x}}_t; {\mathbf{x}}_0, t^2{\mathbf{I}}).
$$

由此得到的边缘密度为

$$
p_t({\mathbf{x}})
=
\int
\mathcal{N}({\mathbf{x}};{\mathbf{x}}_0,t^2{\mathbf{I}})
p_{\mathrm{data}}({\mathbf{x}}_0)
\mathop{}\!\mathrm{d}{\mathbf{x}}_0
=
\bigl[
p_{\mathrm{data}}
*
\mathcal{N}(\boldsymbol{0},t^2{\mathbf{I}})
\bigr]({\mathbf{x}}).
$$

因此，精确的终端边缘分布为 $p_T=p_{\mathrm{data}}*\mathcal{N}(\boldsymbol{0},T^2{\mathbf{I}})$。当 $T$ 相对于数据尺度足够大时，通常用 $\mathcal{N}(\boldsymbol{0},T^2{\mathbf{I}})$ 近似这一边缘分布，并将其作为采样先验：

$$
p_{\mathrm{prior}}({\mathbf{x}}_T)
:=
\mathcal{N}({\mathbf{x}}_T;\boldsymbol{0},T^2{\mathbf{I}}).
$$

在这一设定下，基于 ${\mathbf{x}}$-预测 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}$ 的 PF-ODE（见[式（6.3.2）](../../chapters/06-unified-perspective/03-equivalent-parameterizations.md#eq-clean-pf-ode)）简化为

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = \frac{{\mathbf{x}}(t) - {\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t), t)}{t}.
\end{aligned}
$$

将这一表述代入[式（D.6.4）](06-edm.md#eq-edm-coeff)，神经网络的参数化系数变为

<span id="eq-edm-simple-coefficients"></span>

$$
\begin{aligned}
c_{\mathrm{in}}(t) = \frac{1}{\sqrt{\sigma_{\mathrm{d}}^2 + t^2}}, \quad
    c_{\mathrm{skip}}(t) = \frac{\sigma_{\mathrm{d}}^2}{\sigma_{\mathrm{d}}^2 + t^2}, \quad
    c_{\mathrm{out}}(t) = \pm \frac{t \sigma_{\mathrm{d}}}{\sqrt{\sigma_{\mathrm{d}}^2 + t^2}}.
\end{aligned}
\tag{D.6.5}
$$

由这些表达式可以看出：

-   当 $t \approx 0$ 时，噪声水平可以忽略，因此 $c_{\mathrm{skip}} \approx 1$ 且 $c_{\mathrm{out}} \approx 0$。在这一极限下，跳跃分支占据主导，网络基本上直接传递输入，

    $$
    {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t) \approx {\mathbf{x}}.
    $$

-   当 $t \gg 0$ 时，输入受到强噪声扰动，因此 $c_{\mathrm{skip}} \approx 0$ 且 $c_{\mathrm{out}} \approx \sigma_{\mathrm{d}}$。在此情形下，跳跃分支消失，模型输出完全由学得的残差决定，

    $$
    {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t) \approx \sigma_{\mathrm{d}} {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t){\mathbf{x}}, c_{\mathrm{noise}}(t)\right),
    $$

    这意味着网络 ${\mathbf{F}}_{\boldsymbol{\phi}}$ 根据归一化的带噪输入预测干净信号的缩放替代量；因此，在高噪声水平下，模型输出完全由学得的去噪函数决定。

这种参数化在两种形式之间平滑插值：$t$ 较小时的恒等映射，以及 $t$ 较大时作用于标准化输入的缩放残差预测器。

[^1]: 如前所述，四种预测类型都是等价的，均可归结为 ${\mathbf{x}}$-预测。EDM 采用这一表述，因为它已有充分研究，也自然契合流映射模型生成干净样本的目标。

[^2]: 从 ${\mathbf{x}}$-预测扩散损失出发，代入[式（D.6.1）](06-edm.md#eq-d-parametrization)给出的参数化：

    $$
    \begin{aligned}
          &~\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)\left\lVert {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - {\mathbf{x}}_0\right\rVert_2^2\right]
          \\= &~\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)\Big\Vert c_{\mathrm{skip}}(t) \underbrace{(\alpha_t {\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}})}_{{\mathbf{x}}_t} + c_{\mathrm{out}}(t) {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t) {\mathbf{x}}_t, c_{\mathrm{noise}}(t)\right) - {\mathbf{x}}_0\Big\Vert_2^2\right]
            \\= &~\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)c_{\mathrm{out}}^2(t)\Big\Vert  {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t) {\mathbf{x}}, c_{\mathrm{noise}}(t)\right) - \underbrace{\left(\frac{\left(1-c_{\mathrm{skip}}(t)\alpha_t\right){\mathbf{x}}_0-c_{\mathrm{skip}}(t)\sigma_t \boldsymbol{\epsilon}}{c_{\mathrm{out}}(t)}\right)}_{{\mathbf{x}}_{\mathrm{tgt}}(t)}\Big\Vert_2^2\right]
            \\ = &~\href{../06-edm/#eq-loss-f}{\text{Equation (D.6.2)}}.
        \end{aligned}
    $$

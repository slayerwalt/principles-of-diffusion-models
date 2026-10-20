# 扩散模型训练与采样的初步操作视角 {#operational-view}

在讨论扩散模型的数学原理之前，我们先从操作过程出发，初步了解它们如何训练以及如何生成样本。

基本构造分为两部分。首先，定义一个已知的扰动规则，将干净数据转化为任意指定噪声水平下的带噪观测。其次，训练一个神经网络，从这样的带噪观测中推断干净信号。当网络在多个噪声水平下学会这一预测规则后，就可以将其用于生成：从一个新的噪声样本出发，在噪声水平由高到低的过程中反复应用网络。由此得到一个逐渐呈现出数据分布结构的新样本。

**要点**

在多个噪声水平下构造带噪数据，学习预测干净信号，再从高噪声到低噪声反复应用这一预测，实现生成。

**构造带噪数据。**

先准确说明数据的带噪版本究竟是什么。取一个干净数据样本

$$
{\mathbf{x}}_0\sim p_{\mathrm{data}},
$$

它可以是图像，也可以是我们关心的其他类型的数据。在噪声水平 $t$ 下，混入独立的 Gaussian 噪声来扰动它<sup id="fnref:4"><a class="footnote-ref" href="#fn:4" role="doc-noteref">4</a></sup>：

$$
{\mathbf{x}}_t
=
\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}},
\qquad
{\boldsymbol{\epsilon}}\sim\mathcal{N}(\boldsymbol{0},{\mathbf{I}}).
$$

可以将 $\alpha_t$ 和 $\sigma_t$ 看作两个预先设定的旋钮，控制噪声水平 $t$ 下混合信号中干净信号与噪声各自的含量。

最简单的选择是 $\alpha_t=1$ 和 $\sigma_t=t$，此时

$$
{\mathbf{x}}_t={\mathbf{x}}_0+t{\boldsymbol{\epsilon}}.
$$

此时 $t$ 就直接充当噪声调节旋钮：将它调向零，${\mathbf{x}}_t$ 就接近干净样本；将它调大，噪声项 $t{\boldsymbol{\epsilon}}$ 就逐渐占据主导。

<span id="fig-ill-forward"></span>

[![通过添加噪声来扰动干净样本。](../../en/assets/foundations/forward-diffusion.png)](../../en/assets/foundations/forward-diffusion.png)

**图 1.3：** **前向扩散。** 对干净样本 $\mathbf{x}_0$ 进行扰动，得到带噪观测 $\mathbf{x}_t$，其噪声水平为 $t$。

来源：作者制作。

更一般地，每个噪声水平 $t$ 都定义了带噪样本的一个分布 $p_t$。可以将 $p_t$ 看作一幅*快照*：如果使用相同的 $t$ 值生成大量带噪样本，它们的分布就是 $p_t$。

调度 $(\alpha_t,\sigma_t)$ 的选择应使这些快照从干净数据逐渐过渡到简单的噪声分布：

$$
p_0=p_{\mathrm{data}}
\qquad\longrightarrow\qquad
p_t
\qquad\longrightarrow\qquad
p_T\approx p_{\mathrm{prior}},
$$

其中 $p_{\mathrm{prior}}$ 通常是易于采样的 Gaussian 分布。因此，分布族

$$
\{p_t\}_{t\in[0,T]}
$$

描述了分布如何随噪声水平的增大而逐渐变化。

**训练：学习预测干净信号。**

有了扰动数据的方法，训练流程便几乎可以按固定步骤执行。它依赖一个关键技巧：由于数据是由我们自己扰动的，我们总是知道正确的答案。构造一个训练样本时，先抽取干净数据 ${\mathbf{x}}_0$，选择噪声水平 $t$，抽取噪声 ${\boldsymbol{\epsilon}}$，然后构造

$$
{\mathbf{x}}_t
=
\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}.
$$

${\mathbf{x}}_t$ 是我们自己构造的，因此我们确切知道它由哪个干净样本 ${\mathbf{x}}_0$ 产生。

不过，网络不会直接看到 ${\mathbf{x}}_0$ 或 ${\boldsymbol{\epsilon}}$。它只能看到二元组 $({\mathbf{x}}_t,t)$，即受到扰动的观测及其扰动程度。网络的任务是猜测其背后的干净数据：

$$
{\mathbf{x}}_0^{\mathrm{pred}}
:=
{\mathbf{x}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t).
$$

由于真实目标已知，我们可以用普通的平方误差损失来评价并改进这一猜测：

$$
\left\|
{\mathbf{x}}_0^{\mathrm{pred}}-{\mathbf{x}}_0
\right\|_2^2.
$$

**要点**

也就是说，构造一个干净目标已知的带噪观测，对网络隐藏这个目标，再训练网络将它预测出来。

针对不同数据样本、噪声抽样和噪声水平反复执行这一过程，一个网络就能学会在整个扰动范围内预测干净数据，涵盖从几乎未受扰动到接近纯噪声的各种情况。

这里需要仔细理解，这个损失究竟让网络学会预测什么。在平方误差下，最优预测为

$$
{\mathbf{x}}_0^*({\mathbf{x}}_t,t)
=
{\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_t,t].
$$

带噪观测通常并不对应唯一的干净样本。多个不同的干净样本可能产生相似的 ${\mathbf{x}}_t$。这种不确定性存在于任何非零噪声水平下，并在噪声水平较高时显著增强。因此，网络学到的是：对给定当前带噪观测时可能的干净样本取平均<sup id="fnref:5"><a class="footnote-ref" href="#fn:5" role="doc-noteref">5</a></sup>。

**采样：从新的噪声出发。**

训练得到的网络擅长根据带噪数据猜测干净数据。接下来的问题是，如何利用这种能力生成新的内容。

训练完成后，我们冻结网络参数，并将其记为 ${\boldsymbol{\phi}}^\times$。生成沿着与前述过程相反的方向进行：从噪声本身出发，逆转从数据开始添加噪声的方向，

$$
\tilde{{\mathbf{x}}}_T\sim p_{\mathrm{prior}}.
$$

我们用波浪号标记生成状态 $\tilde{{\mathbf{x}}}_t$，使其在记号上区别于由前向扰动产生的训练变量 ${\mathbf{x}}_t$。

这一区别的意义可能比乍看之下更大。状态 $\tilde{{\mathbf{x}}}_T$ 并非由某个隐藏的真实样本经过扰动而产生，其背后没有等待恢复的真实 ${\mathbf{x}}_0$。因此，生成并不是“倒放”某次特定扰动的“录像”，因为这样的录像根本不存在。在每个噪声水平下，我们只是询问训练好的网络：它当前认为干净信号是什么样子，然后利用这一判断朝数据方向前进一步。

具体来说，假设当前状态为 $\tilde{{\mathbf{x}}}_t$。将其输入网络，得到对干净数据的猜测，

$$
{\mathbf{x}}_0^{\mathrm{pred}}
:=
{\mathbf{x}}_{{\boldsymbol{\phi}}^\times}(\tilde{{\mathbf{x}}}_t,t).
$$

当 $t>0$ 时，还可以计算一个隐含的噪声残差，其定义为从当前状态中去除预测信号后剩下的部分：

$$
{\boldsymbol{\epsilon}}_t^{\mathrm{pred}}
:=
\frac{
\tilde{{\mathbf{x}}}_t-\alpha_t{\mathbf{x}}_0^{\mathrm{pred}}
}{
\sigma_t
}.
$$

这两个量并非相互独立的猜测：按照它们的构造方式，二者恰好可以重新组合成当前状态，

$$
\tilde{{\mathbf{x}}}_t
=
\alpha_t{\mathbf{x}}_0^{\mathrm{pred}}
+
\sigma_t{\boldsymbol{\epsilon}}_t^{\mathrm{pred}}.
$$

单次采样步骤的思想是：保持预测信号和预测噪声不变，使用略低的噪声水平 $t-\Delta t$ 所对应的系数，替换 $t$ 对应的系数，重新混合二者：

$$
\tilde{{\mathbf{x}}}_{t-\Delta t}
=
\alpha_{t-\Delta t}{\mathbf{x}}_0^{\mathrm{pred}}
+
\sigma_{t-\Delta t}{\boldsymbol{\epsilon}}_t^{\mathrm{pred}}.
$$

等价地，这是在当前状态上进行的一次小幅更新，

<span id="eq-ddpm-first-try"></span>

$$
\tilde{{\mathbf{x}}}_{t-\Delta t}
=
\tilde{{\mathbf{x}}}_t
+
(\alpha_{t-\Delta t}-\alpha_t){\mathbf{x}}_0^{\mathrm{pred}}
+
(\sigma_{t-\Delta t}-\sigma_t){\boldsymbol{\epsilon}}_t^{\mathrm{pred}}.
\tag{1.4.1}
$$

这样便完成了一步。接着，将新状态 $\tilde{{\mathbf{x}}}_{t-\Delta t}$ 再次输入网络，得到更新后且通常不同的干净数据预测，再向更低的噪声水平前进一步。采样正是将这个循环从 $t=T$ 一直重复到 $t=0$。

**要点**

也就是说，从新的噪声出发，预测干净信号，前进到更低的噪声水平，然后再次预测。

<span id="fig-ill-reverse"></span>

[![随着噪声水平下降，新样本逐渐形成。](../../en/assets/foundations/diffusion-generation.png)](../../en/assets/foundations/diffusion-generation.png)

**图 1.4：** **扩散生成。** 从新的噪声 $\tilde{\mathbf{x}}_T\sim p_{\mathrm{prior}}$ 出发，随着噪声水平下降，反复进行模型预测来构造新样本。

来源：作者制作。

此时，一个自然的问题是：为什么不直接输出第一次预测 ${\mathbf{x}}_0^{\mathrm{pred}}$，然后停止？因为在高噪声水平下，当前状态包含的干净结构信息很少，预测结果基本上只是许多可能样本的粗略平均。移动到略低的噪声水平后，网络面对的状态不确定性稍小，所以下一次预测能够变得更清晰。多次重复这个过程，才能让细节丰富的样本逐渐形成<sup id="fnref:6"><a class="footnote-ref" href="#fn:6" role="doc-noteref">6</a></sup>。

**一个简单的例子。**

沿用前面的例子，取 $\alpha_t=1$ 和 $\sigma_t=t$，让这一过程更具体。代入上述更新式，得到

$$
\tilde{{\mathbf{x}}}_{t-\Delta t}
=
{\mathbf{x}}_0^{\mathrm{pred}}
+
(t-\Delta t)
\frac{
\tilde{{\mathbf{x}}}_t-{\mathbf{x}}_0^{\mathrm{pred}}
}{
t
},
$$

整理后可写成简单的加权平均，

$$
\tilde{{\mathbf{x}}}_{t-\Delta t}
=
\frac{t-\Delta t}{t}\tilde{{\mathbf{x}}}_t
+
\frac{\Delta t}{t}{\mathbf{x}}_0^{\mathrm{pred}}.
$$

用文字来说，当 $\Delta t$ 很小时，下一个状态就是当前所处位置与网络认为最终应到达位置之间的混合。

因此，每一步都将样本稍微推向网络当前预测的干净数据。这里的*当前*十分关键：移动之后，网络看到的是一个新的、噪声更少的状态，因此通常会改变自己的预测。最终样本并非单次干净数据猜测的输出，而是由一整串猜测逐步构造出来的，每次猜测都依据上一步的结果作出修正。

**这与后续章节有什么联系？**

这里建立的干净数据预测框架，只是从操作过程出发的初步认识。全书中的基本模式始终相同：训练时，从干净数据构造带噪观测，并学习从中预测有用的信息；生成时，从新的噪声出发，利用这些学得的预测，在不断下降的噪声水平下构造新样本。

后续章节将从不同数学视角解释同一过程。[第 2 章](../chapters/02-variational-perspective/index.md#ch-variational)将 DDPM 与潜变量模型联系起来，其中固定的加噪过程充当编码器，学得的逆向转移充当解码器。[第 3 章](../chapters/03-score-based-perspective/index.md#ch-score-based)通过 score 描述去噪，它为更新带噪样本提供方向。[第 4 章](../chapters/04-score-sde-framework/index.md#ch-score-sde)从离散噪声水平转向连续时间，通过逆时间随机动力学描述生成。[第 5 章](../chapters/05-flow-based-perspective/index.md#ch-flow-based)则将生成视为沿连续路径传输样本的过程，从简单先验逐渐到达数据分布。最后，[第 6 章](../chapters/06-unified-perspective/index.md#ch-unified-perspective)解释这些视角之间的联系。

这里要把握的主要思想是：扩散模型从多个噪声水平下的数据带噪版本中学习，生成时反复使用学得的预测，将新的噪声转变为新样本。

<div class="footnote" role="doc-endnotes" markdown="1">

<hr />

<ol start="4" markdown="1">
<li id="fn:4" markdown="block">

公式 ${\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}$ 直接给出了指定噪声水平 $t$ 下的累积扰动。在 DDPM 和 Score SDE 中，也可以通过多个较小的顺序步骤添加噪声，得到该水平下相同的边缘分布；见[第 2.2 节](../chapters/02-variational-perspective/02-ddpm.md#sec-ddpm)及[第 4 章](../chapters/04-score-sde-framework/index.md#ch-score-sde)。在后面介绍的离散 DDPM 记号下，这种累积扰动写为

$$
{\mathbf{x}}_i
=
\bar{\alpha}_i{\mathbf{x}}_0
+
\sqrt{1-\bar{\alpha}_i^2}\,{\boldsymbol{\epsilon}}.
$$

<a class="footnote-backref" href="#fnref:4" aria-label="返回正文中的脚注 4">↩</a>

</li>

<li id="fn:5" markdown="block">

当噪声水平可由上下文确定时，通常将其简写为 ${\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_t]$。

<a class="footnote-backref" href="#fnref:5" aria-label="返回正文中的脚注 5">↩</a>

</li>

<li id="fn:6" markdown="block">

后面将把这一确定性更新与 DDIM 采样及其 Euler 型解释（[Song et al., 2020](../references.md#ref-song2020denoising)）联系起来，见[第 9.2 节](../chapters/09-numerical-solvers/02-ddim.md#sec-revisit-ddim)；随机 DDPM 采样器将在[第 2.2 节](../chapters/02-variational-perspective/02-ddpm.md#sec-ddpm)中介绍。

<a class="footnote-backref" href="#fnref:6" aria-label="返回正文中的脚注 6">↩</a>

</li>

</ol>

</div>

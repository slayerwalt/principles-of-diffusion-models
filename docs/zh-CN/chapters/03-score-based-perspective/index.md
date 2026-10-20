# 3. 基于 score 的视角：从 EBM 到 NCSN {#ch-score-based}

前几章中，我们追溯了扩散模型的变分起源，并说明了它们如何从 VAE 框架中产生。现在转向另一个同样基础的视角：*基于能量的模型*（Energy-Based Models，EBM）（[Ackley et al., 1985](../../references.md#ref-ackley1985learning)；[LeCun et al., 2006](../../references.md#ref-lecun2006tutorial)）。EBM 用一个在数据处较低、在其他位置较高的能量景观表示分布。采样通常依靠 Langevin 动力学，沿着该景观的梯度将样本移向高密度区域。这个称为 *score* 的梯度场指向概率更高的方向。

关键观察是，只要知道 score 就足以进行生成：它将样本移向更可能出现数据的区域，无须计算难以处理的归一化常数。*基于 score* 的扩散模型直接建立在这一思想之上。它们不仅关注干净数据分布，还考虑一系列经过 Gaussian 噪声扰动的分布，这些分布的 score 更容易近似。学习这些 score 可得到一族向量场，引导含噪样本逐步回到数据，从而将生成转化为逐步去噪。

**读完本章后，你将能够……**

-   理解基于 score 建模的核心思想：什么是 score 函数，以及如何将其与 Langevin 动力学结合用于生成；

-   理解去噪 score 匹配如何通过条件化技巧学习单个噪声水平下的 score，以及 Tweedie 公式如何将 score 预测与去噪联系起来；

-   理解 Noise Conditional Score Networks（NCSN）如何将去噪 score 匹配扩展到多个噪声水平、多个噪声水平为何有用，以及 NCSN 的 score 预测与 DDPM 的噪声预测有何关系。

**阅读路线。** 已熟悉基于能量的模型的读者可以略读[第 3.1 节](01-energy-based-models.md#sec-ebm)，从[第 3.2.1 节](02-score-based-models.md#subsec-training-sm)开始阅读。

继续阅读以下各节及[参考文献](../../references.md#references)。

<span id="sec-ebm"></span>

[3.1 基于能量的模型](01-energy-based-models.md#sec-ebm)

<span id="eq-ebm-sm"></span>

[式 (3.1.3)](01-energy-based-models.md#eq-ebm-sm)

<span id="sec-ebm-to-score"></span>

[3.2 从基于能量到基于 score 的生成模型](02-score-based-models.md#sec-ebm-to-score)

<span id="eq-sm"></span>

[式 (3.2.1)](02-score-based-models.md#eq-sm)

<span id="sec-dsm"></span>

[3.3 去噪 score 匹配](03-denoising-score-matching.md#sec-dsm)

<span id="sec-smld"></span>

[3.4 多噪声水平的去噪 score 匹配（NCSN）](04-noise-conditional-score-networks.md#sec-smld)

<span id="sec-comparison-ncsn-ddpm"></span>

[3.5 小结：NCSN 与 DDPM 的比较](05-comparison.md#sec-comparison-ncsn-ddpm)

<span id="sec-ch3-cr"></span>

[3.6 结语](06-closing-remarks.md#sec-ch3-cr)
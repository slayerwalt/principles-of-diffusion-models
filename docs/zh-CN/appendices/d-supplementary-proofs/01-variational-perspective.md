# D.1 变分视角 {#app-sec-var-proof}

## D.1.1 [定理 2.2.1](../../chapters/02-variational-perspective/02-ddpm.md#thm-equiv-marginal-kl): 边缘与条件 KL散度最小化的等价性 {#subsec-proof-equiv-marginal-kl}

***证明。*** **[式（2.2.3）](../../chapters/02-variational-perspective/02-ddpm.md#eq-kl-matching)的推导。**

首先展开右侧的期望：

$$
\begin{aligned}
&\mathbb{E}_{p({\mathbf{x}}_0, {\mathbf{x}}_i)}\left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0) \| p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right]
    \\
    =& \int \int p({\mathbf{x}}_0, {\mathbf{x}}_i)
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0) \| p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \mathop{}\!\mathrm{d}{\mathbf{x}}_i.
\end{aligned}
$$

根据 KL散度的定义，

$$
\begin{aligned}
&\mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0) \| p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    = \int p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \mathop{}\!\mathrm{d}{\mathbf{x}}_{i-1}.
\end{aligned}
$$

将其代入期望，得到

$$
\begin{aligned}
&\int \int \int p({\mathbf{x}}_0, {\mathbf{x}}_i)  p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \mathop{}\!\mathrm{d}{\mathbf{x}}_{i-1} \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \mathop{}\!\mathrm{d}{\mathbf{x}}_i.
\end{aligned}
$$

利用概率的链式法则，

$$
\begin{aligned}
p({\mathbf{x}}_0, {\mathbf{x}}_i)
    &= p({\mathbf{x}}_i)   p({\mathbf{x}}_0|{\mathbf{x}}_i),
\end{aligned}
$$

可将积分改写为

$$
\begin{aligned}
&\int p({\mathbf{x}}_i) \int p({\mathbf{x}}_0|{\mathbf{x}}_i) \int p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \mathop{}\!\mathrm{d}{\mathbf{x}}_{i-1} \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \mathop{}\!\mathrm{d}{\mathbf{x}}_i.
\end{aligned}
$$

于是可以将期望写成嵌套形式：

$$
\begin{aligned}
&\mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right]
    \right]
    \right].
\end{aligned}
$$

接下来分解对数项：

$$
\begin{aligned}
\log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    =\log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    + \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}.
\end{aligned}
$$

将其代回期望，得到两项：

$$
\begin{aligned}
&\mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right]
    \right]
    \right]
    \\
    &+
    \mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right]
    \right]
    \right].
\end{aligned}
$$

由于第二个对数项不依赖于 ${\mathbf{x}}_0$，根据全概率公式，

$$
\begin{aligned}
\mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right]
    \right]
    = \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right].
\end{aligned}
$$

类似地，第一项为 KL散度

$$
\begin{aligned}
\mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)  \|  p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right].
\end{aligned}
$$

综合以上结果，得到如下分解：

$$
\begin{aligned}
&\mathbb{E}_{p({\mathbf{x}}_0, {\mathbf{x}}_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)  \|  p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right]
    \\
    =& \mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)  \|  p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right]
    \right]
    \\
    &+
    \mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)  \|  p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right].
\end{aligned}
$$

**最优性的证明。** 需要证明：

$$
p^*({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)
=
p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)
=
\mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)}
\left[
p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i,{\mathbf{x}}_0)
\right],
\qquad
{\mathbf{x}}_i \sim p_i.
$$

第一个恒等式来自以下事实：假设参数化具有足够的表达能力，KL散度 $\mathcal{D}_{\mathrm{KL}}(p \| p_{\boldsymbol{\phi}})$ 在 $p^* = p$ 时取得最小值。第二个恒等式直接由全概率公式得到。$\blacksquare$

## D.1.2 [定理 2.2.3](../../chapters/02-variational-perspective/02-ddpm.md#thm-ddpm-elbo): 扩散模型的 ELBO {#app-vlb-proof}

***证明。*** 为简化记号，记

$$
{\mathbf{x}}_{1:L}:=({\mathbf{x}}_1,\ldots,{\mathbf{x}}_L).
$$

**步骤 1：应用 Jensen 不等式。** 边缘对数似然为

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0)
=
\log \int p_{\boldsymbol{\phi}}({\mathbf{x}}_0,{\mathbf{x}}_{1:L})\,\mathop{}\!\mathrm{d} {\mathbf{x}}_{1:L},
$$

其中联合生成分布为

$$
p_{\boldsymbol{\phi}}({\mathbf{x}}_0,{\mathbf{x}}_{1:L})
=
p_{\text{prior}}({\mathbf{x}}_L)\prod_{i=1}^L p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}| {\mathbf{x}}_i).
$$

引入前向加噪分布

$$
p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)
=
\prod_{i=1}^L p({\mathbf{x}}_i| {\mathbf{x}}_{i-1}),
$$

并改写为

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0)
=
\log \int
p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)
\frac{p_{\boldsymbol{\phi}}({\mathbf{x}}_0,{\mathbf{x}}_{1:L})}{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\,\mathop{}\!\mathrm{d} {\mathbf{x}}_{1:L}.
$$

应用 Jensen 不等式，得到

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0)
\ge
\mathbb{E}_{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\left[
\log
\frac{p_{\boldsymbol{\phi}}({\mathbf{x}}_0,{\mathbf{x}}_{1:L})}{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\right]
=: \mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi}),
$$

因此

$$
-\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0)
\le
-\mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi}).
$$

**步骤 2：展开 ELBO。** 将因子分解代入 ELBO，得到

$$
\begin{aligned}
\mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi})
&=
\mathbb{E}_{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\Bigg[
\log p_{\text{prior}}({\mathbf{x}}_L)
+
\sum_{i=1}^L \log p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}| {\mathbf{x}}_i)
-
\sum_{i=1}^L \log p({\mathbf{x}}_i| {\mathbf{x}}_{i-1})
\Bigg].
\end{aligned}
$$

因此，

$$
\begin{aligned}
-\mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi})
&=
\mathbb{E}_{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\Bigg[
\log \frac{p({\mathbf{x}}_L| {\mathbf{x}}_0)}{p_{\text{prior}}({\mathbf{x}}_L)}
+
\sum_{i=2}^L
\log \frac{p({\mathbf{x}}_{i-1}| {\mathbf{x}}_i,{\mathbf{x}}_0)}{p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}| {\mathbf{x}}_i)}
-
\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0| {\mathbf{x}}_1)
\Bigg].
\end{aligned}
$$

这里使用了恒等式

$$
p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)
=
p({\mathbf{x}}_L| {\mathbf{x}}_0)\prod_{i=2}^L p({\mathbf{x}}_{i-1}| {\mathbf{x}}_i,{\mathbf{x}}_0),
$$

它由前向过程的 Markov 结构得到。

现在根据各项的依赖关系进行分组：

$$
\begin{aligned}
-\mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi})
&=
\underbrace{
\mathbb{E}_{p({\mathbf{x}}_L| {\mathbf{x}}_0)}
\left[
\log \frac{p({\mathbf{x}}_L| {\mathbf{x}}_0)}{p_{\text{prior}}({\mathbf{x}}_L)}
\right]
}_{\mathcal{L}_{\text{prior}}({\mathbf{x}}_0)}
+
\underbrace{
\mathbb{E}_{p({\mathbf{x}}_1| {\mathbf{x}}_0)}
\left[
-\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0| {\mathbf{x}}_1)
\right]
}_{\mathcal{L}_{\text{recon.}}({\mathbf{x}}_0;\boldsymbol{\phi})}
\\
&\quad+
\underbrace{
\sum_{i=2}^L
\mathbb{E}_{p({\mathbf{x}}_i| {\mathbf{x}}_0)}
\left[
\mathcal{D}_{\mathrm{KL}}
\big(
p({\mathbf{x}}_{i-1}| {\mathbf{x}}_i,{\mathbf{x}}_0)
\|
p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}| {\mathbf{x}}_i)
\big)
\right]
}_{\mathcal{L}_{\text{diffusion}}({\mathbf{x}}_0;\boldsymbol{\phi})}.
\end{aligned}
$$

由此证明了所述分解。$\blacksquare$

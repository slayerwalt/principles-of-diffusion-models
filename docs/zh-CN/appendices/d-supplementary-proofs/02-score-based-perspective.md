# D.2 基于 score 的视角 {#app-sec-score-proof}

## D.2.1 [命题 3.2.1](../../chapters/03-score-based-perspective/02-score-based-models.md#sm-trace)：通过分部积分实现可计算的 score 匹配 {#app-sec-sm-trace}

***证明。***

**展开 $\mathcal{L}_{\text{SM}}(\boldsymbol{\phi})$。** 展开期望内部的平方差：

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}) &= \frac{1}{2}\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right\rVert_2^2 - 2 \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle + \left\lVert{\mathbf{s}}({\mathbf{x}})\right\rVert_2^2 \right] \\
        &= \frac{1}{2}\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right\rVert_2^2 \right] - \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle \right] \\
        &\quad + \frac{1}{2}\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}({\mathbf{x}})\right\rVert_2^2 \right].
\end{aligned}
$$

现在考察交叉乘积项：

$$
\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle \right].
$$

利用

$$
\nabla_{{\mathbf{x}}} \log p_{\text{data}}({\mathbf{x}}) = \frac{\nabla_{{\mathbf{x}}}p_{\text{data}}({\mathbf{x}})}{p_{\text{data}}({\mathbf{x}})},
$$

并假设 $p_{\text{data}}({\mathbf{x}})$ 非零（例如在其支撑集上），交叉乘积项可写成：

$$
\begin{aligned}
\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle \right]
             &= \int {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})^\top  \nabla_{{\mathbf{x}}} \log p_{\text{data}}({\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}} \\
             &= \int {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})^\top  \nabla_{{\mathbf{x}}}p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}} \\
             &= \sum_{i=1}^{D} \int  s_{\boldsymbol{\phi}}^{(i)}({\mathbf{x}}) \partial_{x_i}p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}},
\end{aligned}
$$

其中 $s_{\boldsymbol{\phi}}^{(i)}({\mathbf{x}})$ 是 score 函数的第 $i$ 个分量，

$$
{\mathbf{s}}_{\boldsymbol{\phi}} = \left( s_{\boldsymbol{\phi}}^{(1)}, s_{\boldsymbol{\phi}}^{(2)}, \dots, s_{\boldsymbol{\phi}}^{(D)} \right).
$$

**分部积分。** 使用下面由标准微积分得到的分部积分公式（[Evans, 2010](../../references.md#ref-evans10)）：

设 $u, v$ 是定义在球体 $\mathbb{B}(\boldsymbol{0}, R) \subset \mathbb{R}^D$ 上的可微实值函数，球体半径为 $R>0$。则对于 $i = 1, \ldots, D$，下式成立：

$$
\int_{\mathbb{B}(\boldsymbol{0}, R)} u   \partial_{x_i} v   \mathop{}\!\mathrm{d} {\mathbf{x}} = -\int_{\mathbb{B}(\boldsymbol{0}, R)} v   \partial_{x_i} u   \mathop{}\!\mathrm{d} {\mathbf{x}} + \int_{\partial \mathbb{B}(\boldsymbol{0}, R)} u v \nu_i   \mathop{}\!\mathrm{d} S,
$$

其中 $\nu = (\nu_1, \ldots, \nu_D)$ 是边界 $\partial \mathbb{B}(\boldsymbol{0}, R)$ 的外向单位法向量，该边界是半径为 $R>0$ 的球面；$\mathop{}\!\mathrm{d} S$ 是 $\partial \mathbb{B}(\boldsymbol{0}, R)$ 上的表面测度。

将此公式应用于 $u({\mathbf{x}}) := s_{\boldsymbol{\phi}}^{(i)}({\mathbf{x}})$ 和 $v({\mathbf{x}}) = p_{\text{data}}({\mathbf{x}})$，对所有 $i = 1, \dots, D$ 均如此，并假设

$$
|u({\mathbf{x}}) v({\mathbf{x}})| \to 0 \quad \text{as } R \to \infty.
$$

对所有 $i = 1, \dots, D$ 求和，得到：

$$
\begin{aligned}
\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle \right]
             &= - \sum_{i=1}^{D} \int  \partial_{x_i}s_{\boldsymbol{\phi}}^{(i)}({\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}} \\
             &= - \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[\operatorname{Tr}\left(\nabla_{{\mathbf{x}}}{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right) \right].
\end{aligned}
$$

综合以上结果，有：

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}) &= \underbrace{\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[\operatorname{Tr}\left(\nabla_{{\mathbf{x}}}{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right) + \frac{1}{2}  \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right\rVert_2^2 \right]}_{\widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol{\phi})} \\
        &\quad + \underbrace{\frac{1}{2}\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}({\mathbf{x}})\right\rVert_2^2 \right]}_{=:C},
\end{aligned}
$$

其中 $C$ 仅依赖于分布 $p_{\text{data}}$，证明完毕。$\blacksquare$

## D.2.2 [定理 3.3.1](../../chapters/03-score-based-perspective/03-denoising-score-matching.md#thm-sm-dsm)：SM 与 DSM 最小化的等价性 {#app-sec-sm-dsm}

***证明。*** 展开 $\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma)$ 和 $\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma)$，得到：

$$
\begin{aligned}
&\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma) = \frac{1}{2} \mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}
 \begin{aligned}[t]
         \Big[
    \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2
    &- 2 {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}) \\
    &+ \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}})\right\rVert_2^2
    \Big],
 \end{aligned}
\\
&\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma) = \frac{1}{2} \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})}
 \begin{aligned}[t]
         \Big[
    \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2
    &- 2 {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top \nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) \\
    &+ \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})\right\rVert_2^2
    \Big].
 \end{aligned}
\end{aligned}
$$

两式相减，得到：

$$
\begin{aligned}
&\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma) -\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma)
 \\= \frac{1}{2} &\Bigg( \mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})} \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2 - \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})} \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2\Bigg)
 \\ \quad - &
 \begin{aligned}[t]
 \Bigg(
      \mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}  &\left[ {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}) \right] \\
      &- \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})}\left[ {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}| {\mathbf{x}})\right]
      \Bigg)
 \end{aligned}
 \\ \quad +  \frac{1}{2}&\Bigg(\mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}  \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}})\right\rVert_2^2  -  \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})} \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})\right\rVert_2^2   \Bigg).
\end{aligned}
$$

接下来逐项处理这三项。对于第一项，由于 $p_\sigma(\tilde{{\mathbf{x}}}) = \int p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}}$，可将其改写为：

$$
\begin{aligned}
\mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})} \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2
    &= \int \Big( \int p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) p_{\text{data}}({\mathbf{x}})  \mathop{}\!\mathrm{d}{\mathbf{x}} \Big) \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2 \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}
    \\ &= \int p_{\text{data}}({\mathbf{x}}) \int p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2 \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}} \mathop{}\!\mathrm{d}{\mathbf{x}}
    \\ &= \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})} \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2.
\end{aligned}
$$

因此，第一项为零。对于第二项：

<span id="eq-dsm-sec-term"></span>

$$
\begin{aligned}
          &\mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}\left[ {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}) \right]
      \\=&\int  p_\sigma(\tilde{{\mathbf{x}}}) {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \frac{\nabla_{\tilde{{\mathbf{x}}}} p_\sigma(\tilde{{\mathbf{x}}})}{p_\sigma(\tilde{{\mathbf{x}}})}  \mathop{}\!\mathrm{d}\tilde{\mathbf{x}}
        \\=&\int   {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \int p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}} \mathop{}\!\mathrm{d}\tilde{\mathbf{x}}
    \\=&\int \int  {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top    \nabla_{\tilde{{\mathbf{x}}}}p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d}\tilde{\mathbf{x}} \mathop{}\!\mathrm{d}{\mathbf{x}}
      \\=  &\mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})}\left[ {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}| {\mathbf{x}})\right].
\end{aligned}
\tag{D.2.1}
$$

因此，它也为零。对于第三项，注意到：

$$
C:=\frac{1}{2}\Bigg(\mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}  \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}})\right\rVert_2^2  -  \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})} \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})\right\rVert_2^2   \Bigg)
$$

仅依赖于 $p_{\text{data}}({\mathbf{x}})$ 和 $p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}})$，因此相对于 $\boldsymbol{\phi}$ 为常数。$\blacksquare$

## D.2.3 [引理 3.3.2](../../chapters/03-score-based-perspective/03-denoising-score-matching.md#tweedie)：Tweedie 公式 {#app-sec-tweedie}

首先给出 Tweedie 公式的一种更一般形式，它考虑了随时间变化的 Gaussian 扰动，并在下文给出证明。

**含时变参数的 Tweedie 恒等式。** 设 ${\mathbf{x}}_t|{\mathbf{x}}_0 \sim \mathcal{N}\big(\cdot; \alpha_t{\mathbf{x}}_0, \sigma_t^2{\mathbf{I}}\big)$。则 Tweedie 公式给出

$$
\begin{aligned}
\alpha_t \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0|{\mathbf{x}}_t)}[\mathbf{x}_0|{\mathbf{x}}_t] = {\mathbf{x}}_t + \sigma_t^2 \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t),
\end{aligned}
$$

其中期望相对于后验分布 $p(\mathbf{x}_0|{\mathbf{x}}_t)$ 计算，即 $\mathbf{x}_0$ 在给定观测 ${\mathbf{x}}_t$ 时的分布；$p_t({\mathbf{x}}_t)$ 是 ${\mathbf{x}}_t$ 的边缘密度。

***证明。***

**边缘密度及其 score。** 回顾 ${\mathbf{x}}_t$ 的边缘密度：

$$
\begin{aligned}
p_t({\mathbf{x}}_t) = \int p_t({\mathbf{x}}_t|{\mathbf{x}}_0)  p_0({\mathbf{x}}_0)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

现在计算 score 函数：

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= \frac{\nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t)}{p_t({\mathbf{x}}_t)} = \frac{1}{p_t({\mathbf{x}}_t)} \int \nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t|{\mathbf{x}}_0)  p_0({\mathbf{x}}_0)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

因此，需要计算条件密度的梯度。

**条件密度的梯度与整理。** 条件 Gaussian 密度的梯度为：

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
= -p_t({\mathbf{x}}_t|{\mathbf{x}}_0) \cdot \sigma_t^{-2} ({\mathbf{x}}_t - \alpha_t {\mathbf{x}}_0).
\end{aligned}
$$

将其代入上式，得到：

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t)
&= \int \nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t|{\mathbf{x}}_0)  p_0({\mathbf{x}}_0)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \\
&= -\sigma_t^{-2} \int ({\mathbf{x}}_t - \alpha_t {\mathbf{x}}_0)  p_t({\mathbf{x}}_t|{\mathbf{x}}_0)  p_0({\mathbf{x}}_0)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \\
&= -\sigma_t^{-2} \int ({\mathbf{x}}_t - \alpha_t {\mathbf{x}}_0)  p({\mathbf{x}}_0|{\mathbf{x}}_t)  p_t({\mathbf{x}}_t)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \\
&= -p_t({\mathbf{x}}_t)  \sigma_t^{-2} \left({\mathbf{x}}_t - \alpha_t  \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[{\mathbf{x}}_0\vert {\mathbf{x}}_t] \right).
\end{aligned}
$$

两边除以 $p_t({\mathbf{x}}_t)$，得到：

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= -\sigma_t^{-2} \left({\mathbf{x}}_t - \alpha_t  \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[{\mathbf{x}}_0\vert {\mathbf{x}}_t] \right).
\end{aligned}
$$

整理可得：

$$
\begin{aligned}
{\mathbf{x}}_t + \sigma_t^{2} \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= \alpha_t  \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[{\mathbf{x}}_0\vert {\mathbf{x}}_t].
\end{aligned}
$$

推导完毕。

## D.2.4 Stein 恒等式与 SURE 替代目标 {#app-sec-stein-identity}

**Stein 恒等式。** Stein 恒等式是一种分部积分技巧，它将未知密度下的期望转化为可观测函数及其导数的期望，从而消去配分函数，无需计算未知密度或配分函数，就能得到无偏、可计算的目标和检验。我们先从最简单的一维情形开始，再推广到证明 SURE 替代损失所需的形式。

**一维标准正态情形。** 若 $z \sim \mathcal{N}(0,1)$，且 $f$ 满足适当的衰减条件，则 Stein 恒等式为：

$$
{\mathbb{E}}[f'(z)]  =  {\mathbb{E}}[z f(z)] .
$$

记 $\phi(z) := \tfrac{1}{\sqrt{2\pi}} e^{-z^2/2}$ 为一维标准正态密度。利用 $\phi'(z) = -z\phi(z)$ 和边界项消失这一事实，通过分部积分即可证明。具体计算如下：

$$
{\mathbb{E}}[f'(z)]
= \int_{-\infty}^{\infty} f'(z)  \phi(z)  \mathop{}\!\mathrm{d} z.
$$

进行分部积分，取 $u=\phi(z)$ 和 $\mathop{}\!\mathrm{d} v=f'(z) \mathop{}\!\mathrm{d} z$，得到

$$
\int f'(z)  \phi(z)  \mathop{}\!\mathrm{d} z
= \Big[ f(z)  \phi(z) \Big]_{-\infty}^{\infty}
- \int f(z)  \phi'(z)  \mathop{}\!\mathrm{d} z.
$$

由于 $\phi'(z) = -z\phi(z)$，并且 $f(z)\phi(z)\to 0$ 在 $|z|\to\infty$ 时成立（衰减条件），边界项消失，从而

$$
{\mathbb{E}}[f'(z)]  =  \int f(z)  z  \phi(z)  \mathop{}\!\mathrm{d} z  =  {\mathbb{E}}[z f(z)].
$$

推导完毕，这证明了 Stein 恒等式的一维情形。

**多元标准正态情形。** 若 ${\mathbf{z}} \sim \mathcal{N}(\mathbf{0},{\mathbf{I}}_D)$ 且 $g:{\mathbb{R}}^D \to {\mathbb{R}}$，则 Stein 恒等式为

$$
{\mathbb{E}}[\nabla g({\mathbf{z}})]  =  {\mathbb{E}}[{\mathbf{z}} g({\mathbf{z}})].
$$

等价地，对于 $\mathbf{u}:{\mathbb{R}}^D \to {\mathbb{R}}^D$，

<span id="eq-stein-multi"></span>

$$
\begin{aligned}
{\mathbb{E}}[ \nabla_{{\mathbf{z}}} \cdot\mathbf{u}({\mathbf{z}})]  =  {\mathbb{E}}[{\mathbf{z}}^\top \mathbf{u}({\mathbf{z}})] .
\end{aligned}
\tag{D.2.2}
$$

**SURE 所需的恒等式。** 设 $\tilde{{\mathbf{x}}}={\mathbf{x}}+\sigma{\mathbf{z}}$，其中 ${\mathbf{z}} \sim \mathcal{N}(\mathbf{0},{\mathbf{I}}_D)$，并取任意满足适当正则条件的向量函数 $\mathbf{a}$，则

<span id="eq-stein-for-sure"></span>

$$
\begin{aligned}
{\mathbb{E}}\big[(\tilde{{\mathbf{x}}}-{\mathbf{x}})^\top \mathbf{a}(\tilde{{\mathbf{x}}})  \big|  {\mathbf{x}}\big]
 =  \sigma^2 {\mathbb{E}}\big[ \nabla_{\tilde{{\mathbf{x}}}} \cdot\mathbf{a}(\tilde{{\mathbf{x}}})  \big|  {\mathbf{x}}\big].
\end{aligned}
\tag{D.2.3}
$$

应用[式（D.2.2）](02-score-based-perspective.md#eq-stein-multi)并使用链式法则即可得到该式。

**从条件 MSE 推导 SURE。** 设 ${\mathbf{D}}:{\mathbb{R}}^D \to {\mathbb{R}}^D$ 为去噪器，定义

$$
R({\mathbf{D}};{\mathbf{x}}):={\mathbb{E}} \left[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-{\mathbf{x}}\|_2^2| {\mathbf{x}}\right].
$$

围绕 $\tilde{{\mathbf{x}}}$ 展开：

$$
\begin{aligned}
&~R({\mathbf{D}};{\mathbf{x}}) \\
=&~{\mathbb{E}} \left[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|^2  \big|  {\mathbf{x}}\right]
+2 {\mathbb{E}} \left[({\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}})^\top(\tilde{{\mathbf{x}}}-{\mathbf{x}})  \big|  {\mathbf{x}}\right]
+{\mathbb{E}} \left[\|\tilde{{\mathbf{x}}}-{\mathbf{x}}\|^2  \big|  {\mathbf{x}}\right] \\
=&~{\mathbb{E}} \left[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|^2  \big|  {\mathbf{x}}\right]
+2\Big(\underbrace{{\mathbb{E}}[(\tilde{{\mathbf{x}}}-{\mathbf{x}})^\top {\mathbf{D}}(\tilde{{\mathbf{x}}})|{\mathbf{x}}]}_{\substack{\sigma^2 {\mathbb{E}}[ \nabla_{\tilde{{\mathbf{x}}}} \cdot{\mathbf{D}}(\tilde{{\mathbf{x}}})|{\mathbf{x}}] \\  \text{ by }\href{../02-score-based-perspective/#eq-stein-for-sure}{\text{Equation (D.2.3)}}
}}
-\underbrace{{\mathbb{E}}[(\tilde{{\mathbf{x}}}-{\mathbf{x}})^\top \tilde{{\mathbf{x}}}|{\mathbf{x}}]}_{ \sigma^2 D}\Big)
\\&\qquad\qquad\qquad+\underbrace{{\mathbb{E}}[\|\tilde{{\mathbf{x}}}-{\mathbf{x}}\|^2|{\mathbf{x}}]}_{ \sigma^2 D} \\
=&~{\mathbb{E}} \left[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|^2
+2\sigma^2  \nabla_{\tilde{{\mathbf{x}}}} \cdot{\mathbf{D}}(\tilde{{\mathbf{x}}})
-D\sigma^2 | {\mathbf{x}}\right].
\end{aligned}
$$

因此，*可观测的*替代量

$$
\mathrm{SURE}({\mathbf{D}};\tilde{{\mathbf{x}}})
:= \|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|_2^2
+2\sigma^2  \nabla_{\tilde{{\mathbf{x}}}} \cdot{\mathbf{D}}(\tilde{{\mathbf{x}}})
-D\sigma^2
$$

满足 ${\mathbb{E}} \left[\mathrm{SURE}({\mathbf{D}};\tilde{{\mathbf{x}}}) \big| {\mathbf{x}}\right]=R({\mathbf{D}};{\mathbf{x}})$。所以，最小化 SURE（在期望意义下或经验上）等价于最小化真实的条件 MSE，同时只需使用带噪观测。

## D.2.5 [定理 4.2.1](../../chapters/04-score-sde-framework/02-matching-marginals.md#thm-fpe)：通过 Fokker–Planck 方程实现边缘分布对齐 {#app-sec-scoresde-fpe-proof}

***证明。***

**第一部分：前向 SDE 的 Fokker–Planck 方程。**

考虑前向 SDE：

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}(t) = {\mathbf{f}}({\mathbf{x}}(t), t)  \mathop{}\!\mathrm{d} t + g(t)  \mathop{}\!\mathrm{d} {\mathbf{w}}(t).
$$

扩散矩阵为 $\sigma(t) = g(t) I_D$，因此 $\sigma(t) \sigma(t)^T = g^2(t) I_D$。$p_t({\mathbf{x}})$ 是 ${\mathbf{x}}(t)$ 的边缘密度，其 Fokker–Planck 方程为：

$$
\partial_t p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{f}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] + \frac{1}{2} \sum_{i,j=1}^D \frac{\partial^2}{\partial x_i \partial x_j} \bigl[ (g^2(t) \delta_{ij}) p_t({\mathbf{x}}) \bigr].
$$

计算扩散项：

$$
\sum_{i,j=1}^D \frac{\partial^2}{\partial x_i \partial x_j} \bigl[ g^2(t) \delta_{ij} p_t({\mathbf{x}}) \bigr] = \sum_{i=1}^D \frac{\partial^2}{\partial x_i^2} \bigl[ g^2(t) p_t({\mathbf{x}}) \bigr] = g^2(t) \Delta_{{\mathbf{x}}} p_t({\mathbf{x}}).
$$

于是：

$$
\partial_t p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{f}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] + \frac{1}{2} g^2(t) \Delta_{{\mathbf{x}}} p_t({\mathbf{x}}).
$$

现在利用下式进行改写：

$$
{\mathbf{v}}({\mathbf{x}}, t) = {\mathbf{f}}({\mathbf{x}}, t) - \frac{1}{2} g^2(t) \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}).
$$

由于 $\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}) = \frac{\nabla_{{\mathbf{x}}} p_t({\mathbf{x}})}{p_t({\mathbf{x}})}$，计算可得：

$$
\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] = \nabla_{{\mathbf{x}}} \cdot \biggl[ {\mathbf{f}}({\mathbf{x}}, t) p_t({\mathbf{x}}) - \frac{1}{2} g^2(t) \frac{\nabla_{{\mathbf{x}}} p_t({\mathbf{x}})}{p_t({\mathbf{x}})} p_t({\mathbf{x}}) \biggr].
$$

第二项为：

$$
\nabla_{{\mathbf{x}}} \cdot \biggl[ -\frac{1}{2} g^2(t) \nabla_{{\mathbf{x}}} p_t({\mathbf{x}}) \biggr] = -\frac{1}{2} g^2(t) \Delta_{{\mathbf{x}}} p_t({\mathbf{x}}).
$$

于是：

$$
\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] = \nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{f}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] - \frac{1}{2} g^2(t) \Delta_{{\mathbf{x}}} p_t({\mathbf{x}}).
$$

因此：

$$
\partial_t p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr],
$$

由此验证了 Fokker–Planck 方程的两种形式。

**第二部分：PF-ODE 的边缘密度。**

考虑 PF-ODE：

$$
\frac{\mathop{}\!\mathrm{d} \tilde{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{v}}(\tilde{\mathbf{x}}(t), t), \quad {\mathbf{v}}({\mathbf{x}}, t) = {\mathbf{f}}({\mathbf{x}}, t) - \frac{1}{2} g^2(t) \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}).
$$

**前向：$\tilde{\mathbf{x}}(0) \sim p_0$。** 设 $\tilde{\mathbf{x}}(t)$ 按 PF-ODE 演化，且 $\tilde{\mathbf{x}}(0) \sim p_0$。流映射 $\boldsymbol{\Psi}_t: \mathbb{R}^D \to \mathbb{R}^D$ 定义为：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \boldsymbol{\Psi}_t(\mathbf{x}_0) = {\mathbf{v}}(\boldsymbol{\Psi}_t(\mathbf{x}_0), t), \quad \boldsymbol{\Psi}_0(\mathbf{x}_0) = \mathbf{x}_0.
$$

由于 $\tilde{\mathbf{x}}(t) = \boldsymbol{\Psi}_t(\tilde{\mathbf{x}}(0))$，密度 $\tilde p_t({\mathbf{x}})$（对应于 $\tilde{\mathbf{x}}(t)$）满足连续性方程：

$$
\partial_t \tilde p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) \tilde p_t({\mathbf{x}}) \bigr].
$$

由于 $\tilde{\mathbf{x}}(0) \sim p_0$，有 $\tilde p_0({\mathbf{x}}) = p_0({\mathbf{x}})$。由第一部分，$p_t({\mathbf{x}})$ 满足：

$$
\partial_t p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr].
$$

$\tilde p_t$ 和 $p_t$ 满足相同的连续性方程，且具有相同的初始条件 $p_0$。假设具有足够的光滑性（例如 ${\mathbf{v}} \in C^1$），则在某个适当的函数空间中解是唯一的，因此 $\tilde p_t = p_t$。于是，$\tilde{\mathbf{x}}(t) \sim p_t$ 对所有 $t \in [0, T]$ 成立。

**逆向：$\tilde{\mathbf{x}}(T) \sim p_T$。** 现在设 $\tilde{\mathbf{x}}(t)$ 按 PF-ODE 从 $t = T$ 逆向演化到 $t = 0$，且 $\tilde{\mathbf{x}}(T) \sim p_T$。对应的 ODE 为：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \tilde{\mathbf{x}}(t) = {\mathbf{v}}(\tilde{\mathbf{x}}(t), t).
$$

令 $s = T - t$，则逆向 ODE 变为：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s} \tilde{\mathbf{x}}(T - s) = -{\mathbf{v}}(\tilde{\mathbf{x}}(T - s), T - s).
$$

$\tilde p_{T-s}({\mathbf{x}})$ 是 $\tilde{\mathbf{x}}(T - s)$ 的密度，满足：

$$
\partial_s \tilde p_{T-s}({\mathbf{x}}) = \nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, T - s) \tilde p_{T-s}({\mathbf{x}}) \bigr].
$$

由于 $\tilde{\mathbf{x}}(T) \sim p_T$，有 $\tilde p_T = p_T$。$p_t$ 在 $t = T - s$ 时的 Fokker–Planck 方程为：

$$
\partial_t p_{T-s}({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, T - s) p_{T-s}({\mathbf{x}}) \bigr].
$$

由于 $\partial_t = -\partial_s$，得到：

$$
\partial_s p_{T-s}({\mathbf{x}}) = \nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, T - s) p_{T-s}({\mathbf{x}}) \bigr].
$$

$\tilde p_{T-s}$ 和 $p_{T-s}$ 满足相同的 PDE，并在 $s = 0$ 处具有相同的初始条件（$\tilde p_T = p_T$）。由唯一性可得 $\tilde p_{T-s} = p_{T-s}$，所以 $\tilde{\mathbf{x}}(t) = \tilde{\mathbf{x}}(T - s) \sim p_{T-s} = p_t$ 对所有 $t \in [0, T]$ 成立。

**第三部分：逆时间 SDE 的边缘密度。** 利用正向运行的反向时钟验证边缘分布：

$$
s:=T-t.
$$

定义

$$
{\mathbf{y}}(s):=\bar{\mathbf{x}}(T-s),
\qquad s\in[0,T].
$$

由于逆向过程从 $\bar{\mathbf{x}}(T)\sim p_T$ 出发，有

$$
{\mathbf{y}}(0)\sim p_T.
$$

经过这一时钟变换，逆时间 SDE 变为

$$
\mathop{}\!\mathrm{d}{\mathbf{y}}(s)
=
\Bigl[
-{\mathbf{f}}({\mathbf{y}}(s),T-s)
+
g^2(T-s)\nabla_{{\mathbf{x}}}\log p_{T-s}({\mathbf{y}}(s))
\Bigr]\mathop{}\!\mathrm{d} s
+
g(T-s)\mathop{}\!\mathrm{d}{\mathbf{w}}^{\mathrm{rev}}(s),
$$

其中 ${\mathbf{w}}^{\mathrm{rev}}$ 是正向运行时钟 $s$ 下的标准 Wiener 过程。

记 $q_s$ 为 ${\mathbf{y}}(s)$ 的密度。其 Fokker–Planck 方程为

$$
\begin{aligned}
\partial_s q_s({\mathbf{x}})
&=
-\nabla_{{\mathbf{x}}}\cdot
\Bigl(
\bigl[
-{\mathbf{f}}({\mathbf{x}},T-s)
+
g^2(T-s)\nabla_{{\mathbf{x}}}\log p_{T-s}({\mathbf{x}})
\bigr]
q_s({\mathbf{x}})
\Bigr)
\\
&\qquad
+
\tfrac12 g^2(T-s)\Delta_{{\mathbf{x}}}q_s({\mathbf{x}}).
\end{aligned}
$$

现在验证

$$
q_s({\mathbf{x}})=p_{T-s}({\mathbf{x}})
$$

满足该方程。利用

$$
\nabla_{{\mathbf{x}}}\log p_{T-s}({\mathbf{x}})\,p_{T-s}({\mathbf{x}})
=
\nabla_{{\mathbf{x}}}p_{T-s}({\mathbf{x}}),
$$

右侧变为

$$
\begin{aligned}
&\nabla_{{\mathbf{x}}}\cdot
\bigl[
{\mathbf{f}}({\mathbf{x}},T-s)p_{T-s}({\mathbf{x}})
\bigr]
-
g^2(T-s)\Delta_{{\mathbf{x}}}p_{T-s}({\mathbf{x}})
\\
&\qquad
+
\tfrac12 g^2(T-s)\Delta_{{\mathbf{x}}}p_{T-s}({\mathbf{x}})
\\
&=
\nabla_{{\mathbf{x}}}\cdot
\bigl[
{\mathbf{f}}({\mathbf{x}},T-s)p_{T-s}({\mathbf{x}})
\bigr]
-
\tfrac12 g^2(T-s)\Delta_{{\mathbf{x}}}p_{T-s}({\mathbf{x}}).
\end{aligned}
$$

另一方面，前向 Fokker–Planck 方程给出

$$
\partial_t p_t({\mathbf{x}})
=
-\nabla_{{\mathbf{x}}}\cdot
\bigl[{\mathbf{f}}({\mathbf{x}},t)p_t({\mathbf{x}})\bigr]
+
\tfrac12g^2(t)\Delta_{{\mathbf{x}}}p_t({\mathbf{x}}).
$$

由于 $t=T-s$，链式法则给出

$$
\begin{aligned}
\partial_s p_{T-s}({\mathbf{x}})
&=
-\partial_t p_t({\mathbf{x}})\big|_{t=T-s}
\\
&=
\nabla_{{\mathbf{x}}}\cdot
\bigl[
{\mathbf{f}}({\mathbf{x}},T-s)p_{T-s}({\mathbf{x}})
\bigr]
-
\tfrac12g^2(T-s)\Delta_{{\mathbf{x}}}p_{T-s}({\mathbf{x}}).
\end{aligned}
$$

这正是逆向过程的 Fokker–Planck 方程。

最后，$q_0=p_T$，与初始分布 ${\mathbf{y}}(0)\sim p_T$ 相符。在确保 Fokker–Planck 方程解唯一的正则条件下，

$$
q_s=p_{T-s},
\qquad s\in[0,T].
$$

回到原始扩散时钟 $t=T-s$，得到

$$
\bar{\mathbf{x}}(t)\sim p_t.
$$

因此，当从 $t=T$ 到 $t=0$ 逆向求解逆时间 SDE 时，它遵循与前向 SDE 相同的边缘分布，只是顺序相反。$\blacksquare$

## D.2.6 [命题 4.3.1](../../chapters/04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer)：SM 与 DSM 的最小化解 {#app-sec-min-dsm}

***证明。*** 为求最小化解 $\mathbf{s}^*$，先固定时间 $t$，分析目标函数中的内层期望：

$$
\begin{aligned}
\mathcal  J(t, \boldsymbol{\phi})
&:= \mathbb{E}_{{\mathbf{x}}_0 \sim p_{\text{data}}}
\mathbb{E}_{{\mathbf{x}}_t \sim p_t(\cdot|{\mathbf{x}}_0)}
\left[\left\| \mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t|{\mathbf{x}}_0) \right\|_2^2 \right].
\end{aligned}
$$

要最小化这一期望，需要找到 $\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$，使其对每个 ${\mathbf{x}}_t$ 都最小化期望平方误差。利用 $X_0$ 和 $X_t$ 的联合分布，可将期望改写为：

$$
\begin{aligned}
\mathcal J(t, \boldsymbol{\phi})
= \iint p_{\text{data}}({\mathbf{x}}_0) p_t({\mathbf{x}}_t |{\mathbf{x}}_0)
\left\| \mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p({\mathbf{x}}_t |  {\mathbf{x}}_0) \right\|_2^2
 \mathop{}\!\mathrm{d}{\mathbf{x}}_0  \mathop{}\!\mathrm{d}{\mathbf{x}}_t.
\end{aligned}
$$

对每个固定的 ${\mathbf{x}}_t$，需要最小化：

$$
\begin{aligned}
\int p({\mathbf{x}}_0 | X_t = {\mathbf{x}}_t) p_t( {\mathbf{x}}_t)
\left\| \mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p({\mathbf{x}}_t |  {\mathbf{x}}_0) \right\|_2^2
 \mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

由于 $p_t({\mathbf{x}}_t)$ 相对于 $\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$ 为常数，这等价于最小化：

$$
\begin{aligned}
\int p( {\mathbf{x}}_0 | X_t = {\mathbf{x}}_t)
\left\| \mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p({\mathbf{x}}_t |  {\mathbf{x}}_0) \right\|_2^2
  \mathop{}\!\mathrm{d}{\mathbf{x}}_0
\end{aligned}
$$

当 $\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$ 等于条件期望时取得最小值：

$$
\begin{aligned}
\mathbf{s}^*({\mathbf{x}}_t, t)
= \mathbb{E}_{X_0 \sim p(X_0 | X_t = {\mathbf{x}}_t)}
\left[ \nabla_{{\mathbf{x}}_t} \log p({\mathbf{x}}_t | X_0) \right].
\end{aligned}
$$

现在需要证明它等于 $\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)$。根据 Bayes 公式和边缘概率的定义：

$$
\begin{aligned}
p_t({\mathbf{x}}_t)
= \int p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

先取对数，再对 ${\mathbf{x}}_t$ 求梯度：

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= \frac{\nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t)}{p_t({\mathbf{x}}_t)}
= \frac{\nabla_{{\mathbf{x}}_t} \int p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0}
{\int p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0}.
\end{aligned}
$$

在适当的正则条件下，可以交换梯度与积分：

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= \frac{\int \nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0}
{\int p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0}.
\end{aligned}
$$

利用

$$
\nabla_{{\mathbf{x}}_t}p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
=
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0),
$$

和 Bayes 公式，得到

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)
&=
\int
p({\mathbf{x}}_0|{\mathbf{x}}_t)
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\mathop{}\!\mathrm{d}{\mathbf{x}}_0 \\
&=
\mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}
\left[
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\right]
=
\mathbf{s}^*({\mathbf{x}}_t,t),
\end{aligned}
$$

证明完毕。$\blacksquare$

## D.2.7 Gaussian 分布的闭式 score 函数 {#closed-form-gaussian-score}

为便于后文引用，下面将一般多元正态分布的 score 公式归纳为一个引理：

<span id="score-general-gaussian"></span>

**引理 D.2.1：Gaussian 分布的 score**

设 ${\mathbf{x}} \in \mathbb{R}^D$，考虑多元正态分布

$$
p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) := \mathcal{N}(\tilde{{\mathbf{x}}}; \boldsymbol{\mu}, \boldsymbol{\Sigma}),
$$

其中 $\boldsymbol{\mu} \in \mathbb{R}^D$ 为均值，$\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$ 为可逆协方差矩阵。其 score 函数为

<span id="eq-score-gaussian-formula"></span>

$$
\begin{aligned}
\nabla_{\tilde{{\mathbf{x}}}} \log p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = -\boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}).
\end{aligned}
\tag{D.2.4}
$$

**引理的证明。**

$p(\tilde{{\mathbf{x}}} | {\mathbf{x}})$ 的密度函数为：

$$
p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}) \right).
$$

为计算 score 函数，先对 $p(\tilde{{\mathbf{x}}} | {\mathbf{x}})$ 取对数：

$$
\log p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = -\frac{D}{2} \log(2\pi) - \frac{1}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}).
$$

现在计算 $\log p(\tilde{{\mathbf{x}}} | {\mathbf{x}})$ 关于 $\tilde{{\mathbf{x}}}$ 的梯度：

$$
\nabla_{\tilde{{\mathbf{x}}}} \log p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = - \frac{1}{2} \nabla_{\tilde{{\mathbf{x}}}} \left( (\tilde{{\mathbf{x}}} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}) \right).
$$

利用链式法则，得到：

$$
\nabla_{\tilde{{\mathbf{x}}}} \left( (\tilde{{\mathbf{x}}} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}) \right) = 2 \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}).
$$

因此，score 函数为：

<span id="eq-gaussian-score"></span>

$$
\begin{aligned}
\nabla_{\tilde{{\mathbf{x}}}} \log p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = - \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}).
\end{aligned}
\tag{D.2.5}
$$

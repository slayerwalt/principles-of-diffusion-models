# D.4 理论补充：扩散模型的统一系统视角 {#app-sec-equiv-para}

## D.4.1 [命题 6.3.1](../../chapters/06-unified-perspective/03-equivalent-parameterizations.md#equi-para)：参数化的等价性 {#equivalence-of-parametrizations}

***证明：*** 与关于 DSM 损失的[命题 4.3.1](../../chapters/04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer)相同，匹配损失

$$
\mathbb{E}_t\left[\omega(t)\,\mathbb{E}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[\|\cdot\|_2^2\right]\right]
$$

在下述条件下达到全局最优：内层期望

$$
\mathbb{E}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[\|\cdot\|_2^2\right]
$$

对于每个固定的 $t$ 都取得最小值。由于这是标准的均方误差问题，最小化解唯一。根据去噪 score 匹配（[Vincent, 2011](../../references.md#ref-vincent2011connection)），[命题 4.3.1](../../chapters/04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer)表明最优 score 函数满足

$$
\begin{aligned}
{\mathbf{s}}^*({\mathbf{x}}_t,t) = \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}\left[\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)\right] = \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t).
\end{aligned}
$$

利用恒等式 $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0) = -\frac{1}{\sigma_t} \boldsymbol{\epsilon}$（针对 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0},{\mathbf{I}})$），得到

$$
\begin{aligned}
{\mathbf{s}}^*({\mathbf{x}}_t,t) = -\frac{1}{\sigma_t}\, \boldsymbol{\epsilon}^*({\mathbf{x}}_t,t),
\end{aligned}
$$

其中 $\boldsymbol{\epsilon}^*({\mathbf{x}}_t,t) = \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[\boldsymbol{\epsilon}|{\mathbf{x}}_t]$ 是最优的 ${\boldsymbol{\epsilon}}$-预测，对应损失 $\mathcal{L}_{\text{noise}}({\boldsymbol{\phi}})$。对于 ${\mathbf{x}}$-预测损失 $\mathcal{L}_{\text{clean}}$，最优估计量为

$$
{\mathbf{x}}^*({\mathbf{x}}_t,t) = \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[{\mathbf{x}}_0|{\mathbf{x}}_t],
$$

根据 Tweedie 公式，它与 score 的关系为

$$
\alpha_t\, {\mathbf{x}}^*({\mathbf{x}}_t,t) = {\mathbf{x}}_t + \sigma_t^2\, {\mathbf{s}}^*({\mathbf{x}}_t,t).
$$

对于 ${\mathbf{v}}$-预测损失 $\mathcal{L}_{\text{velocity}}$，最优估计量为

$$
\begin{aligned}
{\mathbf{v}}^*({\mathbf{x}}_t,t) = \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[\alpha_t' {\mathbf{x}}_0 + \sigma_t' \boldsymbol{\epsilon}|{\mathbf{x}}_t]
    = \alpha_t' {\mathbf{x}}^* + \sigma_t' \boldsymbol{\epsilon}^*.
\end{aligned}
$$

将 ${\mathbf{x}}^*$ 和 $\boldsymbol{\epsilon}^*$ 用 ${\mathbf{s}}^*$ 表示并代入，得到

$$
\begin{aligned}
{\mathbf{v}}^*({\mathbf{x}}_t,t)
    = \frac{\alpha_t'}{\alpha_t} {\mathbf{x}}_t + \left( \frac{\alpha_t'}{\alpha_t} \sigma_t^2 - \sigma_t \sigma_t' \right) {\mathbf{s}}^*({\mathbf{x}}_t,t) = f(t) {\mathbf{x}}_t - \frac{1}{2}g^2(t) {\mathbf{s}}^*({\mathbf{x}}_t,t),
\end{aligned}
$$

推导完毕。

$\blacksquare$

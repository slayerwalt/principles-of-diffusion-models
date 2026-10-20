# 7. (Optional) Diffusion Models and Optimal Transport {#ch-ot-eot}

Mapping one distribution to another (with generation as a special case) is a central challenge. Flow matching addresses this by learning a time-dependent velocity field that transports mass from source to target. This naturally connects to transport theory: classical optimal transport seeks the minimal cost path between distributions, while its entropy regularized form, the Schrödinger bridge, selects the most likely controlled diffusion relative to a reference such as Brownian motion.

In this chapter we review the foundations of optimal transport, entropic optimal transport, and Schrödinger bridges as formulations of the distribution–to–distribution problem. This leads to a central question: to what extent do diffusion models realize such optimal transports? They admit two views: as *stochastic processes*, defined through forward and reverse SDEs, and as *deterministic processes*, given by PF-ODEs. The stochastic view aligns directly with entropic optimal transport, while the PF-ODE does not generally correspond to any known transport objective. This gap leaves an open question: under what conditions can diffusion models be regarded as solving an optimal transport problem?

**After this chapter, you will be able to...**

-   distinguish a transport coupling, a deterministic transport map, a dynamic transport flow, and a stochastic Schrödinger bridge, and connect these notions conceptually to deep generative modeling;

-   understand what classical OT, entropic OT, and the Schrödinger bridge problem optimize, and how these formulations are related;

-   explain why a standard diffusion forward/reverse SDE does not, in general, solve a prescribed two-endpoint Schrödinger bridge problem;

-   explain why a diffusion PF-ODE defines a deterministic transport map but does not, in general, coincide with the quadratic-cost optimal transport map.

**Reader route.** This chapter is optional for the main diffusion-model story. Read it for a precise answer to the question “in what sense, if any, is diffusion optimal transport?”

Continue with the sections below and the [chapter references](references.md#chapter-07-references).

[7.1 Prologue of Distribution-to-Distribution Translation](01-distribution-translation.md#sec-distribution-translation)

[7.2 Taxonomy of the Problem Setups](02-transport-problem-setups.md#sec-ot-intro)

[7.3 Relationship of Variant Optimal Transport Formulations](03-relationships-between-formulations.md#sec-relation-ot)

[7.4 Is Diffusion Model's SDE Optimal Solution to SB Problem?](04-diffusion-sdes-and-schrodinger-bridges.md#sec-dm-and-sb)

[7.5 Is Diffusion Model's ODE an Optimal Map to OT Problem?](05-probability-flow-odes-and-optimal-transport.md#sec-pfode-ot)

[7.6 Closing Remarks](06-closing-remarks.md#sec-ch7-cr)

<span id="sec-ot-intro"></span>

[Section 7.2](02-transport-problem-setups.md#sec-ot-intro)

<span id="eq-monge-formulation"></span>

[Equation (7.2.2)](02-transport-problem-setups.md#eq-monge-formulation)

<span id="eq-monge-ampere"></span>

[Equation (7.2.4)](02-transport-problem-setups.md#eq-monge-ampere)

<span id="subsec-ot-linear-reflow"></span>

[Section 7.5.2](05-probability-flow-odes-and-optimal-transport.md#subsec-ot-linear-reflow)

<span id="eq-reflow-loss-gradient"></span>

[Equation (7.5.2)](05-probability-flow-odes-and-optimal-transport.md#eq-reflow-loss-gradient)

<span id="ot-conditional-flow"></span>

[Proposition 7.5.1](05-probability-flow-odes-and-optimal-transport.md#ot-conditional-flow)

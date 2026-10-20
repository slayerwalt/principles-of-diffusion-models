# 12. A Unifying Principle and the Road Ahead {#ch-epilogue}

> *The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise.*
>
> — Edsger W. Dijkstra

Every chapter of this book has been about the same thing: how probability mass moves under a prescribed transformation, and how to reverse, approximate, or exploit that motion to generate data. In this final chapter, we step back to make this structural unity explicit. We first revisit the change-of-variable principle that underpins everything we have covered ([Section 12.1](01-density-transport-backbone.md#sec-epilogue-backbone)), then explain how the same probability-transport viewpoint extends to discrete state spaces through Markov kernels and continuous-time Markov chains ([Section 12.2](02-discrete-spaces.md#sec-epilogue-discrete)), and close with reflections on the broader landscape ([Section 12.3](03-closing-remarks.md#sec-epilogue-closing)).

**After this chapter, you will be able to...**

-   explain probability transport as the common principle underlying the generative modeling frameworks developed throughout this book;

-   connect probability evolution in continuous spaces, through the continuity and Fokker–Planck equations, with its finite-state counterpart, the master equation;

-   explain how reverse modeling changes in discrete state spaces, where transition kernels, probability ratios, and jump rates replace ordinary Euclidean score and velocity fields;

-   compare the variational, score/ratio, and flow perspectives across continuous and discrete state spaces, identifying what remains common and what changes.

**Reader route.** This chapter is an epilogue rather than a new method chapter. Read it as a synthesis: focus on which principles remain unchanged, and which mathematical objects must change, when moving from continuous to discrete state spaces.

Continue with the sections below and the [references](../../references.md#references).

[12.1 The Density-Transport Backbone](01-density-transport-backbone.md#sec-epilogue-backbone)

[12.2 Beyond Continuous States: Probability Evolution on Discrete Spaces](02-discrete-spaces.md#sec-epilogue-discrete)

[12.3 Closing Remarks of the Book](03-closing-remarks.md#sec-epilogue-closing)

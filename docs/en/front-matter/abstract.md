# Abstract {#abstract}

This book focuses on the principles that have shaped the development of diffusion models, tracing their origins and showing how different formulations arise from common mathematical ideas.

Diffusion modeling begins by specifying a *forward corruption process* that gradually turns data into noise. This forward process links the data distribution to a simple noise distribution by defining a continuous family of intermediate distributions. The core objective of a diffusion model is to construct another process that runs in the opposite direction, transforming noise into data while recovering the same intermediate distributions defined by the forward corruption process.

We describe three complementary ways to formalize this idea. The *variational view*, inspired by variational autoencoders, sees diffusion as learning to remove noise step by step, solving small denoising objectives that together teach the model to turn noise back into data. The *score-based view*, rooted in energy-based modeling, learns the gradient of the evolving data distribution, which indicates how to nudge samples toward more likely regions. The *flow-based view*, related to normalizing flows, treats generation as following a smooth path that moves samples from noise to data under a learned velocity field.

These perspectives share a common backbone: a learned time-dependent velocity field whose flow transports a simple prior to the data. With this in hand, sampling amounts to solving a differential equation that evolves noise into data along a continuous generative trajectory. On this foundation, the book discusses *guidance* for controllable generation, *advanced numerical solvers* for efficient sampling, and diffusion-motivated *flow-map models* that learn direct mappings between arbitrary times along this trajectory.

This book is intended for readers with a basic background in deep learning who seek a clear, conceptual, and mathematically grounded understanding of diffusion models. It develops the core principles underlying the subject, explains the ideas that unify its many formulations, and provides a solid foundation for further study in this rapidly evolving area. As such, it serves both as a principled reference for researchers and as an accessible entry point for students and newcomers. Supplementary materials for the book are available at the book website:

`https://the-principles-of-diffusion-models.github.io/`

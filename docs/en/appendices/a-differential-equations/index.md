# A. Crash Course on Differential Equations {#app-de}

Differential equations (DEs) are fundamental tools for modeling dynamic systems and can be broadly categorized into *ordinary differential equations* (ODEs), *stochastic differential equations* (SDEs), and *partial differential equations* (PDEs).

ODEs describe how a system’s state changes over time according to a precise rule, so that knowing the starting point determines the future path exactly. SDEs add randomness to this evolution, modeling how noise or uncertainty influences the system’s behavior, making the outcome probabilistic rather than fixed. PDEs explain how functions depending on several variables, such as time and space, evolve together, capturing phenomena like heat spreading, waves moving, or *the time evolution of probability densities in stochastic systems* (Spoiler: Fokker-Planck equation). These types of differential equations form a fundamental language for understanding how systems evolve over time and space under both deterministic and random influences.

In this chapter, we provide essential prerequisites on differential equations.

**After this chapter, you will be able to...**

-   interpret an ODE through its velocity field, solution trajectory, and the role of existence and uniqueness;

-   use integrating factors to expose the linear structure of semilinear ODEs and understand why this structure is useful for diffusion ODE solvers;

-   distinguish single-step, multistep, and parallel-in-time numerical viewpoints, and recognize Euler, Runge–Kutta, Adams–Bashforth, and Picard iteration as representative examples;

-   understand how an SDE extends deterministic dynamics with Brownian noise, how its differential notation relates to the Itô integral, and how Euler–Maruyama discretizes the resulting stochastic dynamics.

**Reader route.** This appendix provides the differential-equation background used throughout the book. Focus on the connection between dynamics, trajectories, and numerical approximation; the later diffusion chapters will specialize these ideas to probability-flow ODEs and stochastic diffusion processes.

Continue with the sections below and the [appendix references](references.md#appendix-a-references).

[A.1 Foundation of Ordinary Differential Equations](01-ordinary-differential-equations.md#app-ode-intro)

[A.2 Foundation of Stochastic Differential Equations](02-stochastic-differential-equations.md#app-sde-intro)

## Selected topics

<span id="app-exp-int-factor"></span>

[Section A.1.3](01-ordinary-differential-equations.md#app-exp-int-factor)

<span id="app-de-solver"></span>

[Section A.1.4](01-ordinary-differential-equations.md#app-de-solver)

<span id="eq-classic-solvers"></span>

[Equation (A.1.6)](01-ordinary-differential-equations.md#eq-classic-solvers)

<span id="app-de-solver-examples"></span>

[Section A.1.5](01-ordinary-differential-equations.md#app-de-solver-examples)

<span id="app-sde-intro"></span>

[Section A.2](02-stochastic-differential-equations.md#app-sde-intro)

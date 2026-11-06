# Probability, geometry and physics: precise connections

**What is already implemented:** a random *control path* driving a deterministic ODE, independent seeded control realizations, sample-based uncertainty summaries, and exact empirical one-dimensional Wasserstein-1/2 for equally sized samples. **Not implemented:** a stochastic differential equation, Fisher-information metric, Riemannian optimization, high-dimensional optimal transport, or a phase-transition detector.

## 1. From random variables to random trajectories

The generator constructs held Gaussian control levels \(Z_k\sim N(0,1)\) and

\[
u(t)=0.08\sin(0.2t)+0.08 Z_{\lfloor t/T_h\rfloor},
\qquad T_h=1.
\]

Within a single seed, this gives a *fixed function of time* \(u_\omega(t)\), where \(\omega\) labels the random realization. Integrating the ODE yields \(x_\omega(t)\). Across random seeds, the generator induces a probability distribution over trajectories (a random process).

Different seeds are independent *in the pseudorandom generator sense*, conditional on fixed code and seed mapping. Observations across time within **one** trajectory are not independent. Therefore bootstrap resampling individual time points would give misleading uncertainty: replicate at the *seed/trajectory* level instead.

The current experiment uses one fixed training seed and 12 different evaluation seeds; a larger ensemble is a planned robustness study.

## 2. Three notions of response

1. **Pathwise generator response:** paired difference \(x^\varepsilon_\omega(t)-x_\omega(t)\) for the *same* control realization \(\omega\).
2. **Frozen model response:** \(F(S^\varepsilon_\omega)-F(S_\omega)\) for the same paired history and fixed predictor.
3. **Distribution response:** distance between marginal laws \(P_t=\mathrm{Law}(x_\omega(t))\) and \(Q_t=\mathrm{Law}(x^\varepsilon_\omega(t))\) over repeated \(\omega\).

These are not synonyms. A paired response preserves which perturbed trajectory corresponds to which baseline. A marginal distribution distance ordinarily forgets that pairing.

## 3. Empirical one-dimensional optimal transport

For equally weighted empirical distributions with n observations each, sort the samples:

\[
x_{(1)}\le\cdots\le x_{(n)},\quad y_{(1)}\le\cdots\le y_{(n)}.
\]

Then the optimal transport costs on the real line are

\[
W_1(\widehat P,\widehat Q)=\frac1n\sum_{i=1}^n|x_{(i)}-y_{(i)}|,
\]

\[
W_2(\widehat P,\widehat Q)=
\sqrt{\frac1n\sum_{i=1}^n(x_{(i)}-y_{(i)})^2}.
\]

This is a **mathematically exact formula for the empirical measures**, not an exact calculation of unknown population distances. It does not require a neural network or an approximate transport solver.

The code compares the distributions of scalar states at **t=100**, with 12 independent baseline and intervention trajectories. That is one marginal time slice, **not** a Wasserstein distance between entire trajectory distributions in function space.

For a chosen pairing, the paired mean absolute difference is an admissible transport cost; hence empirical \(W_1\) cannot exceed that paired cost. They can coincide in one dimension when the intervention preserves sample ordering, as happened in the pilot. Do not claim this always holds.

See [ode_sensitivity.py](../code/ode_sensitivity.py) and [the empirical results](../reports/SENSITIVITY_VALIDATION.md).

## 4. Statistical geometry and Fisher information (learning roadmap)

A family of probability models \(\{p_\theta(x)\}\) is a statistical manifold only when a suitable differentiable parameterization and regularity conditions are specified. Its Fisher-information matrix can be defined by

\[
I_{ij}(\theta)=\mathbb E_{X\sim p_\theta}
[\partial_{\theta_i}\log p_\theta(X)\,\partial_{\theta_j}\log p_\theta(X)].
\]

Under suitable identifiability and regularity conditions this defines a Riemannian metric on parameter space. That metric is **not the same thing** as the Euclidean derivative \(dx/d\varepsilon\), the Wasserstein distance, or model attention weights. We have not estimated Fisher information, demonstrated curvature or built a Riemannian learning objective.

A defensible later experiment would:
- specify the parameterized distribution \(p_\theta\) (possibly by an observation model with a tractable likelihood);
- prove or check regularity, support and identifiability;
- compute and validate the Fisher matrix on a simple analytic example;
- compare its local distance notion with Wasserstein and trajectory-response measurements.

Before using differential-geometric vocabulary, identify the manifold, its points, tangent vectors and metric.

## 5. Physics interpretation and limitations

For the autonomous, unforced normal form \(\dot x=ax-x^3\), the sign of \(a\) controls local stability; see [mathematical derivations](MATHEMATICAL_DERIVATIONS.md). Random forcing, finite observation intervals and abrupt coefficient changes make inference from the observed time series more complicated.

**Statistical physics** connects microscopic randomness and macroscopic statistical behavior, such as fluctuations, response functions and phase transitions. For this repository, the practical bridge is:
- random control realizations \(\rightarrow\) an ensemble of trajectories;
- a dynamical Jacobian \(\rightarrow\) local deterministic stability;
- response statistics \(\rightarrow\) quantified intervention sensitivity;
- empirical state distributions \(\rightarrow\) probabilistic distance.

It would be incorrect to label the current abrupt regime switch an *early-warning detection success*. Known bifurcation time and an actual early-warning forecast are distinct.

## 6. Suggested reading order

Linear algebra and multivariable calculus \(\rightarrow\) probability and conditional expectation \(\rightarrow\) ODE stability/variational equations \(\rightarrow\) Jacobians and tangent spaces \(\rightarrow\) one-dimensional optimal transport \(\rightarrow\) statistical manifolds/Fisher information \(\rightarrow\) Riemannian geometry and statistical physics.

Foundational sources:
- Peyré & Cuturi, *Computational Optimal Transport*: https://www.math.ucdavis.edu/~saito/data/acha.read.f19/peyre-cuturi_computational-optxport.pdf
- Gabriel Peyré, optimal-transport course notes (1D quantile derivation): https://optimaltransport.github.io/slides-peyre/CourseOT.pdf
- Scheffer et al., “Early-warning signals for critical transitions,” *Nature* (2009): https://www.nature.com/articles/nature08227
- “How Stochasticity Influences Leading Indicators of Critical Transitions”: https://pubmed.ncbi.nlm.nih.gov/29713924/

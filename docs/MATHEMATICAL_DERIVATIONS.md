# Mathematical derivations: generator dynamics and sensitivity

**Purpose.** Derive each equation used by the sensitivity experiment. Distinguish mathematical identities, numerical approximations, and scientific hypotheses. Refer to [the implementation](../code/ode_sensitivity.py) and [numerical validation](../reports/SENSITIVITY_VALIDATION.md).

## 1. Continuous-time system

The current scalar generator is

\[
\dot x(t)=g(x(t),u(t),t)=a(t)x(t)-x(t)^3+b\,u(t),\qquad x(0)=x_0,
\]

where \(b=1\) by default, \(a(t)=-0.8\) for \(t<60\), \(a(t)=0.6\) for \(t\ge60\), and \(u(t)\) is a deterministic sinusoid plus seeded, piecewise-constant Gaussian control levels. Time dependence enters through \(a(t)\) and \(u(t)\).

**Important:** This is an ordinary differential equation driven by a *randomly chosen control path*. It is not an Itô stochastic differential equation; the program does not implement white-noise forcing \(dW_t\) or stochastic calculus. Conditional on a fixed sampled control sequence, the solution is deterministic.

The numerical generator fixes \(a\) and \(u\) within each integration step, using their values at the step midpoint and classical fourth-order Runge–Kutta (RK4). Control and parameter change boundaries occur at step midpoints. For exact alignment of designed intervention times, select event times on integration-grid boundaries. Error statements in this document are conditional on the discrete event schedule.

## 2. Response to a control intervention

For a scalar intervention amplitude \(\varepsilon\) and onset time \(t_0\), let

\[
u_\varepsilon(t)=u(t)+\varepsilon H(t-t_0), \quad
\dot x_\varepsilon=a(t)x_\varepsilon-x_\varepsilon^3+
 b\,[u(t)+\varepsilon H(t-t_0)] .
\]

Here \(H\) is the step indicator. Hold the random control realization, initial state and parameters fixed between simulations. Define the *pathwise sensitivity*

\[
s_u(t)=\left.\frac{\partial x_\varepsilon(t)}{\partial\varepsilon}\right|_{\varepsilon=0}.
\]

Differentiate the ODE with respect to \(\varepsilon\). The chain rule gives

\[
\frac{d s_u}{dt}=\underbrace{(a(t)-3x(t)^2)}_{g_x} s_u
                 +\underbrace{b H(t-t_0)}_{g_\varepsilon},
\qquad s_u(0)=0.
\]

This is a **variational equation**. It is linear in \(s_u\), but its coefficient depends on the nonlinear baseline trajectory \(x(t)\).

Before \(t_0\), the forcing is zero. Starting at \(s_u(0)=0\), the sensitivity remains exactly zero prior to intervention.

## 3. Response to a generator-parameter intervention

Let \(a_\delta(t)=a(t)+\delta H(t-t_0)\). The derivative

\[
s_a(t)=\left.\frac{\partial x_\delta(t)}{\partial\delta}\right|_{\delta=0}
\]

obeys

\[
\boxed{\frac{d s_a}{dt}=(a(t)-3x(t)^2)s_a+x(t)H(t-t_0),\quad s_a(0)=0.}
\]

**Why the forcing terms differ:** \(g\) is linear in control with derivative \(b\), whereas \(\partial g/\partial a=x\). Therefore control-offset sensitivity and parameter-offset sensitivity need not have the same sign, scale or time profile. Compare physical units carefully; equal numeric offsets in \(u\) and \(a\) are not necessarily equal physical interventions.

## 4. Local approximation and finite differences

For sufficiently small amplitude, and holding the event schedule fixed,

\[
x_\varepsilon(t)-x(t)=\varepsilon s_u(t)+O(\varepsilon^2).
\]

A centered numerical derivative is

\[
D_h(t)=\frac{x_{+h}(t)-x_{-h}(t)}{2h}.
\]

For a smooth parameter dependence, the centered derivative has truncation error \(O(h^2)\) in \(h\) before floating-point roundoff dominates. The nonlinear dynamics and long-time perturbation amplification can make large \(h\) unsuitable.

The repository evaluates \(h\in\{0.08,0.04,0.02,0.01\}\). The experiment compares RK4 tangent integration with RK4 finite-difference trajectories on the **same grid and same random control sequence**. Their discrepancy measures *derivative consistency for the discretized solver*, not the absolute error of the continuous-time solution. A separate time-step refinement experiment is needed to bound that error.

Do not use a single finite difference at \(h=0.08\) as mathematical ground truth for a derivative at \(h=0\).

## 5. Stability and the generator Jacobian

The state Jacobian is

\[
J(t)=\frac{\partial g}{\partial x}=a(t)-3x(t)^2.
\]

For the **unforced autonomous** system \(\dot x=ax-x^3\), equilibria solve \(x(a-x^2)=0\). The origin is locally asymptotically stable when \(a<0\), unstable when \(a>0\); the nonzero equilibria \(x=\pm\sqrt{a}\) (when \(a>0\)) have eigenvalue \(a-3a=-2a<0\). The sign change at \(a=0\) is a pitchfork bifurcation in that idealized unforced model.

The current experiment contains a nonzero random control and an abrupt parameter change. It does **not** demonstrate a formal statistical-physics phase transition, automatic change-point detection, or critical slowing down. Its known switching time is provided by the simulator. A later study of gradual forcing through a critical regime would be a distinct experiment.

## 6. Forecast response is not generator response

Let \(S_t=(x_{t-L},\ldots,x_{t-1})\) denote a window of *sampled* observations. For a frozen one-step forecasting function \(F\),

\[
\Delta_G(t)=x^\varepsilon_t-x_t,\qquad
\Delta_F(t)=F(S_t^\varepsilon)-F(S_t).
\]

These have the same physical units but answer different questions. We report \(\mathbb E_t |\Delta_G|\), \(\mathbb E_t|\Delta_F|\), and **response mismatch**

\[
M=\mathbb E_t\big|\Delta_F(t)-\Delta_G(t)\big|.
\]

The mismatch is a *response-tracking* statistic, not a universal intrinsic property of \(F\), and its value depends on intervention, time window and available inputs.

The initial linear forecaster uses \(F(S)=w_0+\sum_{j=1}^{L}w_j S_j\), so its local response derivative is explicitly

\[
\frac{\partial F(S_t^\varepsilon)}{\partial\varepsilon}\bigg|_0
=\sum_{j=1}^L w_j\,s_u(t-L+j-1).
\]

The generator derivative for the target is \(s_u(t)\). This gives a precise *differential response mismatch* for follow-up work. A forecaster without the future control signal cannot, in general, predict an unpredictable immediate external intervention. Evaluate information availability before interpreting any mismatch.

## 7. Numerical implementation details

The solver appends \(x,s_u,s_a\) into a three-dimensional vector and integrates their coupled right-hand sides with classical RK4. The same fixed random realization is generated in [controlled_ode_framework.py](../code/controlled_ode_framework.py). Outputs include an intermediate Jacobian series as a diagnostic; it is **not** a verified Lyapunov exponent.

The unit tests verify baseline reconstruction, pre-onset zero derivative, centered finite-difference agreement, expected refinement behavior, and error handling. See [run instructions](REPRODUCIBILITY.md).

## 8. Scope of the result

The variational equations are standard results of sensitivity analysis; deriving or implementing them is **not a novel theorem**. They give a validated reference against which a future claim about model sensitivity can be falsified. A genuine contribution would require a new testable model–system relationship, formal conditions or compelling evidence against established baselines.

### Mathematical sources

- Stoer & Bulirsch, *Introduction to Numerical Analysis*, section 7.2.19, “Sensitivity Analysis of Initial-Value Problems”: https://www.math.kent.edu/~reichel/courses/numerical.analysis.2/Numerical-book.pdf
- Kuehn, “A mathematical framework for critical transitions: bifurcations, fast–slow systems and stochastic dynamics”: https://www.sciencedirect.com/science/article/pii/S0167278911000443

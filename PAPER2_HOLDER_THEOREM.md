# Paper 2 Draft: Hölder Path Geometry and Horizon Laws

This note records the cleanest theorem-level version of the second paper
direction currently supported by the project.

## 1. Model

We consider a deterministic mean path $(\mu_t)_{t\ge 1}$ observed through
Gaussian noise:

\[
  X_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t \stackrel{i.i.d.}{\sim} \mathcal N(0,\sigma^2).
\]

The goal is to estimate the endpoint $\mu_T$ using only the last $n$ samples.

### 1.1 Hölder path class

Fix $H \in (0,1]$ and $\zeta > 0$. Define the class

\[
  \mathcal H(H,\zeta,T)
  =
  \Bigl\{\mu_{1:T} :
  |\mu_t - \mu_s| \le \zeta |t-s|^H \text{ for all } 1 \le s,t \le T\Bigr\}.
\]

This is the deterministic path class used in the theorem. The parameter $H$
controls temporal roughness; $H=1$ is the smooth / ramp-like case, and $H=1/2$
is the diffusive case that matches the square-root regime seen in the empirical
work.

## 2. Upper bound

Let $\widehat\mu_T^{(n)}$ be the uniform-window estimator based on the last $n$
observations:

\[
  \widehat\mu_T^{(n)} = \frac{1}{n}\sum_{j=0}^{n-1} X_{T-j}.
\]

Assuming the same finite-sample carrier term as in Paper 1,

\[
  \mathbb E\bigl[|\widehat\mu_T^{(n)} - \bar\mu_T^{(n)}|\bigr]
  \le C_K n^{-1/2},
\]

where $\bar\mu_T^{(n)} = n^{-1}\sum_{j=0}^{n-1}\mu_{T-j}$, the total error
satisfies

\[
  \mathbb E\bigl[|\widehat\mu_T^{(n)} - \mu_T|\bigr]
  \le C_K n^{-1/2} + c_H \zeta n^H,
\]

for a path-dependent constant $c_H$.

### Optimal horizon

Balancing the two terms yields

\[
  n^*(H,\zeta)
  \asymp
  \zeta^{-\frac{2}{1+2H}},
  \qquad
  E_{\min}(H,\zeta)
  \asymp
  \zeta^{\frac{1}{1+2H}}.
\]

More precisely, if the staleness term is written as $c_H\zeta n^H$, then

\[
  n^*(H,\zeta)
  =
  \Bigl(\frac{C_K}{2H c_H \zeta}\Bigr)^{\!\frac{2}{1+2H}},
\]

and the corresponding minimum obeys

\[
  E_{\min}(H,\zeta)
  =
  \frac{1+2H}{2H}
  \Bigl(2H c_H\Bigr)^{\!\frac{1}{1+2H}}
  C_K^{\frac{2H}{1+2H}}
  \zeta^{\frac{1}{1+2H}}.
\]

## 3. Lower bound

The lower bound should be stated as a Le Cam two-point argument over a pair of
localized deterministic Hölder paths.

### 3.1 Hypothesis class

Fix a horizon length $h$ and a window end point $T$. Let $\psi:[0,1]\to[0,1]$
be a fixed nondecreasing function such that $\psi(0)=0$, $\psi(1)=1$, and
\(|\psi(u)-\psi(v)|\le |u-v|^H\) for all $u,v\in[0,1]$. Define two mean paths
on the last $h$ time points by

\[
  \mu_t^{\pm}
  =
  \pm a\,\psi\!\left(\frac{t-(T-h)}{h}\right),
  \qquad t=T-h+1,\dots,T,
\]

and extend them arbitrarily outside the window in a way that preserves the same
Hölder-$H$ seminorm. Because the scaling is local, the Hölder constraint is
satisfied whenever

\[
  a \le c_0\,\zeta h^H
\]

for a fixed geometric constant $c_0$ determined by $\psi$.

### 3.2 KL control

Under Gaussian observation noise,

\[
  \mathrm{KL}(P_+\,\|\,P_-)
  =
  \frac{1}{2\sigma^2}\sum_{t=T-h+1}^{T}(\mu_t^+ - \mu_t^-)^2
  \le
  C_\psi\,\frac{a^2}{\sigma^2}\,h,
\]

for a constant $C_\psi$ depending only on the bump shape. Choosing

\[
  a = c_0\,\zeta h^H
\]

keeps the paths in the Hölder ball. Then the KL bound becomes

\[
  \mathrm{KL}(P_+\,\|\,P_-)
  \le
  C_\psi c_0^2\,\frac{\zeta^2}{\sigma^2}\,h^{2H+1}.
\]

Choosing

\[
  h \asymp \left(\frac{\sigma}{\zeta}\right)^{\frac{2}{1+2H}}
\]

keeps the KL divergence bounded by a constant.

### 3.3 Endpoint separation

The endpoint gap is

\[
  \Delta_h = |\mu_T^+ - \mu_T^-| = 2a.
\]

With that choice, the endpoint gap is exactly the Hölder scale:

\[
  \Delta_h = 2a \asymp \zeta h^H.
\]

### 3.4 Choice of horizon

The two competing scales are therefore

\[
  \text{statistical floor} \asymp \sigma h^{-1/2},
  \qquad
  \text{staleness floor} \asymp \zeta h^H.
\]

Balancing them gives

\[
  h^*(H,\zeta) \asymp \left(\frac{\sigma}{\zeta}\right)^{\frac{2}{1+2H}}.
\]

Substituting back yields the minimax rate

\[
  E_{\min}(H,\zeta)
  \asymp
  \sigma^{\frac{2H}{1+2H}}\zeta^{\frac{1}{1+2H}}.
\]

### 3.5 Le Cam consequence

If the KL is bounded by a constant, then total variation is bounded away from 1,
and Le Cam gives

\[
  \inf_{\widehat\mu_T}
  \sup_{\mu\in\mathcal H(H,\zeta,T)}
  \mathbb E|\widehat\mu_T - \mu_T|
  \ge c\,\sigma^{\frac{2H}{1+2H}}\zeta^{\frac{1}{1+2H}}
\]

for a universal constant $c>0$ depending only on the bump shape.

This matches the upper-bound exponent.

## 4. Corollaries

### 4.1 Cube-root law

At \(H=1\),

\[
  n^*(1,\zeta) \asymp \zeta^{-2/3},
  \qquad
  E_{\min}(1,\zeta) \asymp \zeta^{1/3}.
\]

This is the ramp-like regime already proved in Paper 1.

### 4.2 Square-root law

At \(H=1/2\),

\[
  n^*(1/2,\zeta) \asymp \zeta^{-1},
  \qquad
  E_{\min}(1/2,\zeta) \asymp \zeta^{1/2}.
\]

This is the diffusive regime suggested by the simulations.

### 4.3 Anti-persistent and intermediate regimes

For \(H<1/2\), the horizon exponent becomes larger than one and the optimal
memory horizon grows faster as drift weakens. For \(H \in (1/2,1)\), the law
interpolates smoothly between square-root and cube-root.

## 5. What remains to prove

1. A fully polished minimax lower bound over a precise Hölder-ball path class.
2. A clean constant characterization for \(c_H\).
3. A formal statement of the estimator class being lower-bounded.
4. An online estimator for \(H\) if this becomes a controller instead of just a
   theorem.

## 6. Relation to Paper 1

Paper 1 corresponds to \(H=1\) and the worst-case linear-staleness regime.
This note records the broader family that appears once the path geometry is made
explicit.

## 7. References to use later

- Le Cam's two-point method for lower bounds.
- Tsybakov's nonparametric estimation framework for Hölder classes.
- Standard bias--variance trade-offs for pointwise estimation.

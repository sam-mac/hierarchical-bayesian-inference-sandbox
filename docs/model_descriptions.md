# Bayesian Models for OU-Level Productivity and Surveys
*A mathematical specification of the baseline and upgraded models*

> This document presents two closely related hierarchical Bayesian models for linking monthly OU-level productivity to intermittently observed survey scores:
> 1) a **Baseline model** (your original PyMC implementation); and  
> 2) an **Upgraded model** adding Student-t robustness, AR(1) temporal structure, an explicit missingness mechanism, and a site-level hierarchy.  
> All notation is explicitly defined; equations use a Bayesian “generative process” style and \(\LaTeX\) math.

---

## 0. Notation and Data

**Indices**
- \(g \in \{1,\dots,G\}\): operating unit (OU).
- \(s \in \{1,\dots,S\}\): site (each OU belongs to one site \(s(g)\)).
- \(t \in \{1,\dots,T\}\): month.
- \(i \in \{1,\dots,N\}\): row index for the long monthly panel; each \(i\) maps to \((g(i), t(i))\).

**Observed quantities**
- \(y^{\text{raw}}_{i} > 0\): OU-level productivity (monthly).
- \(x^{\text{raw}}_{i}\): OU-level survey score (observed only at “wave” months).
- \(n_i \ge 0\): number of survey respondents (0 if no survey).
- \(R_i \in \{0,1\}\): survey presence indicator (1 if a survey is observed this month, 0 otherwise).
- \(W_i \in \{0,1\}\): **scheduled** indicator (1 if month \(t(i)\) is a planned survey wave for OU \(g(i)\), 0 otherwise). Typically \(W_i\) is known from design (e.g., months 6 and 12).

**Standardization (computed once from data)**
\[
y_i = \log y^{\text{raw}}_{i}, 
\qquad 
y'_i=\frac{y_i - m_y}{s_y},
\qquad
x'_i=\frac{x^{\text{raw}}_i - m_x}{s_x},
\]
where \(m_y,s_y\) are the empirical mean/SD of \(\log y^{\text{raw}}\), and \(m_x,s_x\) are the mean/SD of observed \(x^{\text{raw}}\).  
Define the sets \(O=\{i: R_i=1\}\) (survey observed) and \(D=\{i: W_i=1\}\) (scheduled).

**Latent variable**
- \(x^*_i\): **latent** (error-free) standardized survey construct for row \(i\).

---

## 1. Baseline Model (Original Implementation)

### 1.1 OU-level varying intercepts and slopes (partial pooling via LKJ)

For each OU, we model an intercept \(\alpha_g\) and a slope \(\beta_g\) that relate \(x^*\) to \(y'\). We pool these across OUs with a correlated multivariate normal prior:
\[
\begin{bmatrix}\alpha_g\\ \beta_g\end{bmatrix}
\sim
\mathcal{N}_2\!\Big(
\begin{bmatrix}\mu_\alpha\\ \mu_\beta\end{bmatrix},\;
\Sigma
\Big),
\qquad
\Sigma
=\operatorname{diag}(\sigma_\alpha,\sigma_\beta)\,R\,\operatorname{diag}(\sigma_\alpha,\sigma_\beta),
\]
\[
\mu_\alpha \sim \mathcal{N}(0,1),\quad
\mu_\beta \sim \mathcal{N}(0,0.5^2),\quad
(\sigma_\alpha,\sigma_\beta) \sim \mathrm{HalfNormal}(1),\quad
R \sim \mathrm{LKJ}(\eta=2).
\]

**LKJ–Cholesky implementation.** In PyMC, we draw a Cholesky factor \(L\) such that \(LL^\top=\Sigma\) and transform standard normals. With matrix \(Z\in\mathbb{R}^{G\times 2}\) (rows \(z_g\sim \mathcal{N}(0,I_2)\)),
\[
\begin{bmatrix}a_g & b_g\end{bmatrix} = z_g L^\top,
\qquad
\alpha_g = \mu_\alpha + a_g,
\quad
\beta_g  = \mu_\beta  + b_g.
\]
The transpose \(L^\top\) is required because \(z_g\) is a **row** vector. (If you used column vectors you’d use \(L z_g\) instead.)

> **Why \(n=2\) in LKJ?** Because the correlated random-effects vector is 2-dimensional: \((\alpha,\beta)\). If you add a third varying coefficient (e.g., a time-trend slope), use \(n=3\).

### 1.2 Time effects (centered)

We include **month effects** for the outcome and for the latent survey process, centered to have mean zero so that intercepts remain interpretable:
\[
\lambda_t \stackrel{iid}{\sim}\mathcal{N}(0,1),\quad \sum_t \lambda_t = 0;
\qquad
\tau_t \stackrel{iid}{\sim}\mathcal{N}(0,1),\quad \sum_t \tau_t = 0.
\]

### 1.3 Latent survey process and measurement model

Latent process (z-scale):
\[
x^*_i \sim \mathcal{N}\!\big(\mu_{x,g(i)} + \gamma_x\,\tau_{t(i)},\ \sigma_x^2\big),
\qquad
\mu_{x,g} \sim \mathcal{N}(\mu_x,\sigma_{\mu_x}^2),
\]
\[
\mu_x \sim \mathcal{N}(0,1),\quad \sigma_{\mu_x}\sim\mathrm{HalfNormal}(1),\quad
\sigma_x\sim\mathrm{HalfNormal}(1),\quad
\gamma_x=0.2\ \text{(fixed in the baseline)}.
\]

Measurement model (only where observed, \(i\in O\)):
\[
x'_i \mid x^*_i, n_i, \sigma_{\text{meas}}
\sim
\mathcal{N}\!\Big(x^*_i,\ \frac{\sigma_{\text{meas}}^2}{\max(\sqrt{n_i},1)^2}\Big),
\qquad \sigma_{\text{meas}}\sim \mathrm{HalfNormal}(1).
\]

> The variance shrinks like \(1/n_i\): a larger respondent count yields a tighter link between \(x^*\) and the observed survey.

### 1.4 Outcome model (standardized log-productivity)

\[
y'_i \mid \alpha_{g(i)},\beta_{g(i)},x^*_i,\lambda_{t(i)},\sigma_y
\sim
\mathcal{N}\!\big(\alpha_{g(i)} + \lambda_{t(i)} + \beta_{g(i)}\,x^*_i,\ \sigma_y^2\big),
\quad \sigma_y \sim \mathrm{HalfNormal}(0.5).
\]

### 1.5 Joint posterior

Let \(\theta\) collect all parameters and hyperparameters. The posterior is
\[
p(\theta, x^*\mid \text{data})
\propto
\Big[\prod_{i=1}^N \mathcal{N}\!\big(y'_i \mid \alpha_{g(i)} + \lambda_{t(i)} + \beta_{g(i)}x^*_i,\ \sigma_y^2\big)\Big]
\Big[\prod_{i\in O} \mathcal{N}\!\big(x'_i \mid x^*_i,\ \sigma_{\text{meas}}^2/n_i^\dagger\big)\Big]
\,p(\theta)\,p(x^*\mid\theta),
\]
with \(n_i^\dagger=\max(\sqrt{n_i},1)^2\) and priors as above.

### 1.6 Interpretation (raw scale)

For OU \(g\), a +1 point change in **raw** survey score corresponds to an approximate multiplicative change in **raw** productivity:
\[
\Delta_g \equiv \exp\!\Big(\tfrac{s_y}{s_x}\,\beta_g\Big),
\qquad
\%\text{ change} = 100(\Delta_g-1).
\]

---

## 2. Upgraded Model (Robust, Dynamic, Hierarchical, Missingness-Aware)

The upgraded model alters and extends the baseline as follows.

### 2.1 Two-level hierarchy for \((\alpha,\beta)\)

Add **site-level** correlated effects and retain OU-level residuals:
\[
\begin{aligned}
\begin{bmatrix}\alpha^{(site)}_s\\ \beta^{(site)}_s\end{bmatrix}
&\sim \mathcal{N}_2\!\Big(\begin{bmatrix}0\\0\end{bmatrix},\ \Sigma_{\text{site}}\Big),
&& \Sigma_{\text{site}}=\operatorname{diag}(\sigma^\text{site}_\alpha,\sigma^\text{site}_\beta)\,R_\text{site}\,\operatorname{diag}(\cdot),\\
\begin{bmatrix}\alpha^{(ou)}_g\\ \beta^{(ou)}_g\end{bmatrix}
&\sim \mathcal{N}_2\!\Big(\begin{bmatrix}0\\0\end{bmatrix},\ \Sigma_{\text{ou}}\Big),
&& \Sigma_{\text{ou}}=\operatorname{diag}(\sigma^\text{ou}_\alpha,\sigma^\text{ou}_\beta)\,R_\text{ou}\,\operatorname{diag}(\cdot),\\
\alpha_g &= \mu_\alpha + \alpha^{(site)}_{s(g)} + \alpha^{(ou)}_g,
\qquad
\beta_g  = \mu_\beta  + \beta^{(site)}_{s(g)} + \beta^{(ou)}_g,
\end{aligned}
\]
with LKJ priors on \(R_\text{site}\) and \(R_\text{ou}\) (e.g., \(\eta=2\)) and Half-Normal or Half-Student-t priors on the scales.

### 2.2 AR(1) temporal structure (centered)

Replace i.i.d. month effects by mean-zero **AR(1)** processes:
\[
\lambda_t \sim \text{AR(1)}(\phi_\lambda,\sigma_\lambda),\quad
\tau_t \sim \text{AR(1)}(\phi_\tau,\sigma_\tau),
\qquad
\sum_t \lambda_t=0,\ \sum_t \tau_t=0,
\]
with \(|\phi_{\lambda,\tau}|<1\) and \(\sigma_{\lambda,\tau}>0\). This induces temporal correlation for outcome and latent survey processes and improves interpolation between waves.

### 2.3 Latent survey process with learned time magnitude

\[
x^*_i \sim \mathcal{N}\!\big(\mu_{x,g(i)} + \gamma_x\,\tau_{t(i)},\ \sigma_x^2\big),
\qquad
\gamma_x \sim \mathcal{N}(0,0.3^2),
\]
with \(\mu_{x,g}\sim \mathcal{N}(\mu_x,\sigma_{\mu_x}^2)\) and suitably chosen weakly-informative scales (optionally “empirical-Bayes”-flavored).

### 2.4 Measurement model (unchanged)

\[
x'_i \mid x^*_i, n_i, \sigma_{\text{meas}}
\sim
\mathcal{N}\!\Big(x^*_i,\ \frac{\sigma_{\text{meas}}^2}{\max(\sqrt{n_i},1)^2}\Big),
\qquad i\in O.
\]

### 2.5 Missingness mechanism (design + MNAR on scheduled months)

- **Design:** \(W_i\) indicates months eligible for a survey (e.g., every 6 months).  
- **Response model on scheduled months:** if additional nonresponse depends on the (unobserved) latent construct, model it as:
\[
R_i \;\big|\; W_i=1 \;\sim\; \mathrm{Bernoulli}(p_i),
\qquad
\mathrm{logit}\,p_i = \rho_0 + \rho_x\,x^*_i + \rho^{(ou)}_{g(i)} + \rho^{(time)}_{t(i)}.
\]
Priors: \(\rho_0,\rho_x\sim \mathcal{N}(0,1)\); \(\rho^{(ou)}\) and \(\rho^{(time)}\) mean-zero random effects with Half-Normal scales.  
This couples the probability a survey is observed to the latent state \(x^*\) (MNAR). Only **scheduled** rows \(i\in D\) enter this likelihood.

### 2.6 Robust outcome likelihood

Replace Gaussian noise with **Student-t**:
\[
y'_i \mid \cdots \sim \mathrm{StudentT}\big(\nu_y,\ \mu_i,\ \sigma_y\big),
\qquad
\mu_i=\alpha_{g(i)} + \lambda_{t(i)} + \beta_{g(i)}x^*_i,
\]
with \(\nu_y > 2\) (e.g., \(\nu_y=2+\mathrm{Exp}(1/15)\)) and \(\sigma_y \sim \mathrm{HalfNormal}(0.5)\).  
The heavy tails reduce the leverage of outliers/misspecification.

### 2.7 Posterior

\[
\begin{aligned}
p(\theta, x^* \mid \text{data})
\propto\;
&\underbrace{\prod_{i=1}^N \mathrm{StudentT}\big(y'_i \mid \nu_y,\ \alpha_{g(i)}+\lambda_{t(i)}+\beta_{g(i)}x^*_i,\ \sigma_y\big)}_{\text{outcome}}\\
&\times\underbrace{\prod_{i\in O} \mathcal{N}\big(x'_i \mid x^*_i,\ \sigma_{\text{meas}}^2/n_i^\dagger\big)}_{\text{measurement}}\\
&\times\underbrace{\prod_{i\in D}\mathrm{Bernoulli}\big(R_i \mid \mathrm{logit}^{-1}(\rho_0+\rho_x x^*_i+\rho^{(ou)}_{g(i)}+\rho^{(time)}_{t(i)})\big)}_{\text{missingness}}
\; p(\theta)\;p(x^*\mid\theta).
\end{aligned}
\]

---

## 3. LKJ–Cholesky Prior: Intuition and Practicalities

- **Purpose.** It produces a proper prior over **correlation matrices** \(R\) with a single concentration parameter \(\eta\). Larger \(\eta\) shrinks \(R\) toward the identity (independence); \(\eta=1\) is uniform over valid correlations; \(\eta>1\) favors weaker correlations.
- **Dimension.** The `n` argument equals the **dimension of the random-effects vector** you want to correlate. Use \(n=2\) for \((\alpha,\beta)\), \(n=3\) if you add another varying slope, etc.
- **Cholesky transform.** Working with the Cholesky factor \(L\) improves sampling geometry. You generate standard normals \(z\) and map them through \(L^\top\) (for row-vector layout) so that \(\mathrm{Cov}(z L^\top)=LL^\top\).

---

## 4. Inference and “Truth”

Both models are fit with **NUTS** (No-U-Turn Sampler), a self-tuning Hamiltonian Monte Carlo method. HMC simulates a Markov chain whose invariant distribution is the **exact posterior** under the model; approximation error is purely Monte Carlo and vanishes as draws \(\to\infty\) (up to floating-point and integrator step-size error). This is categorically different from variational approximations, which target a surrogate family.

**Diagnostics** to assert correctness:
- \(\widehat{R}\approx 1.00\), large ESS (bulk and tail), **0 divergences**, reasonable BFMI/energy, stable trace plots.

---

## 5. Missing Data Behavior and “Posterior Collapse”

- **Unobserved months (\(i\notin O\)).** There is no measurement term; \(x^*_i\) is informed by its hierarchical/time priors **and** by the outcome model through \(\beta_{g(i)}\) and \(y'_i\).
- **Observed months (\(i\in O\)).** The posterior for \(x^*_i\) tightly concentrates (“collapses”) around \(x'_i\) with variance \(\sigma_{\text{meas}}^2/n_i^\dagger\). Larger \(n_i\) yields a stronger collapse.
- **Scheduled MNAR (upgraded model).** On scheduled months, the probability of observation depends on \(x^*\) via \(\rho_x\). This corrects potential selection bias if missingness is related to the latent state.

---

## 6. Assumptions (Explicit)

- **Linearity (on z-log scale).** \(y'\) varies linearly with \(x^*\).
- **Gaussian survey noise;** Student-t outcome (upgraded).
- **Exchangeable groups** with (possibly) correlated random effects via LKJ.
- **Temporal structure:** i.i.d. (baseline) or AR(1) (upgraded), centered to mean zero.
- **Design-based scheduling** and (optionally) MNAR nonresponse on scheduled months.
- **Standardization** constants \((m_x,s_x,m_y,s_y)\) are treated as fixed.

---

## 7. Model Evaluation and Interpretation

- **Posterior predictive checks (PPC):** Compare replicates \(y'^{\text{rep}}\) to observed \(y'\) (global, over time, by OU). Student-t should improve tail fit.
- **Survey calibration checks:** At observed months, predictive intervals for \(x'\) and posterior intervals for \(x^*\) should align with respondent counts \(n_i\).
- **Information criteria:** PSIS-LOO/WAIC for model comparison (watch \(p_{\text{eff}}\) and Pareto-\(k\)).
- **Substantive summaries:** OU/site forests for \(\beta\), correlation between \(\alpha\) and \(\beta\), and raw-scale effect \(100(\exp((s_y/s_x)\beta_g)-1)\%\).

---

## 8. Overfitting Control and Adding Covariates

- **Partial pooling** (hierarchical priors) reduces effective parameters.
- **LKJ concentration** (\(\eta>2\)) and **Half-t** priors on random-effect scales stabilize covariance learning.
- **Shrinkage for many fixed effects:** ridge (Normal with small SD), ARD (per-coefficient scales), or **regularized horseshoe** for sparse signals.
- **Student-t likelihood** adds robustness to outliers but does not by itself reduce dimensionality.
- Track **\(p_{\text{eff}}\)** (from LOO) as you add structure.

---

## 9. Generative Recipes (Summary)

### Baseline
1. Draw hyperparameters \(\mu_\alpha,\mu_\beta, \sigma_\alpha,\sigma_\beta, R\); obtain \(L\).
2. For each OU, draw \(z_g\sim\mathcal{N}(0,I_2)\), set \((\alpha_g,\beta_g)= (\mu_\alpha,\mu_\beta)+ z_g L^\top\).
3. Draw centered time effects \(\lambda_t,\tau_t\stackrel{iid}{\sim}\mathcal{N}(0,1)\) and center to mean zero.
4. Draw \(\mu_x,\sigma_{\mu_x},\sigma_x,\sigma_{\text{meas}},\sigma_y\).
5. For each \(i\): draw \(x^*_i\sim\mathcal{N}(\mu_{x,g(i)}+0.2\,\tau_{t(i)},\sigma_x^2)\).
6. If \(i\in O\): draw \(x'_i\sim\mathcal{N}(x^*_i,\sigma_{\text{meas}}^2/n_i^\dagger)\).
7. Draw \(y'_i\sim\mathcal{N}(\alpha_{g(i)}+\lambda_{t(i)}+\beta_{g(i)}x^*_i,\sigma_y^2)\).

### Upgraded
Same, except:
- Two-level LKJ hierarchy for \((\alpha,\beta)\).
- AR(1) \(\lambda_t,\tau_t\) (centered).
- \(\gamma_x\) learned.
- Student-t outcome \(y'\).
- On scheduled months \(i\in D\): draw \(R_i\sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\rho_0+\rho_x x^*_i+\rho^{(ou)}_{g(i)}+\rho^{(time)}_{t(i)}))\); only if \(R_i=1\) do you observe \(x'_i\).

---

## 10. Practical Remarks

- **LKJ with more coefficients.** If you add a third OU-level varying slope, use LKJ with \(n=3\). Guard against overfitting with larger \(\eta\) (e.g., 4), Half-t scales, or **independent** random effects if \(G\) is small relative to \(n\).
- **Transpose in code.** With `z.shape==(G,n)` (rows are groups), use `ab = z @ L.T` to obtain covariance \(LL^\top\).
- **Centering time effects.** Enforces \(\sum_t \lambda_t=0\) and \(\sum_t \tau_t=0\); this prevents ambiguity between average levels and time means and keeps \(\alpha\) interpretable.
- **When to use MNAR.** Only if you have reason to believe that **conditional** on design and modeled covariates, missingness still depends on the latent construct.

---

*References-in-spirit (conceptual):*  
LKJ priors (Lewandowski–Kurowicka–Joe), Cholesky parameterization for hierarchical covariance; robust regression via Student-t; AR(1) and state-space processes for time-correlated latent effects; PSIS-LOO and \(p_{\text{eff}}\) for Bayesian model comparison.
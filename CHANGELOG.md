# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Severity (Klugman 5e Appendix A.2):** `Exponential`, `Pareto` (Type II /
  Lomax), `Lognormal`, `Weibull`, `Gamma`, `BurrXII`, `LogLogistic`,
  `Paralogistic`, `InverseParalogistic`, `InverseGaussian` (Wald),
  `TransformedGamma`, `InverseTransformedGamma`, `TransformedBeta`
  (4-parameter parent of the Burr family). Every severity ships with
  closed-form pdf, cdf, ppf, rvs, mean, and limited expected value;
  every closed-form LEV is validated against `scipy.integrate.quad`.
- **Frequency (Klugman 5e Chapter 6):** `Poisson`, `Binomial`,
  `NegativeBinomial`, `Geometric`, `ZIP` (Zero-Inflated Poisson),
  `ZINB` (Zero-Inflated Negative Binomial).
- **Goodness-of-fit:** Kolmogorov-Smirnov and upper-tail emphasized
  Anderson-Darling tests with parametric-bootstrap p-values
  (Lilliefors-correct), chi-squared with equiprobable bins and
  parameter-aware df adjustment, PP- and QQ-plot helpers.
- **Fitting:** `DistributionFitter.fit_and_rank` resolves named
  candidates against the registries, fits each via MLE, and returns
  rows with loglik / k / AIC / BIC / params sorted by the chosen
  criterion. Failed fits surface their exception in the `error` field
  instead of crashing the run.
- **Profile-likelihood confidence intervals:** `profile_likelihood_ci`
  on every distribution. Brackets each boundary geometrically in the
  natural parameter space (log / logit-aware) and refines via Brent.
- **Layer statistics tests:** `excess_pure_premium`,
  `increased_limits_factor`, `layer_expected_value` are now backed by
  numerical tests against closed-form expectations.

## [0.0.1] - 2026-04-26

### Added

- Initial project scaffold.
- Base `ActuarialDistribution` class with standard interface (pdf, cdf, ppf,
  rvs, mle_fit, loglik, aic, bic, limited_expected_value,
  excess_pure_premium).
- `BurrXII` distribution stub (3-parameter Burr Type XII).
- `GeneralizedPareto` distribution stub (GPD for extreme value modeling).
- `TransformedBeta` distribution stub (4-parameter transformed beta family).
- `ZeroInflatedPoisson` distribution stub (ZIP for frequency with excess zeros).
- `DistributionFitter` class for multi-candidate fitting and AIC/BIC ranking.
- `GoodnessOfFit` class for KS, Anderson-Darling, Chi-squared tests, and
  PP/QQ plots.
- Smoke tests for imports and class instantiation.

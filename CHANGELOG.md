# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-05-01

### Added

- Severity distributions (`actudist.severity`): Exponential, Pareto,
  Lognormal, Weibull, Gamma, Burr Type XII, Log-Logistic, Paralogistic,
  Inverse Paralogistic, Inverse Gaussian, Transformed Gamma, Inverse
  Transformed Gamma, Transformed Beta. Each exposes pdf, cdf, ppf, rvs,
  closed-form LEV where one exists, excess pure premium, and ILF.
- Frequency distributions (`actudist.frequency`): Poisson, Binomial,
  Geometric, Negative Binomial, Zero-Inflated Poisson, Zero-Inflated
  Negative Binomial.
- `DistributionFitter.fit_and_rank` returns AIC/BIC rankings across
  registered candidates, with per-candidate fit-failure rows pushed to
  the bottom.
- `GoodnessOfFit`: KS and Anderson-Darling with parametric-bootstrap
  p-values (Lilliefors-corrected), chi-squared with equiprobable bins,
  PP and QQ plots.
- Profile-likelihood CIs via `ActuarialDistribution.profile_likelihood_ci`.
- Layer helpers in `actudist.layers`: excess pure premium, ILF, finite
  layer expected value.

### Changed

- Package layout split from a single `core` module into per-distribution
  modules under `actudist.severity` and `actudist.frequency`.
- MLE driver now optimizes in unconstrained space via log/logit
  reparameterization with a Nelder-Mead fallback.

## [0.0.1] - 2026-04-26

### Added

- Initial project scaffold and packaging metadata.

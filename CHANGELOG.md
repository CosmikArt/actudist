# Changelog

All notable changes to this project will be documented in this file.

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

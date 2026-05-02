[![PyPI version](https://img.shields.io/pypi/v/actudist?color=blue)](https://pypi.org/project/actudist/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# actudist

Python library for P&C loss distributions. Severity and frequency families
follow the parameterizations in Klugman, *Loss Models*. Every severity
distribution exposes layer statistics (LEV, excess pure premium, ILF) as
methods. MLE accepts right-censored and one- or two-sided truncated samples.

## Installation

```bash
pip install actudist
```

From source:

```bash
git clone https://github.com/CosmikArt/actudist.git
cd actudist
pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np
from actudist import DistributionFitter, GoodnessOfFit
from actudist.severity.lognormal import Lognormal

rng = np.random.default_rng(42)
claims = Lognormal(mu=8.0, sigma=1.2).rvs(size=2000, random_state=rng)

fit = Lognormal()
fit.mle_fit(claims)
print(f"mu={fit.mu:.3f}  sigma={fit.sigma:.3f}")
print(f"AIC={fit.aic(claims):.1f}  BIC={fit.bic(claims):.1f}")
print(f"LEV(50000)={fit.limited_expected_value(50_000):.1f}")
print(f"EPP(50000)={fit.excess_pure_premium(50_000):.1f}")

fitter = DistributionFitter(["Lognormal", "Pareto", "Weibull"])
ranking = fitter.fit_and_rank(claims, criterion="aic")
for row in ranking:
    print(f"{row['name']:12s}  aic={row['aic']:.1f}")

gof = GoodnessOfFit(fit, claims)
result = gof.ks_test(n_boot=200, random_state=42)
print(f"KS stat={result['statistic']:.4f}  p={result['p_value']:.3f}")
```

## Modules

- `actudist.severity`: 13 continuous distributions (Exponential, Pareto,
  Lognormal, Weibull, Gamma, Burr XII, Log-Logistic, Paralogistic and its
  inverse, Inverse Gaussian, Transformed and Inverse Transformed Gamma,
  Transformed Beta).
- `actudist.frequency`: Poisson, Binomial, Geometric, Negative Binomial,
  ZIP, ZINB.
- `actudist.fitting`: registries and `DistributionFitter` for AIC/BIC
  ranking.
- `actudist.gof`: KS and Anderson-Darling with parametric-bootstrap
  p-values, chi-squared, PP and QQ plots.
- `actudist.layers`: excess pure premium, ILF, finite-layer expected loss.

`profile_likelihood_ci` lives on the base class; call it on any fitted
instance.

## References

- Klugman, S. A., Panjer, H. H., & Willmot, G. E. *Loss Models: From Data to
  Decisions* (5th ed.). Wiley.
- Hogg, R. V. & Klugman, S. A. *Loss Distributions*. Wiley.
- Society of Actuaries. Exam STAM / FAM syllabus and study materials.

## Contributing

Run `pytest` before sending a PR.

## Author

Isaac López

MIT License. See [LICENSE](LICENSE).

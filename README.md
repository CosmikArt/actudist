[![PyPI version](https://img.shields.io/pypi/v/actudist?color=blue)](https://pypi.org/project/actudist/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

# actudist

**Actuarial probability distributions — heavy tails, MLE fitting, and goodness-of-fit.**

---

## What is actudist?

`scipy.stats` provides a solid foundation for statistical distributions, but it
falls short for actuarial practitioners in several important ways:

- **Missing actuarial-specific distributions.** Burr Type XII, Inverse
  Paralogistic, Transformed Beta, Transformed Gamma, and other heavy-tail
  severity models central to P&C loss modeling are absent or require awkward
  reparameterizations.
- **No zero-inflated models.** Frequency data in insurance almost always
  exhibits excess zeros. ZIP and ZINB belong in any actuarial distribution
  toolkit.
- **No actuarial fitting workflows.** Real loss data is censored by policy
  limits, truncated by deductibles, and observed through excess layers.
  `scipy.stats.fit` does not handle any of these.
- **No layer statistics.** Limited expected values, excess pure premiums, and
  increased limits factors are first-class concepts in pricing — they should be
  first-class methods on every distribution object.

`actudist` fills these gaps with a focused, opinionated library: 23+
distributions parameterized the way actuaries expect, MLE fitting that respects
censoring and truncation, profile likelihood confidence intervals, and a
complete goodness-of-fit testing suite.

---

## Installation

```bash
pip install actudist
```

From source:

```bash
git clone https://github.com/CosmikArt/actudist.git
cd actudist
pip install -e .
```

---

## Quickstart

Fit a Burr XII to claim severity data, compare against Lognormal and Pareto
via AIC, then run goodness-of-fit tests:

```python
import numpy as np
from actudist import DistributionFitter, GoodnessOfFit
from actudist.severity.burrxii import BurrXII

# Simulated claim severity data (right-skewed, heavy-tailed)
rng = np.random.default_rng(42)
claims = np.concatenate([
    rng.lognormal(mean=8.0, sigma=1.5, size=800),
    rng.pareto(a=1.5, size=200) * 50_000,
])

# Fit a Burr XII distribution via MLE
burr = BurrXII()
burr.mle_fit(claims)
print(f"Burr XII AIC: {burr.aic(claims):.1f}")

# Compare multiple candidates
fitter = DistributionFitter(candidates=["BurrXII", "Lognormal", "Pareto"])
rankings = fitter.fit_and_rank(claims, criterion="aic")
for r in rankings:
    print(f"{r['name']:>14}  AIC={r['aic']:.1f}  BIC={r['bic']:.1f}")

# Goodness-of-fit testing (parametric-bootstrap p-values)
gof = GoodnessOfFit(distribution=burr, data=claims)
print(gof.ks_test(n_boot=500))
print(gof.anderson_darling_test(n_boot=500))
gof.qq_plot()

# Profile-likelihood 95% CI for each Burr XII parameter
for p in ("alpha", "theta", "gamma"):
    lo, hi = burr.profile_likelihood_ci(claims, p)
    print(f"{p}: [{lo:.3f}, {hi:.3f}]")
```

---

## Features

| Module | Description |
|---|---|
| `severity` | 13+ heavy-tail distributions: Pareto, Lognormal, Weibull, Burr Type XII, Generalized Pareto, Inverse Gaussian, Log-Logistic, Paralogistic, Inverse Paralogistic, Transformed Beta, Transformed Gamma, and more |
| `frequency` | Poisson, Negative Binomial, Binomial, Zero-Inflated Poisson (ZIP), Zero-Inflated Negative Binomial (ZINB) |
| `compound` | Compound distributions: Poisson-Gamma, Poisson-Lognormal, NB-Pareto, and custom frequency-severity combinations |
| `fitting` | MLE with support for censored and truncated data, profile likelihood confidence intervals, censored/truncated likelihood contributions |
| `gof` | Kolmogorov-Smirnov, Anderson-Darling, Chi-squared tests, PP plots, QQ plots, AIC/BIC model comparison |
| `layers` | Excess-of-loss layer statistics: limited expected value (LEV), excess pure premium, increased limits factors |

---

## References

- Klugman, S. A., Panjer, H. H., & Willmot, G. E. *Loss Models: From Data to
  Decisions* (5th ed.). Wiley.
- Hogg, R. V. & Klugman, S. A. *Loss Distributions*. Wiley.
- Society of Actuaries. Exam STAM / FAM syllabus and study materials.

---

## Contributing

Contributions are welcome. Please open an issue first to discuss proposed
changes. All code must include type hints, docstrings, and unit tests.

```bash
git clone https://github.com/CosmikArt/actudist.git
cd actudist
pip install -e ".[dev]"
pytest
```

---

## Author

**Isaac López**

---

MIT License. See [LICENSE](LICENSE).

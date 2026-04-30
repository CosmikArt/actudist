# actudist — working agreement

These rules are non-negotiable for any session that edits this repository.

## Authorship

- All commits are authored by **Isaac López <jisaaclopg@gmail.com>**.
- **Never** add `Co-Authored-By: Claude <noreply@anthropic.com>`, `🤖 Generated with Claude Code`, or any AI-attribution footer to commit messages, PR descriptions, or source files.
- Do not change `git config user.name` or `user.email` in this repo.
- Author email in `pyproject.toml` must match the git committer email above.

## Publishing

- **Do not** publish to PyPI or TestPyPI without an explicit, in-session instruction from the maintainer. Building wheels locally for inspection is fine; uploading is not.

## Source of truth for math

- Parameterizations follow **Klugman, Panjer & Willmot, *Loss Models: From Data to Decisions* (5th ed.)**, Appendix A for severity and Chapter 6 for frequency. When `actudist` and `scipy.stats` disagree on parameter conventions, `actudist` matches Klugman.
- Every distribution's class docstring cites the Klugman section/page where its pdf, cdf, and LEV formulas come from.
- Do **not** invent closed-form formulas. If a formula (especially LEV, MGF, or anything involving the incomplete beta/gamma functions) is not directly transcribed from a primary source, validate it numerically against `scipy.integrate.quad` to a tolerance of `1e-6` *before* committing. If a Klugman formula does not match numerics, stop and ask — do not silently "adjust" coefficients.

## Test discipline

- No distribution is committed without its full numerical test suite (see `README` / project plan for the current required tests per distribution).
- `pytest` must be green before every commit. Coverage target for v0.1.0 is ≥85%.
- Smoke tests (instantiation, imports) are necessary but not sufficient.

## Numerical hygiene

- Use `scipy.special.xlog1py`, `numpy.log1p`, `scipy.special.logsumexp` whenever heavy tails or near-zero survival functions are involved.
- Reparameterize all positive-supported MLE parameters to log-space and use `scipy.optimize.minimize(method="L-BFGS-B")` with explicit bounds. Do not rely on the default optimizer.
- When a distribution lacks a closed-form `ppf`, implement a numerical fallback via `scipy.optimize.brentq` over the cdf. Do **not** leave it as `NotImplementedError`.

## Architectural conventions

- One file per distribution under `src/actudist/severity/` or `src/actudist/frequency/`.
- Each distribution registers itself via `@register_severity("Name")` or `@register_frequency("Name")` (see `fitting.py`).
- `SeverityDistribution` exposes `pdf`, `cdf`, `ppf`, `rvs`, `loglik`, `mle_fit`, `limited_expected_value`, `excess_pure_premium`, `mean`. `FrequencyDistribution` exposes `pmf` instead of `pdf` and does **not** carry layer statistics.

## Commit hygiene

- Atomic commits: one distribution, one refactor, or one feature per commit.
- Conventional-commit prefixes: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `release`.
- Push only when the maintainer explicitly asks.

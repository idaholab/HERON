# Uncertainty

What is random in a HERON case, how much to sample, and which statistics to
report or optimize. Settings are in `input/case.md`.

## Where randomness comes from

| Source | How to include it | Sampled |
|--------|-------------------|---------|
| Time series | `<ARMA>` generators (`time-series.md`) | Once per realization (full multi-year history) |
| Cash flow parameters | `<uncertainty>` with a RAVEN distribution in a `<driver>`, `<reference_price>`, `<reference_driver>`, or `<scaling_factor_x>` | Once per realization |
| Design variables | `sweep_values` / `opt_bounds` | Not random: set by the outer loop |

A `<CSV>` history and `<fixed_value>`s are deterministic.

A cash flow with an `<uncertainty>` forces the bilevel workflow (`modes.md`).

## Realizations: `<num_arma_samples>`

Each design point is evaluated with `<num_arma_samples>` realizations (default
`1`), and each realization is a complete dispatch plus cash flow evaluation.

- Runtime is proportional to the number of realizations.
- The standard error of a mean falls as 1/sqrt(N). Going from 10 to 40
  realizations halves the noise; going from 1 to 4 does the same.
- Tail statistics (a 5th percentile, value at risk, expected shortfall) need
  many more realizations than a mean. With N realizations, the 5th
  percentile rests on about 0.05 N points, so 20 realizations give roughly one.
- One realization yields a single scenario, not a distribution. Its
  standard deviation and percentiles are not meaningful.
- Start with a few realizations while building the model, and raise the
  count once results look right. In opt mode, raise it before trusting the
  optimum, because the optimizer compares noisy values (`modes.md`).

## Statistics HERON reports

For each economic metric and each total activity, the inner run computes
statistics across realizations and the outer run writes them as columns named
`<prefix>_<variable>`, or `<prefix>_<parameter>_<variable>` when the statistic
has a percent or threshold parameter.

| Statistic (`<result_statistics>` / `<stats_metric>`) | Prefix | Default direction in opt |
|------------------------------------------------------|--------|--------------------------|
| `expectedValue` | `mean` | max |
| `minimum` / `maximum` | `min` / `max` | max |
| `median` | `med` | max |
| `variance` / `sigma` | `var` / `std` | min |
| `percentile` (`percent`, default 5 and 95) | `perc` | max |
| `variationCoefficient`, `skewness`, `kurtosis` | `varCoeff`, `skew`, `kurt` | min |
| `samples` | `samp` | (none) |
| `sharpeRatio`, `sortinoRatio`, `gainLossRatio` | `sharpe`, `sortino`, `glr` | max |
| `expectedShortfall`, `valueAtRisk` (`threshold`, default 0.05) | `es`, `VaR` | min |

Examples: `mean_NPV`, `std_NPV`, `perc_5_NPV`, `VaR_0.05_NPV`,
`mean_TotalActivity__gen__production__electricity`.

Which are written by default depends on the mode:

| Mode | Statistics always written (every economic metric) | Total activity statistics |
|------|----------------------------------------------------|---------------------------|
| opt | `mean`, `std`, `med` | the same |
| sweep | `mean`, `std`, `med`, plus `max`, `min`, `perc` (5 and 95), `samp`, `var` | the same, minus the financial ones (Sharpe, Sortino, ES, VaR, GLR) |

`mean`, `std`, and `med` are always present because `Case` starts with
`expectedValue`, `sigma`, and `median` in its result statistics. Anything in
`<result_statistics>` is added on top. In opt mode the optimization objective
is always included too. The gold sweep files (for example
`tests/workshop/simple/gold/1_simple_o/sweep.csv`) show this set of columns.

## Choosing the objective in opt mode

The objective is `<stats_metric>` of `<opt_metric>` (default: expected NPV).

- `expectedValue` of NPV is risk-neutral: the best average design, whatever
  the spread.
- Use a risk measure when downside matters: `valueAtRisk` or
  `expectedShortfall` (low tail), `percentile`, or `sharpeRatio` and
  `sortinoRatio`/`gainLossRatio` (return per unit risk).
- The direction `<type>` defaults from the statistic (table above). For
  `LC`, the defaults are reversed. Check the direction when you pair an
  unusual statistic with a metric: maximizing a variance is rarely intended.
- A tail statistic as the objective is noisy; use many realizations.
- With a clustered time series model that has fewer clusters than segments,
  keep the objective at `expectedValue`. Spread and tail statistics are
  inflated by the clustering (`time-series.md`, "Clustering and higher
  moments").

## Reading the spread

- With a clustered model (fewer clusters than segments), `std_NPV` and other
  spread statistics are overestimated. Treat them as upper bounds, not
  estimates (`time-series.md`).
- A large `std_NPV` relative to `mean_NPV` means the design's economics depend
  on the scenario. Compare `perc_5_NPV` (or `min_NPV`) against `perc_95_NPV`
  (or `max_NPV`) to see by how much.
- If two designs have means within about two standard errors
  (`std / sqrt(N)`), the realizations cannot tell them apart. Add
  realizations before preferring one.
- A design whose `min_NPV` is negative can still have a positive mean. Report
  both, along with how many realizations support them.
- All realizations share the same capacities, so spread comes only from the
  time series and uncertain cash flow parameters, not from design.

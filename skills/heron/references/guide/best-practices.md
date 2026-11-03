# Best practices

A checklist for designing a study and reviewing an input. Each item points to
the file with the details.

## Build up in stages

1. **Start small.** The fewest components that make the question meaningful,
   one year of hourly data, and `<debug/>`. Add components one at a time and
   re-run debug after each.
2. **Make the dispatch physical before looking at money,** then check cash
   flow signs and magnitudes against a back-of-the-envelope estimate
   (capacity times price times hours). What to look at, in order, is in
   `references/guide/modes.md`, "Debug".
3. **Sweep coarse, then refine.** A 3 by 3 grid with 2 to 5 realizations
   tells you more per minute than an optimizer on an unvetted model.
4. **Optimize last,** with enough realizations that the objective is not
   noise, then confirm with a small sweep around the result.

Before any of this, estimate the run size (`references/guide/modes.md`,
"Run cost").

## Modeling

- Name components and resources with letters, digits, and underscores.
  Names become variable names (`<component>_capacity`) and column names.
- Keep resource spelling identical everywhere. Write them once, in a list,
  and copy.
- Use `fixed_value` for known capacities, and sweep or optimize only true
  decisions. Every extra decision variable multiplies sweep cost or makes
  optimization harder.
- Put availability profiles (wind, solar) in `<capacity_factor>`, and keep
  the capacity as installed size.
- Add an `independent` market or storage anywhere a `fixed` producer's
  output needs somewhere to go.
- Prefer `<ratio>` to `<poly>`, and keep `<Function>` code to plain
  arithmetic (`references/spec/functions.md`).

## Units and time

- Pick one unit system and state it in comments: capacity in MW, prices in
  $/MWh, hourly steps give MWh. Keep every price per the same quantity as
  the activity it multiplies.
- `<time_discretization>` must match the data exactly
  (`references/guide/time-series.md`).
- `<ProjectTime>`, `<lifetime>`, and the discount rate share a time unit
  (years). `period="year"` cash flows occur once per project year, and the
  default hourly ones every time step.
- `rolling_window_length` should be a multiple of the steps in a day for
  daily-cycling systems, long enough that a window does not cut a storage
  cycle, and must not leave a single trailing step.

## Economics

- Check every `<CashFlow>` against the sign table in
  `references/spec/economics.md`. If you give a positive price for a cost,
  make the sign explicit with a multiplier, and comment it.
- Capex: `one-time`, driven by `<variable>X_capacity</variable>`. Fixed O&M:
  `period="year"`. Variable O&M, fuel, and sales: hourly `repeating`, driven
  by `<activity>`.
- Set `<lifetime>` deliberately. A lifetime shorter than `<ProjectTime>`
  charges `one-time` cash flows again on replacement.
- `LC` replaces all other metrics and needs a nonlinear solver. Extra
  `<EconMetrics>` and `<result_statistics>` are cheap.

## Uncertainty and statistics

- Choose the number of realizations from the statistic you will quote: tens
  for means, hundreds for tails (`references/guide/uncertainty.md`).
- Match the opt objective statistic to the decision (mean for risk-neutral,
  VaR or ES for risk-averse), and check its direction.
- Report spread (`std`, percentiles) with every mean, unless the time series
  model is clustered, in which case only the mean is reliable
  (`references/guide/time-series.md`, "Clustering and higher moments").

## Reproducibility

- Keep the HERON input, data files, trained model, training input, and any
  `<Function>` Python file in one version-controlled directory, with
  relative paths.
- Record the RAVEN, TEAL, and HERON versions and the solver used.
- Regenerate `outer.xml`, `inner.xml`, and `heron.lib`; never edit them by
  hand.
- Set `<seed>` in `<BayesianOpt>` so an optimization can be repeated. A
  trained time series model has its own seed.
- Keep a small debug-mode version of each study as a regression check.

## Pitfalls at a glance

| Pitfall | Where it is described |
|---------|-----------------------|
| Market revenue counted as a cost | `references/spec/economics.md` |
| `sweep_values` / `opt_bounds` used against the mode | `references/spec/valued-params.md` |
| ARMA and CSV mixed; time grid differs from the data | `references/guide/time-series.md` |
| Too few realizations | `references/guide/uncertainty.md` |
| Optimum at an `opt_bounds` edge | `references/guide/results.md` |
| Debug-mode numbers read as results | `references/guide/modes.md` |
| Resource spelling mismatch; storage window effects | `references/guide/modeling-components.md` |

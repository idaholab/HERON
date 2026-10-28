# Best practices

A checklist for designing a study and reviewing an input. Each item points to
the file with the details.

## Build up in stages

1. **Start small.** Begin with the fewest components that make the question
   meaningful, one year of hourly data, and `<debug/>` (`modes.md`). Add
   components one at a time and re-run debug after each.
2. **Make the dispatch physical before looking at money.** In the dispatch
   plot, check that every component's activity has the right sign and shape,
   storage cycles, nothing sits pinned at a bound for no reason, and every
   resource balances. Only then look at the cash flows.
3. **Check cash flow signs and magnitudes** in the cash flow plot, against
   a back-of-the-envelope estimate (capacity times price times hours).
4. **Sweep coarse, then refine.** A 3 by 3 grid with 2 to 5 realizations
   tells you more per minute than an optimizer on an unvetted model.
5. **Optimize last,** with enough realizations that the objective is not
   noise, then confirm with a small sweep around the result.

## Modeling

- Name components and resources with letters, digits, and underscores.
  Names become variable names (`<component>_capacity`) and file column names.
- Keep resource spelling identical everywhere. Write them once, in a list, and
  copy.
- Use `fixed_value` for known capacities and a swept or optimized one only for
  true decisions. Every extra decision variable multiplies sweep cost or
  makes optimization harder.
- Put availability profiles (wind, solar) in `<capacity_factor>`, and
  keep the capacity as installed size.
- Give every swept or optimized value a `debug_value`.
- Use `dispatch="fixed"` or `"independent"` only. Do not use `dependent` in
  new inputs (`input/components.md`).
- Add an `independent` market or storage anywhere a `fixed` producer's
  output needs somewhere to go.
- Prefer `<ratio>` to `<poly>`. If you must use `<poly>`, validate the
  dispatch carefully.
- Keep `<Function>` code to plain arithmetic, and test it in debug mode.

## Units and time

- Pick one unit system and state it in comments: capacity in MW, prices in
  $/MWh, hourly steps give MWh. Keep all prices per the same quantity as the
  activity they multiply.
- `<time_discretization>` must match the data exactly. Setting
  `<time_variable>` and `<year_variable>` to the names in the training data
  avoids surprises.
- `<ProjectTime>`, `<lifetime>`, and the discount rate share a time unit
  (years). `period="year"` cash flows occur once per project year, and the
  default hourly ones occur every time step.
- `rolling_window_length` should be a multiple of the number of steps in a
  day for daily-cycling systems, and large enough that a window does not end
  in the middle of a storage cycle. A window must not leave a single trailing
  step.

## Economics

- Check every `<CashFlow>` against the sign table in `input/economics.md`.
- Costs are negative and revenues positive when you give the price. If
  you give a positive price for a cost, make the sign explicit with a
  multiplier, and comment it.
- Capex goes in a `one-time` cash flow driven by `<variable>X_capacity</variable>`;
  fixed O&M uses `period="year"`; variable O&M, fuel, and sales use hourly
  `repeating` cash flows driven by `<activity>`.
- Set `<lifetime>` deliberately. A lifetime shorter than `<ProjectTime>`
  charges `one-time` cash flows again on replacement.
- Only enable what you need: extra metrics (`<EconMetrics>`) and result
  statistics (`<result_statistics>`) add columns, not much cost, but `LC`
  replaces all other metrics and needs a nonlinear solver.

## Uncertainty and statistics

- Choose the number of realizations from the statistic you will quote
  (`uncertainty.md`): tens for means, hundreds for tails.
- Pick the opt objective statistic to match the decision (mean for
  risk-neutral; VaR or ES for risk-averse), and check its direction.
- Report spread (`std`, percentiles) with every mean.
- If the time series model is clustered with fewer clusters than segments,
  only the mean is reliable. Spread is inflated, so do not optimize a
  non-mean statistic on it, and say so when reporting spread
  (`time-series.md`, "Clustering and higher moments").

## Reproducibility

- Keep the HERON input, the data files, the trained model file, the training
  input, and any `<Function>` Python file in the same version-controlled
  directory. Use relative paths.
- Record the RAVEN, TEAL, and HERON versions and the solver used.
- The generated `outer.xml`, `inner.xml`, and `heron.lib` are products of the
  input; regenerate them, and never edit them by hand.
- For optimization, set `<seed>` in `<BayesianOpt>` so a run can be repeated.
  A trained time series model has its own seed (`<seed>` in the ROM).
- Keep a small debug-mode version of each study as a regression check.

## Pitfalls at a glance

| Pitfall | Where it is described |
|---------|-----------------------|
| Market revenue counted as a cost | `input/economics.md` |
| `sweep_values` / `opt_bounds` used against the mode | `valued-params.md` |
| ARMA and CSV mixed | `time-series.md` |
| Time grid differs from the data | `time-series.md` |
| Too few realizations | `uncertainty.md` |
| Optimum at an `opt_bounds` edge | `results.md` |
| Debug-mode numbers read as results | `modes.md` |
| Resource spelling mismatch | `components.md` |
| Storage window effects | `components.md` |

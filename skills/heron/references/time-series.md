# Time series

How HERON gets time-varying data (prices, loads, wind, solar) and how that
data lines up with the analysis. Declaring the sources is covered in
`input/data-generators.md`.

## Synthetic history (ARMA) or static history (CSV)?

| | `<ARMA>` | `<CSV>` |
|---|----------|---------|
| What it is | A trained, serialized RAVEN synthetic history model (`.pk`) | A fixed history read from a file |
| Realizations | A new random history per inner sample (`<num_arma_samples>`) | The same history every time (unless the file has several `RAVEN_sample_ID`s) |
| Use when | Variability matters and you have training data | You want a deterministic, known scenario, or a quick first model |
| Cost | Each realization is a dispatch solve | Cheapest case is a flat sweep: one solve per design point (`modes.md`) |

A case cannot use both. The template driver stops with "Mixing ARMA and CSV
sources is not yet supported".

## Training a synthetic history model

HERON does not train models; RAVEN does, ahead of time, and HERON loads the
serialized model. Working examples are `tests/integration_tests/ARMA/train_*.xml`
(RAVEN `ARMA`) and `tests/integration_tests/TSA/train_*_TSA.xml`. The pattern
in those files:

1. Load the training CSV into a `HistorySet`. The pivot parameter is the
   time variable.
2. Train a `ROM` of subType `ARMA` (the usual choice, with a `Fourier` series
   for daily or yearly cycles and a `Segment grouping="interpolate"` with a
   `macroParameter` of `Year` for multi-year histories).
3. Serialize it to a `.pk` file with an `IOStep` to a `Files` output.
4. Sample the ROM once and inspect the output before using it in HERON.

Run it with `raven_framework train_x.xml`, then point HERON at the `.pk`:

```xml
<ARMA name="synth" variable="PRICE,LOAD">arma.pk</ARMA>
```

### Names that must agree

| In HERON | Must equal |
|----------|------------|
| `variable="PRICE,LOAD"` on the `<ARMA>` generator | Signal (target) names in the trained model |
| `<time_variable>` in `<Case>` | The pivot parameter in the training data (default `time`; the training examples use `Time`) |
| `<year_variable>` in `<Case>` | The macro parameter (default `Year`) |
| `<ARMA variable="PRICE">synth</ARMA>` in a component | A `variable` listed on the generator named `synth` |

Two generators cannot list the same variable name. The default `time` and
`Year` differ in case from the training examples' `Time` and `Year`, so set
`<time_variable>` explicitly when your training data uses `Time`.

## How the data lines up with the analysis

- **Time steps.** The model's time axis must match `<time_discretization>`
  exactly. At run time the dispatch checks that the requested start and end
  are inside the history and that the number of steps equals the history
  length ("Requested number of steps (N) does not match ..."). HERON does not
  interpolate.
- **Years (macro steps).** At parse time HERON asks the model how many years
  it covers and compares to the project life (`<ProjectTime>`, less the
  construction year):
  - A single-year model is repeated for every project year (a RAVEN
    `Multicycle` is added).
  - A multi-year (interpolated) model must cover at least the project life.
    A longer model is truncated; a shorter one is an error ("interpolates N
    macro steps, but project life is M").
- **Clusters.** A model trained with clustering represents the year as a few
  representative segments, each standing for many segments of the original.
  By default (`evalMode` clustered) HERON dispatches only the representative
  clusters and weights their economics by how many segments each represents.
  This is much cheaper than a full year. In the outputs a cluster
  index (`_ROM_Cluster`) appears alongside the time and year indices.
  Clustering trades accuracy of the spread for speed. See "Clustering and
  higher moments" below before quoting anything other than a mean.
- **Realizations.** One realization is one full synthetic history over all
  years. `<num_arma_samples>` realizations are sampled per design point (or
  `<inner_samples>` in debug mode).

## Clustering and higher moments

When a clustered model has fewer clusters than segments, each cluster
represents several segments and `DispatchManager` multiplies that cluster's
cash flows and activity by the count of segments it represents (its
multiplicity, from `segments_represented`). This preserves the expected value
well. It does not preserve the spread. One dispatched segment is counted
m times, so within a realization the m segments it stands for behave as
perfectly correlated, where in the real year they would partly average out.
The result is that `std_NPV`, `var_NPV`, and the statistics built on higher
moments or tails are **overestimated** (inflated), relative to the same case
run with one cluster per segment.

What this means in practice:

- **Trust:** `mean_<metric>` and mean total activity. The default
  `expectedValue` objective is a sound choice with clustering.
- **Do not trust quantitatively:** `std`, `var`, `varCoeff`, `skew`, `kurt`,
  percentiles and extremes (`perc`, `min`, `max`), `valueAtRisk`,
  `expectedShortfall`, and the ratios that divide by spread (`sharpeRatio`,
  `sortinoRatio`, `gainLossRatio`). They will look worse (wider, riskier)
  than the system really is, and the error grows as the number of clusters
  falls below the number of segments.
- **Do not optimize** a spread, tail, or risk-adjusted statistic
  (`<stats_metric>` other than `expectedValue`) on a clustered model. The
  optimizer will favor designs by an error that is part of the model. If the
  decision needs one, use a model with one cluster per segment (or the
  `full` `evalMode`) and accept the extra run time, or report the value as a
  conservative bound only.
- To check how much it matters, run a small case both ways and compare
  `std_NPV`. The means should agree. The gap in `std_NPV` is the inflation.

HERON does not warn about any of this: it will run and report these statistics
for a clustered model without comment, so the check is on you.

## Static history files

See `input/data-generators.md` for the required columns. The extras:

- The year and time columns must match the names in `<Case>`.
- With one `RAVEN_sample_ID`, the case may qualify for the flat sweep
  (`modes.md`). With several, each ID is a realization.
- A one-year file is repeated for every project year. A multi-year file must
  cover the project life.
- Each requested `variable` must be a header. HERON checks this at parse time.

## Validate before using

1. In the RAVEN training run, plot or print a sampled synthetic history and
   compare its range and shape to the training data.
2. Run HERON in debug mode (`modes.md`). The parse prints "detected:
   interpolated / clustered" for each generator, and the dispatch plot shows
   the signal alongside the dispatch (the `GRO_debug_synthetics` signals).
3. Check that the signal's units match how the component uses it. A price in
   $/MWh against an activity in MW over hourly steps is consistent, and a
   price in $/kWh is not.

## Common problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Expected ARMA variable ... was not passed to DispatchManager" | `variable` is not a signal in the trained model | Compare with the model's targets |
| "Requested number of steps ... does not match" | `<num_steps>` or `<end_time>` differs from the model's time axis | Make `<time_discretization>` match the training data |
| "Requested end time ... is greater than time variable ..." | Case runs longer than the history | Shorten `<end_time>` or retrain |
| "interpolates N macro steps, but project life is M" | Model covers fewer years than `<ProjectTime>` | Retrain for more years or reduce `<ProjectTime>` |
| Results do not change between realizations | Model is deterministic, or `<num_arma_samples>` is 1 | Check the model's noise terms (the sine test model is deterministic) |

# Choosing a mode

A HERON study moves through three stages: **debug**, then **sweep**, then
**opt**. Each stage answers a different question, and each is more expensive
than the last. Settings for the nodes named here are in `input/case.md`.

## How HERON runs a case

`heron input.xml` does not solve anything itself (for the default
`<workflow>standard</workflow>`). It parses the input and writes RAVEN input
files next to the HERON input file, then you run those with RAVEN:

| File written | What it is |
|--------------|------------|
| `outer.xml` | The RAVEN workflow you run (`raven_framework outer.xml`). |
| `inner.xml` | The dispatch workflow that `outer.xml` launches once per design point. Not written in debug mode or in the flat sweep case below. |
| `heron.lib` | A pickle of the parsed case, components, and sources. The dispatch reads it at run time. |

The outer workflow (and the debug workflow) write results to a working
directory named `<Case name>_o`. The inner workflow's own working directory is
named `<Case name>_i`.

Which workflow gets written depends on the case
(`templates/template_driver.py`, `TemplateDriver.create_workflow`):

| Condition (checked in this order) | Workflow |
|-----------------------------------|----------|
| `<debug>` is present | **Debug**: one flat run, fixed capacities, plots |
| Only `<CSV>` sources, `mode` is `sweep`, no uncertain cash flow parameters, and the CSV has a single sample | **Flat sweep**: one dispatch per design point, no `inner.xml` |
| Anything else | **Bilevel**: the outer run samples or optimizes capacities and launches `inner.xml` for each |

Mixing `<ARMA>` and `<CSV>` sources is rejected outright ("Mixing ARMA and CSV
sources is not yet supported").

## Debug

Purpose: check that the model is specified correctly before spending compute.
Add `<debug/>` to `<Case>`; remove it for real runs.

What it changes:
- Runs a **single design point**. Each `sweep_values` or `opt_bounds` uses its
  `debug_value` attribute when given.
- Runs `<inner_samples>` realizations (default 1) instead of
  `<num_arma_samples>`, and `<macro_steps>` years (default 1) instead of
  `<ProjectTime>`.
- Writes `dispatch_print.csv` and `cashflows*.csv` (and a `dispatch.nc`
  database), plus a dispatch plot and a cash flow plot (each switchable with `<dispatch_plot>` and
  `<cashflow_plot>`; `sep_by_resource="True"` gives one figure per resource).
- Writes `network.png` (the resource graph of the components) next to the
  input file when `heron` parses a debug case.

**Always set `debug_value`** on every swept or optimized capacity. Without it,
the capacity is not held fixed: debug mode samples it from its range, so the
dispatch you inspect is not a design you chose.

```xml
<capacity>
  <sweep_values debug_value="100">50, 100, 200</sweep_values>
</capacity>
```

What to check in the debug output, in order:
1. The resource graph (`network.png`): are the components connected the way
   you meant? Connections exist only through shared resource names.
2. The dispatch plot: does each component's activity have the sign and size
   you expect? Does storage charge when prices are low and discharge when
   they are high? Is any demand left unmet?
3. The cash flow plot: do revenues and costs have the right sign and order
   of magnitude? A market that shows as a cost is the classic sign error
   (see `input/economics.md`).

Do not read economics from a debug run. Its single year and few realizations
are only meant to prove the model runs and behaves.

## Sweep

Purpose: map how the metric responds to design choices on an explicit grid.
Set `<mode>sweep</mode>` and give `sweep_values` for each component
capacity you want to vary (others use `fixed_value`).

- The outer run is a **grid over the product** of all `sweep_values` lists.
  Three capacities with 4, 5, and 3 values is 60 design points, and each point
  is a full inner run of `<num_arma_samples>` realizations.
- Size the grid first with a handful of points per dimension. Refine around
  the interesting region in a second sweep, rather than starting fine.
- Output is `sweep.csv` in `<Case name>_o/`: one row per design point, with
  the capacities and the result columns. In the bilevel workflow these are
  statistics named `<stat>_<variable>`: `mean_`, `std_`, and `med_` always,
  plus `max_`, `min_`, `perc_5_`, `perc_95_`, `samp_`, and `var_` in sweep
  mode (for example `mean_NPV`, `perc_5_NPV`). Add others with
  `<result_statistics>` (see `uncertainty.md`). In the flat sweep (single
  static history) there is no statistic, so the columns are the plain values
  (`NPV`, `TotalActivity__<comp>__<tag>__<resource>`).
- Prefer sweep over opt when there are one or two decision variables, when
  you want to see the shape of the response (not just the best point), or when
  the objective may be flat or noisy enough to confuse an optimizer.

## Opt

Purpose: find the best capacities without enumerating a grid. Set
`<mode>opt</mode>`, give `opt_bounds` (low, high) for each capacity to be
optimized, and add `<optimization_settings>`.

- `<opt_metric>` is the economic metric (default `NPV`) and `<stats_metric>`
  is the statistic of it across realizations (default `expectedValue`). The
  direction (`min` or `max`) defaults from the statistic, so check it with
  `<type>` if you use risk or spread statistics (see `uncertainty.md`).
- Algorithms (`<algorithm>`): `BayesianOpt` (default) or `GradientDescent`.
  The values below are the template defaults, which `<limit>`,
  `<persistence>`, and `<convergence>` override
  (`templates/snippets/optimizers.py`):

  | | BayesianOpt | GradientDescent |
  |---|---|---|
  | Max outer runs (`limit`) | 100 | 800 |
  | Consecutive converged iterations (`persistence`) | 4 | 1 |
  | Convergence tolerances | acquisition 1e-5 | gradient 1e-4, objective 1e-8 |

- Output is in `<Case name>_o/`: `opt_soln.csv` is an index (`trajID`,
  `filename`) of one `opt_soln_<trajID>.csv` per optimizer trajectory, and
  those files hold the results, plus an optimization path plot. Rows are
  iterations; columns are `iteration`, `accepted`, the capacities, and result
  statistics named `<stat>_<variable>`, such as `mean_NPV`, `std_NPV`,
  `med_NPV`. Interpretation is in `results.md`.
- Every outer iteration is a full inner run, so noise in the objective
  matters: with few realizations, the optimizer can chase sampling noise.
  Raise `<num_arma_samples>` before trusting a result.
- Always confirm an optimum by re-running a small sweep around it.

## Decision guide

| Situation | Use |
|-----------|-----|
| New or modified input; anything unexpected in results | debug |
| Understanding sensitivity, 1 to 2 decision variables | sweep |
| Many decision variables, or a grid is too expensive | opt |
| Fixed capacities, you only want the statistics of one design | sweep with one value per capacity |
| Checking an optimizer's answer | sweep around it |

`<workflow>` can also be `MOPED` (a single monolithic Pyomo solve instead of
RAVEN), `combined`, or `DISPATCHES`. Those are alternative solvers that do
not follow this debug, sweep, opt progression. Use `standard` unless you
know you need one.

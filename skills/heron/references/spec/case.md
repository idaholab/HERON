# `<Case>` node

`<Case name="...">` sets the solve mode, the time grid for dispatch,
project-wide economics, the dispatcher, and sampling. Child order does not
matter.

## Required children

| Node | Purpose |
|------|---------|
| `<mode>` | `sweep` (evaluate a grid of `sweep_values`) or `opt` (search `opt_bounds` for the best value of a metric). |
| `<time_discretization>` | The time grid for each dispatch (see below). |
| `<economics>` | Project-wide economics (see below). |
| `<dispatcher>` | How each dispatch is solved (see below). |

## `<time_discretization>`

Defines the time steps inside one cycle (usually one year) of dispatch.

| Child | Meaning | Default |
|-------|---------|---------|
| `<time_variable>` | Name of the time variable. Must match the time variable in any ARMA or CSV data. | `time` |
| `<year_variable>` | Name of the cycle (year) variable. Must match the ARMA or CSV data. | `Year` |
| `<start_time>` | Time value where dispatch starts. | `0` |
| `<end_time>` | Time value where dispatch ends. | (none) |
| `<num_steps>` | Number of time steps. Give this or `<time_interval>`. | (none) |
| `<time_interval>` | Length of one step, in units of the time variable. An integer number of steps must fit between start and end. | (none) |

Example, hourly over one day (24 steps from 0 to 23):

```xml
<time_discretization>
  <year_variable>YEAR</year_variable>
  <time_variable>HOUR</time_variable>
  <end_time>23</end_time>
  <num_steps>24</num_steps>
</time_discretization>
```

## `<economics>`

Settings applied to every component's cash flows.

| Child | Meaning |
|-------|---------|
| `<ProjectTime>` | Number of cycles (years) the analysis covers. |
| `<DiscountRate>` | Discount rate for the multi-year cash flow analysis, as a fraction (e.g. `0.08`). |
| `<tax>` | Tax rate, as a fraction. |
| `<inflation>` | Inflation rate per cycle, as a fraction. |
| `<verbosity>` | Integer output level for the economics calculation. |
| `<EconMetrics>` | Extra metrics to report for every run: any of `<NPV/>`, `<IRR/>`, `<PI/>`, `<LC/>`. Default is NPV. In opt mode the `<opt_metric>` is always added. Requesting `LC` replaces the other metrics, because levelized cost cannot be computed alongside them. |

## `<dispatcher>`

Holds exactly one dispatcher:

- `<pyomo/>`: the standard optimizer, with perfect foresight over rolling
  windows. Options:
  - `<rolling_window_length>`: time steps per window. Default `24`. Longer
    windows reduce edge effects (for example, storage being emptied at the
    end of each window) but solve more slowly.
  - `<solver>`: the Pyomo solver. Default `cbc` (`glpk` on Windows). Some
    features need a capable solver: polynomial transfer functions need a
    nonlinear solver (the tests use `ipopt`), and `ramp_freq` adds binary
    variables, so it needs a mixed-integer solver.
  - `<tol>`: relative solver tolerance. The default depends on the solver.
  - `<debug_mode>`: extra printing. Default `False`. Do not use for
    production runs.
- `<custom>`: a user-supplied Python dispatcher, with `<location>` the path
  to the module. **Untested and probably broken:** it has no regression
  test, and its only example (`tests/integration_tests/mechanics/custom_dispatcher/`)
  does not match the signature the code calls (`dispatch(meta, state)` in
  `src/dispatch/CustomDispatcher.py`). Do not recommend it. If a user needs
  it, read that source first and tell them it is unsupported.

## Sampling and run size

| Node | Meaning | Default |
|------|---------|---------|
| `<num_arma_samples>` | Number of stochastic realizations (inner samples) per design point. Each one is a separate dispatch and cash flow evaluation. | `1` |
| `<debug>` | Runs a reduced-size check of one design point and writes dispatch and cash flow plots (details below). | off |
| `<parallel>` | `<outer>` and `<inner>` process counts, and an optional `<runinfo>` block (`<expectedTime>`, `<clusterParameters>`, `<memory>`, `<RemoteRunCommand>`) for clusters. Its presence also writes HPC queue settings; see `references/guide/modes.md`, "Running in parallel". | no parallelism |

`<debug>` children, all optional (an empty `<debug/>` is valid):

| Child | Meaning | Default |
|-------|---------|---------|
| `<inner_samples>` | Realizations to run, replacing `<num_arma_samples>`. | `1` |
| `<macro_steps>` | Cycles (years) to run, replacing `<ProjectTime>`. | `1` |
| `<dispatch_plot>` | Write a dispatch plot. Attribute `sep_by_resource="True"` makes one figure per resource. | `True` |
| `<cashflow_plot>` | Write a cash flow plot. | `True` |

In debug mode each `sweep_values` or `opt_bounds` is replaced by its
`debug_value` attribute if one is set, e.g.
`<sweep_values debug_value="100">1, 100</sweep_values>`.

## `<optimization_settings>` (opt mode)

| Child | Meaning | Default |
|-------|---------|---------|
| `<opt_metric>` | Metric to optimize: `NPV`, `IRR`, `PI`, or `LC`. | `NPV` |
| `<stats_metric>` | Statistic of that metric across realizations: `expectedValue`, `median`, `minimum`, `maximum`, `percentile` (attribute `percent`), `valueAtRisk` or `expectedShortfall` (attribute `threshold`, an alpha in 0 to 1), `sharpeRatio`, `sortinoRatio` or `gainLossRatio` (attribute `threshold`, `median` or `zero`), and others. | `expectedValue` |
| `<type>` | `min` or `max`. | Depends on the statistic: `max` for central and ratio statistics, `min` for spread and risk statistics. The direction is reversed when `opt_metric` is `LC`. |
| `<algorithm>` | `<BayesianOpt>` (options `<acquisition>`, `<kernel>`, `<seed>`, `<modelSelection>`) or `<GradientDescent>` (options `<growthFactor>`, `<shrinkFactor>`, `<initialStepScale>`). | `BayesianOpt` |
| `<initialCount>` | Initial samples (BayesianOpt) or trajectories (GradientDescent). | |
| `<persistence>` | Number of consecutive converged iterations required before stopping. | |
| `<convergence>` | `<objective>`, `<gradient>`, and `<stepSize>` tolerances. | |
| `<limit>` | Maximum number of outer runs. | |

## Other optional `<Case>` children

| Node | Meaning |
|------|---------|
| `<result_statistics>` | Extra statistics to report, as empty nodes (e.g. `<percentile percent="5"/>`, `<valueAtRisk/>`). Mean, standard deviation, and median are always reported. |
| `<label name="...">value</label>` | A fixed tag passed through to the outputs. Not used in any calculation. |
| `<dispatch_vars>` | Extra design variables that are not tied to a component (below). |
| `<validator>` | Checks each solved dispatch and re-solves with tighter limits (below). |
| `<data_handling><inner_to_outer>` | Data format passed from the inner runs to the outer run: `netcdf` (default) or `csv` (slower, but easier to inspect when debugging). |
| `<verbosity>` | `silent`, `quiet`, `all` (default), or `debug`. |
| `<workflow>` | `standard` (default), `MOPED`, `combined`, or `DISPATCHES`. These are alternative solve workflows. |

### `<dispatch_vars>`

A design variable that no component owns, such as a bid adjustment that a
`<Function>` reads. Each `<variable name="X">` holds one ValuedParam. A
`sweep_values` or `opt_bounds` makes it an outer design variable named
`X_dispatch` (a column in `sweep.csv`), and it reaches the dispatch as
`meta['HERON']['RAVEN_vars']['X']` (`references/spec/functions.md`). Elsewhere
it can be referred to as `<variable>X</variable>`.

```xml
<dispatch_vars>
  <variable name="bid_adjust">
    <sweep_values debug_value="1">1, 100</sweep_values>
  </variable>
</dispatch_vars>
```

### `<validator>`

Only works with `<pyomo/>`. After each window is solved, the validator
checks the dispatch. Each violation it reports becomes a new upper or lower
production limit at that time step, and the window is solved again, up to
100 attempts ("Exceeded validation attempt limit!"). Validators:

| Node | Does |
|------|------|
| `<Example>` | Demonstration only. Limits the change in any activity between steps to `<delta>`, within `<tolerance>`. |
| `<FARM_SISO>`, `<FARM_MIMO>` | From the FARM plugin, if installed. Constrains dispatch with a reduced-order model of the plant's control. See FARM's documentation. |

Messages are printed with `DEBUGG ... validation concerns raised:`. A new
validator subclasses `src/validators/Validator.py` and is registered in
`src/validators/Factory.py`.

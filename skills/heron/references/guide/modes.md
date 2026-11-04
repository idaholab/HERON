# Choosing a mode

A HERON study moves through three stages: **debug**, then **sweep**, then
**opt**. Each stage answers a different question, and each is more expensive
than the last. Settings for the nodes named here are in `references/spec/case.md`.

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

A case may not mix `<ARMA>` and `<CSV>` sources
(`references/guide/time-series.md`). The math of the bilevel problem is in
`references/theory/bilevel.md`.

## Run cost

Every dispatch solve is one rolling window of one cluster (or segment) of
one year of one realization. Estimate before running:

```
solves = design points x realizations x years x clusters x windows per cluster
```

- **Design points:** product of the `sweep_values` list lengths (sweep), or
  up to `<limit>` iterations, each with several evaluations (opt; below).
- **Realizations:** `<num_arma_samples>`.
- **Years:** `<ProjectTime>`. Every project year is dispatched, even when a
  single-year model is replayed for each one.
- **Clusters:** the number of representative clusters in a clustered model,
  otherwise 1. The steps per cluster are `<num_steps>`.
- **Windows:** `<num_steps>` / `rolling_window_length`, rounded up.

A window of 24 hourly steps with a few components solves in a fraction of a
second with cbc, so the count of solves is usually what matters. Time one
debug run (the solve time is printed as `DEBUGG solve time`) and multiply.
Binary variables (`ramp_freq`) and nonlinear solvers (`<poly>`, `LC`) cost
much more per solve.

## Running in parallel

`<parallel>` in `<Case>` sets how many inner runs (design points) run at
once (`<outer>`), and how many realizations each inner run evaluates at once
(`<inner>`). Their product should not exceed the cores available. HERON
warns ("System may be overloaded") when it does.

| Setting | Default when `<parallel>` is present |
|---------|--------------------------------------|
| `<inner>` | `<num_arma_samples>` |
| `<outer>` | Number of sampled capacities + 1 (one gradient evaluation for opt) |

Any `<parallel>` node also writes HPC settings into the RunInfo of
`outer.xml`. HERON looks for a preset matching the hostname in
`templates/parallel/` (`sawtooth`, `bitterroot`). On any other machine it
uses `<mode>mpi<runQSUB/></mode>`, which submits to a PBS queue. Check the
`<RunInfo>` in the generated `outer.xml` before running, especially on a
workstation. `<runinfo>` inside `<parallel>` adds cluster settings
(`<expectedTime>`, `<clusterParameters>`, `<memory>`, `<RemoteRunCommand>`)
to that block. For example, on a machine with no preset, this in `<Case>`:

```xml
<parallel>
  <runinfo>
    <expectedTime>72:0:0</expectedTime>
    <clusterParameters>-P nst</clusterParameters>
    <memory>4g</memory>
  </runinfo>
</parallel>
```

gives this `<RunInfo>` in `outer.xml`, for an opt case named `Opt_Runs` with
two optimized capacities and `<num_arma_samples>3</num_arma_samples>`
(`batchSize` is the default `<outer>`, 2 + 1; `NumMPI` the default `<inner>`):

```xml
<RunInfo>
  <JobName>Opt_Runs_o</JobName>
  <WorkingDir>Opt_Runs_o</WorkingDir>
  <Sequence>optimize, plot</Sequence>
  <batchSize>3</batchSize>
  <expectedTime>72:0:0</expectedTime>
  <clusterParameters>-P nst</clusterParameters>
  <mode>mpi
    <runQSUB />
    <memory>4g</memory>
  </mode>
  <NumMPI>3</NumMPI>
  <internalParallel>True</internalParallel>
</RunInfo>
```

## Debug

Purpose: check that the model is specified correctly before spending compute.
Add `<debug/>` to `<Case>`; remove it for real runs.

What it changes:
- Runs a **single design point**. Each `sweep_values` or `opt_bounds` uses its
  `debug_value` attribute when given.
- Runs `<inner_samples>` realizations (default 1) instead of
  `<num_arma_samples>`, and `<macro_steps>` years (default 1) instead of
  `<ProjectTime>`.
- Writes dispatch and cash flow CSVs, a `dispatch.nc` database, and plots
  (names and contents in `references/guide/results.md`). `network.png`, the
  resource graph, is written next to the input file when `heron` parses a
  debug case.

**Always set `debug_value`** on every swept or optimized capacity. Without it,
the capacity is not held fixed: debug mode samples it from its range, so the
dispatch you inspect is not a design you chose.

What to check in the debug output, in order:
1. The resource graph (`network.png`): are the components connected the way
   you meant? Connections exist only through shared resource names.
2. The dispatch plot: does each component's activity have the sign and size
   you expect? Does storage charge when prices are low and discharge when
   they are high? Is any demand left unmet?
3. The cash flow plot: do revenues and costs have the right sign and order
   of magnitude? A market that shows as a cost is the classic sign error
   (see `references/spec/economics.md`).

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
  the capacities and statistics of the results such as `mean_NPV`
  (`references/guide/uncertainty.md` lists which). The flat sweep has no
  statistics, so its columns are plain `NPV` and
  `TotalActivity__<comp>__<tag>__<resource>`.
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
  `<type>` if you use risk or spread statistics (see `references/guide/uncertainty.md`).
- Algorithms (`<algorithm>`): `BayesianOpt` (default) or `GradientDescent`.
  Template defaults, which `<limit>`, `<persistence>`, and `<convergence>`
  override:

  | | BayesianOpt | GradientDescent |
  |---|---|---|
  | Max outer runs (`limit`) | 100 | 800 |
  | Consecutive converged iterations (`persistence`) | 4 | 1 |
  | Convergence tolerances | acquisition 1e-5 | gradient 1e-4, objective 1e-8 |

- Output is in `<Case name>_o/`: `opt_soln.csv` indexes one
  `opt_soln_<trajID>.csv` per optimizer trajectory, which holds one row per
  iteration, plus an optimization path plot. How to read them is in
  `references/guide/results.md`.
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

---
name: heron
description: Build, configure, and run HERON (Holistic Energy Resource Optimization Network) technoeconomic analyses. Use when the user is writing or editing a HERON XML input file; defining components, resources, or economics; setting up time series (synthetic history / ARMA) or other uncertainty; choosing between debug, sweep, and opt modes; or asking how to structure a HERON study. For failing or suspicious runs, use the heron-debug skill instead.
---

# HERON

HERON is a stochastic techno-economic analysis tool for energy systems.
HERON optimizes or performs parameter sweeps for system designs for economic performance metrics considering parametric and time series uncertainties subject to optimal unit commitment.

## Input file anatomy

A HERON input is one XML file with root `<HERON>` and three required children:

1. **`<Case>`**: how to solve. Mode, time discretization, project
   economics, dispatcher, and sampling.
2. **`<Components>`**: what the system is. One `<Component>` per unit, with
   what it produces, stores, or demands, and its cash flows.
3. **`<DataGenerators>`**: where external data comes from. Synthetic
   history models (`<ARMA>`), static histories (`<CSV>`), Python functions
   (`<Function>`), and RAVEN ROMs (`<ROM>`).

Components refer to data generators by `name`, and components are connected
to each other only through shared resource names. Read
`references/input-file.md` before writing or editing an input.

## Recommended workflow

Move through the modes in order, and only advance once the current stage
looks right. Details are in `references/modes.md`.

1. **Debug.** Add `<debug/>` to `<Case>` and give every swept or optimized
   capacity a `debug_value`. Run it and check the resource graph, the dispatch
   plot, and the cash flow plot. This proves the model runs and behaves
   physically. Do not draw economic conclusions from it.
2. **Sweep.** Remove `<debug/>`, use `<mode>sweep</mode>`, and start with a
   coarse grid and few `<num_arma_samples>`. This shows how the metric responds
   to each design choice.
3. **Opt.** Only if the design space is too large to sweep. Raise
   `<num_arma_samples>` first so the optimizer is not chasing sampling noise,
   then confirm the optimum with a small sweep around it.

`heron input.xml` only writes the RAVEN workflow (`outer.xml`, usually
`inner.xml`, and `heron.lib`); running it is a separate `raven_framework`
step on `outer.xml`. If a step fails or the results look wrong, switch to
the heron-debug skill.

## References

Load these only when the task needs them:

| File | Read when |
|------|-----------|
| `references/input-file.md` | Writing or editing any part of a HERON input file |
| `references/components.md` | Defining producers, demanders, storage, capacities, transfer functions |
| `references/valued-params.md` | Choosing how a value is supplied (fixed, sweep/opt, Function, SyntheticHistory, ...) |
| `references/time-series.md` | Training or using synthetic history models, realizations, clustering |
| `references/uncertainty.md` | Deciding what is stochastic, sample counts, statistics and objectives |
| `references/modes.md` | Choosing debug vs. sweep vs. opt, the files HERON generates, and where results land |
| `references/results.md` | Reading `sweep.csv`, `opt_soln*.csv`, and debug plots and metrics |
| `references/best-practices.md` | Reviewing a study design or input for common mistakes |

The `references/input/` files hold the per-node detail (`case.md`,
`components.md`, `economics.md`, `valued-params.md`, `functions.md`,
`data-generators.md`); `references/input-file.md` is the index to them.

## Source of truth

These references were written from HERON's source. When they disagree with
the code, the code is right, so check it and tell the user about the
mismatch. Where to look:

| Question | Source |
|----------|--------|
| What a `<Case>` node accepts, defaults, and metric/statistic names | `src/Cases.py` (`get_input_specs`, `stats_metrics_meta`, `economic_metrics_meta`) |
| What a component, producer, storage, or demand accepts | `src/Components.py` |
| Cash flow rules | `src/Economics.py`, `src/CashFlowUser.py` |
| ValuedParam types and where they are allowed | `src/ValuedParams/`, `src/ValuedParamHandler.py` |
| Transfer functions | `src/TransferFuncs/` |
| Data generators, path resolution, ARMA/CSV checks | `src/Placeholders.py`, `src/input_loader.py` |
| Which RAVEN workflow gets written and why | `templates/template_driver.py` |
| What the generated workflows contain | `templates/*.py`, `templates/xml/` |
| How dispatch is solved and failures are raised | `src/dispatch/pyomo_dispatch.py`, `PyomoModelHandler.py` |
| How results are computed per run | `src/DispatchManager.py` |
| Working examples | `tests/integration_tests/` (small, one feature each) and `tests/workshop/` (full studies) |

`heron --definition` prints the input definition as HERON itself sees it.

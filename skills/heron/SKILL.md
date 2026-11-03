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
   (`<Function>`), and RAVEN ROMs (`<ROM>`). Required, but may be empty.

Components refer to data generators by `name`, and components are connected
to each other only through shared resource names. A name that does not match
is an error.

To start a new input, copy `assets/minimal_sweep.xml` (or
`assets/minimal_opt.xml`). Each is a complete, commented case with the
required nodes and correctly signed cash flows.

## Gotchas

- **Signs.** Activity is positive when a component adds a resource and
  negative when it removes one, and capacities follow it. A market
  (`<demands>`) has negative activity, so its sales need
  `<multiplier>-1</multiplier>` on the driver or they count as a cost, with
  no warning. Check every cash flow against `references/spec/economics.md`.
- **Tag must match mode.** `sweep_values` for sweep, `opt_bounds` for opt,
  each with a `debug_value`. HERON does not warn when they are swapped.
- **Use `dispatch="fixed"` or `"independent"` only.** `dependent` is legacy.

## Recommended workflow

Move through the modes in order, and only advance once the current stage
looks right. Details are in `references/guide/modes.md`.

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

Load only what the task needs. Guides help you decide; specs list every
option of one node.

**Guides** (`references/guide/`)

| File | Read when |
|------|-----------|
| `modes.md` | Choosing debug vs. sweep vs. opt, the files HERON generates, and where results land |
| `modeling-components.md` | Turning a physical system into components: interactions, capacities, transfer functions, storage |
| `choosing-values.md` | Choosing how a value is supplied (fixed, sweep/opt, ARMA, CSV, Function, ...) |
| `time-series.md` | Using synthetic history models or static histories, realizations, clustering |
| `uncertainty.md` | Deciding what is stochastic, sample counts, statistics and objectives |
| `results.md` | Reading `sweep.csv`, `opt_soln*.csv`, and debug plots and metrics |
| `best-practices.md` | Reviewing a study design or input for common mistakes |

**Specs** (`references/spec/`)

| File | Read when writing |
|------|-------------------|
| `case.md` | `<Case>`: mode, time grid, project economics, dispatcher, sampling, debug, optimization settings |
| `components.md` | `<Component>`: produces, stores, demands, capacity, transfer functions, ramping, storage options |
| `economics.md` | A component's `<economics>` and `<CashFlow>` nodes, and the sign conventions |
| `valued-params.md` | Any value slot (`fixed_value`, `sweep_values`, `opt_bounds`, `ARMA`, `CSV`, `Function`, `activity`, ...) |
| `functions.md` | The Python functions behind `<Function>` values or a storage `<strategy>` |
| `data-generators.md` | `<ARMA>`, `<CSV>`, `<Function>`, or `<ROM>` sources, and file paths |
| `test-info.md` | A regression test input for HERON itself |

**Theory** (`references/theory/`). Read only for "why" questions: why a
dispatch or an NPV came out as it did, or what HERON actually computes.

| File | Read when |
|------|-----------|
| `dispatch.md` | The dispatch optimization: objective, balance, bounds, transfer, ramping, storage, rolling windows |
| `economics.md` | How cash flows become NPV, IRR, PI, LC: tax, inflation, MACRS, replacement, discounting |
| `bilevel.md` | The outer/inner problem structure and its consequences |
| `time-indexing.md` | Realizations, years, segments, clusters, multiplicity, and how results are summed |

**Source** (`references/source-map.md`): where in HERON's code each topic
lives. When a reference and the code disagree, the code is right; tell the
user about the mismatch.

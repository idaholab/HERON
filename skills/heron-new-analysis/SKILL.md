---
name: heron-new-analysis
description: Guided, step-by-step workflow for building a new HERON analysis from scratch with the user. Use when the user wants to start a new HERON study or input file and needs help deciding what to model, rather than editing one node of an existing input. Walks through choosing sweep or opt, inventorying components and resources, interviewing the user about each component's dispatch, capacity, and economics, setting up data generators and the time grid, and a first debug run. For syntax of individual nodes, use the heron skill; for failing runs, use heron-debug.
---

# Building a new HERON analysis

This is an interview, not a code generator. The user knows the system and
the question; you know HERON. Work through the phases below in order. At the
end of each phase, summarize what was decided and get the user's agreement
before starting the next. Ask a few questions at a time, not a form.

Node syntax lives in the heron skill. Load from it only what each phase
needs:

- `../heron/assets/minimal_sweep.xml`, `../heron/assets/minimal_opt.xml`:
  starting templates
- `../heron/references/spec/*.md`: every option of each node
- `../heron/references/guide/*.md`: how to decide

Keep a running decision record (below) in the conversation and update it at
every checkpoint. Write the XML as decisions are made, not all at the end.

## Phase 1. The question, and sweep or opt

Ask:
1. What decision is the study meant to inform? (Usually: what capacities to
   build, or whether a design is economic.)
2. Which quantities are design choices? These become the swept or optimized
   capacities. Everything else is fixed or data.
3. What metric decides it? (NPV, IRR, PI, levelized cost; and which
   statistic over realizations, usually the expected value.)
4. How fine a resolution is needed on each design choice, and how much
   compute is available?

Choose the mode with `../heron/references/guide/modes.md` ("Run cost" and
"Decision guide"). Rule of thumb: one or two design variables at a useful
resolution is a sweep; more than that, or a grid too large to run, is opt.
If no capacity varies, it is a sweep with one value per capacity.

The mode fixes the tag for every design value: `sweep_values` for sweep,
`opt_bounds` for opt, and a `debug_value` on each either way.

**Checkpoint:** mode, design variables, metric and statistic.

## Phase 2. System inventory

List every component and every resource before configuring any of them.

1. Ask for the units in the system: generators, storage, conversion
   (e.g. electrolysis), markets and loads, and fuel or other inputs bought
   from outside.
2. For each, ask which resources it takes in and which it puts out. Agree on
   exact resource names now; components connect only through matching names.
3. Draw the resource graph as text and show it to the user:

   ```
   npp ──electricity──> grid (market)
    └───electricity──> electrolyzer ──H2──> h2_market
   ```

4. Check that every resource has at least one producer and one consumer, and
   that anything leaving the system boundary goes to a market or sink
   component. A resource nobody consumes, or a demand nobody can supply, is
   usually an infeasible dispatch later.

Use `../heron/references/guide/modeling-components.md` to map each unit to
`<produces>`, `<stores>`, or `<demands>`.

**Checkpoint:** component list, resource names, resource graph.

## Phase 3. Each component

Take one component at a time, in the order of the graph (sources first).
For each, go through `references/component-interview.md`, which lists the
questions for: interaction and resources, dispatch policy, capacity,
transfer function, storage options, and cash flows.

For every value the user gives, label where it comes from and record it.
Use `../heron/references/guide/choosing-values.md` to decide:

| Label | Means | Becomes |
|-------|-------|---------|
| fixed | A known constant | `<fixed_value>` |
| design | Swept or optimized (phase 1) | `<sweep_values>` / `<opt_bounds>` with `debug_value` |
| data | A time series from a generator | `<ARMA>` / `<CSV>` (phase 4) |
| function | Computed from other values | `<Function>` |
| uncertain | A random parameter drawn per realization | `<uncertainty>` (cash flow parameters only) |

Write the component's XML right after its interview. Check each cash flow's
sign against `../heron/references/spec/economics.md` as it is written: say
out loud whether it should be a revenue or a cost, and confirm that the
driver's sign and `<multiplier>` produce that. A market's sales need
`<multiplier>-1</multiplier>`.

Anything labeled "data" gets a placeholder generator name now; phase 4
defines it.

**Checkpoint (per component):** the component's XML and its rows in the
decision record.

## Phase 4. Data generators and the time grid

1. For each "data" value, find out what file supplies it: a trained
   synthetic history ROM (`.pk`) or a CSV. A case uses one or the other,
   not both (`../heron/references/guide/time-series.md`). If the user has
   only raw data and wants stochastic realizations, a ROM must be trained in
   RAVEN first; point them to the training examples in that guide.
2. Interrogate each file with HERON (the `heron` launcher activates the
   RAVEN environment itself):

   ```bash
   heron --inspect-data path/to/arma.pk     # or a CSV
   ```

   It reports the signal names, time and year variable names, the years
   covered, and the time axis and cluster count for each evaluation mode,
   with a `<time_discretization>` to match.
3. Fill in `<DataGenerators>` and `<Case>` from that report using
   `references/time-grid.md`: `variable` names, `<time_variable>`,
   `<year_variable>`, `<time_discretization>`, and a check of `<ProjectTime>`
   against the years the data covers.
4. Replace the phase 3 placeholders with the real generator and variable
   names.

**Checkpoint:** `<DataGenerators>`, `<time_discretization>`, and any
mismatch between the data and the project life.

## Phase 5. Debug run

1. Add `<debug/>` to `<Case>`. Confirm every `sweep_values` and
   `opt_bounds` has a `debug_value`.
2. `heron input.xml`, then `raven_framework outer.xml` in the same directory.
3. Go through `references/debug-checklist.md` with the user: the resource
   graph, the dispatch plot, each cash flow's sign and size, and the
   order of magnitude of the metric.
4. If anything fails or looks wrong, switch to the heron-debug skill, fix
   the input, and rerun. Change one thing at a time.
5. When the debug run is clean, remove `<debug/>`. Suggest the next step
   from `../heron/SKILL.md` ("Recommended workflow"): a coarse sweep with few
   realizations before anything expensive.

Do not draw economic conclusions from the debug run.

## Decision record

Keep this table current and show it at each checkpoint:

| Component | Item | Value | Label | Notes |
|-----------|------|-------|-------|-------|
| npp | capacity (electricity) | 1000 MW | fixed | |
| grid | price | `synth` / `PRICE` | data | $/MWh |
| npp | fixed O&M | 120,000 $/MW-yr | uncertain | triangular, ±20% |

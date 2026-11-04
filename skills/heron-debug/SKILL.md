---
name: heron-debug
description: Diagnose HERON runs that fail or produce suspicious results. Use when a HERON input fails to parse, RAVEN errors during the outer or inner workflow, a dispatch solve is infeasible or crashes, or outputs look wrong (NaN or extreme NPV, flat or nonsensical dispatch, optimizer not converging). For building or configuring a model, use the heron skill instead.
---

# Debugging HERON

A HERON study has a pipeline with several stages, and the first step in
debugging is deciding which stage failed. The error text usually says.

## Triage

1. **Which stage?** Use the table below.
2. **Find the first real error.** RAVEN prints long tracebacks, and the useful
   line is usually the first `Error` or `Exception`, not the last. In a
   bilevel run the outer log often says only that an inner run failed. Go to
   the inner run's own output (below).
3. **Reproduce small.** Add `<debug/>` to `<Case>` with a `debug_value` on
   every swept or optimized capacity (`../heron/references/guide/modes.md`).
   Debug mode runs a single design point, one realization, one year, with
   plots, so failures show up in seconds.
4. **Look up the message.** Search `references/error-catalog.md` for a
   distinctive fragment of it (e.g. `grep -n "does not match" references/error-catalog.md`)
   rather than reading the whole file. Messages are abbreviated there, so
   search for a few plain words, not the full line with its values.
5. **Change one thing at a time.** Re-run after each fix, because one error often
   hides the next.

| Stage | Where it fails | Typical signs |
|-------|----------------|---------------|
| 1. HERON parsing | `heron input.xml` (before any RAVEN file is written) | `IOError`, `ValueError`, `KeyError` naming a node, component, or data generator; unknown-node messages from RAVEN's input checker |
| 2. Workflow generation | still `heron input.xml`, after parsing | "Mixing ARMA and CSV sources...", "raven_framework not in PATH", messages from `templates/` |
| 3. RAVEN outer | `raven_framework outer.xml` | Fails at start; cannot find `inner.xml` or `heron.lib`, bad paths, executable not found |
| 4. RAVEN inner | One design point fails inside `<Case>_i` | Outer reports a failed run with no cause. Look in the inner run's folder |
| 5. Dispatch | Inside the inner run, the HERON `DispatchManager` | "Solve was unsuccessful", "Requested number of steps...", "Expected ARMA variable..." |
| 6. Output | Run finishes, results look wrong | No error; see "Suspicious results" |

`heron input.xml` succeeding only means stages 1 and 2 passed. It does not
run the analysis.

## Where to look

| What | Where |
|------|-------|
| Generated workflows | `outer.xml`, `inner.xml`, `heron.lib` next to the HERON input. Read them to see what HERON asked RAVEN to do. Never hand-edit them; fix the input and regenerate. |
| Outer results | `<Case name>_o/`: `sweep.csv`, or `opt_soln.csv` and `opt_soln_<trajID>.csv` |
| One inner run | `<Case name>_o/<prefix>/`, one subfolder per design point (below) |
| Debug-mode output | `<Case name>_o/`; file names in `../heron/references/guide/results.md`, "Reading debug output" |
| Infeasible solve | The console output of that run (`references/infeasible-solves.md`) |

HERON's own dispatch messages are printed with a `DEBUGG` prefix: the solver
used, each solve attempt, its solve time, and any validation concerns.
`<verbosity>debug</verbosity>` in `<Case>` adds more RAVEN messages.

### Inner runs

The outer run launches `inner.xml` once per design point, each in its own
subfolder of `<Case name>_o/`, named by the run's prefix: `1`, `2`, ..., or
`b<batch>_r<run>` when runs are batched in parallel. In that subfolder:

- `generalOut` is the inner run's console output. **The real error of a
  failed inner run is here.**
- `<Case name>_i/` is the inner run's working directory.
- `inner.xml` is the inner workflow with this design point's capacities
  written in as `<constant>` values in its sampler.

RAVEN marks a run as failed when `<Case name>_i/.ravenStatus` is missing,
printing "Could not find ... .ravenStatus, assuming failed RAVEN run". If
the files are not where described, list the `_o` directory and trust what is
on disk.

**To rerun one failed design point by itself,** run RAVEN on the inner input
in that subfolder:

```bash
cd <Case name>_o/<prefix>
raven_framework inner.xml
```

It finds `heron.lib` (and any `<Function>` file) by the relative paths in its
`<Files>` block, so leave the directory layout as it is. If it cannot find
them, compare those paths with what is on disk. Setting
`<data_handling><inner_to_outer>csv</inner_to_outer></data_handling>` in
`<Case>` (and regenerating) makes the inner results CSV files you can read
directly.

### Solver failures and `<Function>` errors

- An infeasible, unbounded, or failed solve: read
  `references/infeasible-solves.md` for what HERON prints and how to read it.
- A `<Function>` that errors or misbehaves inside the dispatch: read
  `references/function-debugging.md`.

## Suspicious results (no error)

Check in this order, and stop at the first problem:

1. **Debug plots first.** Look at the dispatch plot for one design point.
   Does each component's activity have the expected sign, size, and daily shape?
   Is storage cycling? Is anything pinned at zero or at capacity all the time?
2. **Signs.** Revenue showing as cost, or the reverse, is the most frequent
   cause of a wrong NPV. Check each cash flow against the sign table in
   `../heron/references/spec/economics.md`.
3. **Connections.** Misspelled resources create separate networks. Check
   `network.png`.
4. **Units.** Check that capacities, prices, and time steps are consistent,
   and that `ProjectTime`, `lifetime`, and the discount rate are in years.
5. **Data.** Compare the plotted signal against the training data. Look for
   a flat or constant series. To check the synthetic history model itself,
   use the heron-tsa skill (`references/validation.md`).
6. **Tag and mode.** `sweep_values` vs. `opt_bounds` against `<mode>`, and
   `debug_value` present.
7. **Sample size.** Noisy or non-repeatable results: raise `<num_arma_samples>`
   (`../heron/references/guide/uncertainty.md`).
8. **Capacity at a bound.** An optimum on the edge of `opt_bounds` means the
   bound is limiting, not that the design is optimal. Widen it.

Then search `references/error-catalog.md`, "Suspicious results", for the
symptom. To understand why the dispatch made a choice, see
`../heron/references/theory/dispatch.md`.

## References

| File | Read when |
|------|-----------|
| `references/error-catalog.md` | You have a specific error message or symptom (search it; do not read it whole) |
| `references/infeasible-solves.md` | A dispatch solve failed: infeasible, unbounded, or other non-optimal status |
| `references/function-debugging.md` | A `<Function>` value or storage `<strategy>` fails or gives wrong values |
| `../heron/references/guide/results.md` | You need to read outputs correctly before deciding they are wrong |
| `../heron/references/theory/` | You need the math behind a dispatch or economic result |

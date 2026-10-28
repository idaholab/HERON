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
   every swept or optimized capacity (see the heron skill, `references/modes.md`).
   Debug mode runs a single design point, one realization, one year, with
   plots, so failures show up in seconds.
4. **Look up the message** in `references/error-catalog.md`.
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
| Generated workflows | Next to the HERON input: `outer.xml`, `inner.xml`, `heron.lib`. Open them to see what HERON actually asked RAVEN to do (sampled variables, aliases, statistics). Never hand-edit them to "fix" a study; fix the HERON input and regenerate. |
| Outer run output | `<Case name>_o/` (`sweep.csv`, or `opt_soln_<trajID>.csv` with `opt_soln.csv` as an index of them) |
| Inner run output | The outer run launches `inner.xml` once per design point through RAVEN's code interface. RAVEN makes one subfolder per run inside `<Case name>_o/`, named by the run's prefix (`1`, `2`, ... or `b<batch>_r<run>` when runs are batched). The inner run's console output is the file `generalOut` in that subfolder, and its own working directory (`<Case name>_i`) is nested inside. Read `generalOut` of a failed run for the real error |
| How RAVEN decides an inner run failed | It looks for `.ravenStatus` in the inner working directory and, if it is missing, prints "RAVENInterface WARNING: Could not find ... .ravenStatus, assuming failed RAVEN run." So that warning means the inner run died; the cause is in `generalOut` |
| Debug mode output | In `<Case name>_o/`: `dispatch_print.csv` (dispatch and signals), `cashflows.csv` plus one `cashflows_<id>.csv` per realization (RAVEN writes an index file and per-history files for a HistorySet), `dispatch.nc` (NetCDF database), the dispatch plots (`dispatch_id<sample>_y<year>_c<cluster>_f<figure>.png`, plus `dispatch_id..._SIGNAL.png` for the signals) and the cash flow plots (`Cash_Flows_from_Project_Year_0_to_<N>.png`, `Cumulative_from_Project_Year_0_to_<N>.png`, `Cumulative_pie_from_Project_Year_0_to_<N>.png`). `network.png` is written next to the input file when `heron` parses a debug case. These names are the ones HERON's own debug-mode tests expect (`tests/integration_tests/mechanics/debug_mode/tests`, `test_cashplot/tests`) |
| Solver failures | `constraint_violations.log` in the working directory of the failed run, listing violated constraints; the console output also prints the resource map and Pyomo model |

**Verification status of this table.** The debug-mode and sweep output names
match what HERON's regression tests expect. The inner-run subfolder layout
and the `generalOut` file come from RAVEN's source (`Code.py`,
`RAVENInterface.py`), not from a captured run. If a name or location here
does not match what you see on disk, trust the disk and list the directory.

To see the inner failure text, run the inner workflow directly on one
design point. Set `<data_handling><inner_to_outer>csv</inner_to_outer></data_handling>`
in `<Case>` to make inner results easy to read (`input/case.md` in the heron
skill). Increase `<verbosity>` to `debug` for more messages.

HERON's own dispatch solver messages are printed with a `DEBUGG` prefix. They show
the solver used, each solve attempt, and any validation concerns.

## Suspicious results (no error)

Check in this order, and stop at the first problem:

1. **Debug plots first.** Look at the dispatch plot for one design point.
   Does each component's activity have the expected sign, size, and daily shape?
   Is storage cycling? Is anything pinned at zero or at capacity all the time?
2. **Signs.** Revenue showing as cost, or the reverse, is the most frequent
   cause of a wrong NPV. Compare each cash flow's driver sign and price sign
   against the table in the heron skill's `references/input/economics.md`.
3. **Connections.** Misspelled resources create separate networks. Check
   `network.png`.
4. **Units.** Price per MWh against activity in MW over hourly steps works. Check
   that capacities, prices, and time steps use consistent units, and that
   `ProjectTime`, `lifetime`, and the discount rate are in the same unit (years).
5. **Data.** Is the signal what you expect? Compare the plotted signal against
   the training data. Look for a flat or constant series.
6. **Tag and mode.** `sweep_values` vs. `opt_bounds` against `<mode>`, and
   `debug_value` present.
7. **Sample size.** Noisy or non-repeatable results: raise `<num_arma_samples>`
   (`uncertainty.md` in the heron skill).
8. **Capacity at a bound.** An optimum on the edge of `opt_bounds` means the
   bound is limiting, not that the design is optimal. Widen it.

Then check the symptoms in `references/error-catalog.md` under "Suspicious
results".

## References

| File | Read when |
|------|-----------|
| `references/error-catalog.md` | You have a specific error message or symptom |
| the heron skill's `references/results.md` | You need to read outputs correctly before deciding they are wrong |

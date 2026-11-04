# First debug run checklist

Phase 5. The goal is to show the model runs and behaves physically, not to
get an answer. Output file names and contents are in
`../heron/references/guide/results.md` ("Reading debug output").

## Before running

- [ ] `<debug/>` is in `<Case>`.
- [ ] Every `sweep_values` / `opt_bounds` has a `debug_value`. Without one,
      debug mode samples the capacity and the run is not a design you chose.
- [ ] `<inner_samples>` and `<macro_steps>` are left at 1 (the defaults)
      unless there is a reason to look at more.
- [ ] Every component reference to a generator (`<ARMA>synth</ARMA>`)
      matches a `name` in `<DataGenerators>`, and every `variable` is one
      the generator lists.

## Running

```bash
heron input.xml                 # writes outer.xml (and heron.lib); prints the parse checks
raven_framework outer.xml       # runs the debug case
```

Read the `heron` output before running RAVEN. For each generator it prints
"detected: interpolated / clustered" (or "NOT interpolated", "NOT
clustered"). Confirm that matches what `heron inspect` reported in phase 4.

If either command fails, switch to the heron-debug skill.

## What to check, in order

Go through these with the user and record anything unexpected. Fix one
thing at a time and rerun.

### 1. Resource graph (`network.png`, next to the input)

- [ ] It matches the graph agreed in phase 2.
- [ ] No component stands alone, and no resource has only a producer or
      only a consumer.

### 2. Dispatch (`dispatch_id*_y*_c*_f*.png`, `dispatch_print.csv`)

For each component:

- [ ] The sign is right: producers positive, demands and consumed resources
      negative.
- [ ] The size is right: activity stays within capacity (times capacity
      factor), and above the minimum level if one is set.
- [ ] `fixed` components sit at their available capacity at every step.
- [ ] `independent` components respond to price: generators run when the
      price covers their variable cost; storage charges at low prices and
      discharges at high ones.
- [ ] Transfer ratios hold: consumed over produced matches the `<ratio>`.
- [ ] The signal plot (`*_SIGNAL.png`) has the range and shape the user
      expects for that data (units, peaks, daily pattern).

Everything idle or flat at zero usually means a sign error in a price (the
dispatcher sees selling as a cost) or a capacity of the wrong sign.

### 3. Cash flows (`cashflows_<id>.csv`, `Cash_Flows_*.png`)

Columns are `<component>_<cashflow>_CashFlow`, by `cfYears` (year 0 is
construction).

For every cash flow in the decision record:

- [ ] The sign matches what was decided in phase 3: revenues positive,
      costs negative. A market sale that is negative is the classic sign
      error (missing `<multiplier>-1</multiplier>` on the driver).
- [ ] Capital costs appear in year 0 (and again at each replacement, per
      `<lifetime>`), and nowhere else.
- [ ] Yearly cash flows appear in every operating year.
- [ ] The size is plausible. Estimate one by hand, e.g. fixed O&M = price ×
      capacity, or sales ≈ average price × average dispatch × steps per year
      (× cluster multiplicity, for a clustered ROM), and compare.

### 4. The metric

- [ ] The order of magnitude of NPV (or the chosen metric) is plausible:
      roughly the yearly net cash flow times the project life, less capex.
- [ ] It is not NaN or extreme. If it is, go to heron-debug.

Do not quote this number. One realization, one year, and one design point
are not a result.

## After a clean debug run

1. Remove `<debug/>`.
2. Start with a coarse grid (sweep) or loose tolerances (opt) and a few
   `<num_arma_samples>`, as in `../heron/SKILL.md` ("Recommended workflow").
3. Before quoting results, use `../heron/references/guide/results.md`
   ("Checklist before reporting a number").

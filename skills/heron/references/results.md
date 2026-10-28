# Interpreting results

What HERON writes, what each column means, and how to read it. For naming of
statistics see `uncertainty.md`; for the output file names by mode see
`modes.md`.

## Files

| Mode | File (in `<Case name>_o/`) | One row per |
|------|----------------------------|-------------|
| sweep | `sweep.csv` | design point (a combination of `sweep_values`) |
| opt | `opt_soln.csv` (an index: columns `trajID`, `filename`) and `opt_soln_<trajID>.csv` (the results) | optimizer iteration, in `opt_soln_<trajID>.csv` |
| debug | dispatch CSV, cash flow CSV, plots, `dispatch` NetCDF database | realization (dispatch) or project year (cash flows) |

Extra columns you may see in a sweep, such as `ProbabilityWeight*`,
`PointProbability`, and `prefix`, are RAVEN bookkeeping. Ignore them.

## Column groups in `sweep.csv` / `opt_soln*.csv`

(In `opt_soln_<trajID>.csv` the columns are `iteration`, `accepted`, then the
groups below. There is no `trajID` column in it.)

1. **Design variables.** One `<component>_capacity` column per capacity that
   was swept, optimized, or fixed (`npp_capacity`, ...). Sign follows the
   capacity sign convention: a demand's capacity is negative, as are
   capacities measured in a consumed resource (`input/economics.md`).
2. **Economic metric statistics.** `<stat>_<metric>` columns for NPV (and IRR,
   PI, or LC if requested). Examples: `mean_NPV`, `std_NPV`, `med_NPV`,
   `perc_5_NPV`.
3. **Total activity statistics.** `<stat>_TotalActivity__<component>__<tracker>__<resource>`.
4. **Optimization bookkeeping** (opt only): `iteration` and `accepted`
   (see below).

### Total activity

`TotalActivity__<component>__<tracker>__<resource>` is the activity of that
component and resource integrated over the time grid (trapezoid rule, in
resource units times time units), weighted by how many segments each cluster
represents, and then summed over all project years. It is not discounted.
Check the units against your time step: hourly steps of MW give MWh.

| Part | Values |
|------|--------|
| `<tracker>` | `production` for producers and demands; `level`, `charge`, `discharge` for storage |
| sign | Follows the activity sign convention: positive adds to the system, negative removes. A market that takes electricity has a negative total. |

Storage `level` totals are integrals of the stored amount, not a throughput.
Use `charge` and `discharge` for energy moved.

### Economic metrics

- **NPV** is the net present value of all component cash flows over the
  project (`<ProjectTime>` years plus the construction year), discounted at
  `<DiscountRate>`, in the currency units of your prices. It is the sum of the
  hourly and yearly cash flows after tax and inflation.
- **IRR** is the discount rate that makes NPV zero. It does not exist when
  there is no sign change in the yearly net cash flow.
- **PI** is the profitability index, computed by TEAL as `-NPV / FCFF[0]`,
  where `FCFF[0]` is the free cash flow in the construction year (the
  up-front investment, a negative number), so PI is positive when NPV is
  positive and it scales with the investment.
- **LC** (column `LC_Mult`) is the multiplier on the levelized cost cash flow's
  price that sets NPV to the target. Read it as "multiply the price you gave by
  this to break even". It needs a nonlinear solver (`input/case.md`).

## Reading a sweep

1. Sort by `mean_NPV` (or `med_NPV`, which is less sensitive to outliers).
   Both are always written (`uncertainty.md`). In a flat sweep, sort by `NPV`.
2. Look for the shape: where does NPV rise, flatten, and fall as each
   capacity changes? An edge optimum (best at the lowest or highest swept
   value) means the grid should be widened.
3. Compare spread against differences: if `std_NPV` is larger than the NPV
   difference between two designs, the data cannot separate them.
4. Check the total activity columns for design points that look odd: an
   unused component has near-zero total activity, which usually means
   its cost exceeds its revenue.

## Reading an optimization

- The `accepted` column is a label with these values (seen in the test gold
  files): `first` (the starting point), `accepted` (a step the optimizer took),
  `rejected` (evaluated, not taken), `rerun` (an `accepted` point re-evaluated
  by the optimizer), and `final` (the reported solution, repeating the last
  accepted point).
- The `final` row is the optimizer's result. Also look at how much the
  objective changed over the last several `accepted` rows, to judge convergence.
- Compare the result against the bounds. A capacity at a bound suggests the
  bound constrains the design.
- With several trajectories (`opt_soln_<trajID>.csv`), check that they agree.
  Different optima suggest a flat or noisy objective.
- Confirm by sweeping near the optimum (`modes.md`).

## Reading debug output

| Output | What to read |
|--------|--------------|
| Dispatch plot | One panel per component or resource. Positive means supplied, negative means consumed. Check the shape against prices and demand. Legends distinguish the components. |
| Signal plot (`*_SIGNAL.png`) | The time series sampled for the realization (price, load, capacity factor). |
| Cash flow plots (`Cash_Flows_from_Project_Year_0_to_<N>.png`, `Cumulative_...png`, `Cumulative_pie_...png`) | Cash flows by component and year, their running total, and a breakdown pie. Look at size and sign. |
| `dispatch_print.csv` | Dispatch variables named `Dispatch__<component>__<tracker>__<resource>`, indexed by time, year, and cluster. Use it to check exact values. |
| `cashflows.csv`, `cashflows_<id>.csv` | One column per cash flow, named `<component>_<cashflow>_CashFlow` (depreciation columns keep their own names), indexed by `cfYears`. |

`cfYears` is the index `0, 1, 2, ..., ProjectTime`
(`DispatchManager.save_variables`). TEAL stores the project length as
`ProjectTime + 1` ("one for the construction year") and sizes every cash flow
array to it, so each array has `ProjectTime + 1` entries. Year 0 is the
construction year, where a component's first `one-time` cash flow (capex) is
charged, and years `1..ProjectTime` are operating years. In debug mode
`ProjectTime` is replaced by `<macro_steps>`.

## Checklist before reporting a number

- Capacities and sign conventions are what you intended.
- The dispatch plot is physically sensible for at least one realization.
- The number of realizations is large enough for the statistic quoted.
- Each reported optimum is interior to its bounds and confirmed by a sweep.
- Units of price, capacity, and time step are stated alongside the number.
- A capacity column of `-42` means the inner run was not given a capacity by
  the outer; that is a placeholder, not a design value.

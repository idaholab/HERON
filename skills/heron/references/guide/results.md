# Interpreting results

What HERON writes, what each column means, and how to read it. For naming of
statistics see `references/guide/uncertainty.md`; for the output file names by mode see
`references/guide/modes.md`.

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
   was swept, optimized, or fixed (`npp_capacity`, ...), signed as in the
   input (`references/spec/economics.md`).
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

- **NPV**: all component cash flows over `<ProjectTime>` years plus the
  construction year, after tax and inflation, discounted at
  `<DiscountRate>`, in the currency of your prices.
- **IRR**: the discount rate that makes NPV zero. It does not exist without
  a sign change in the yearly net cash flow.
- **PI**: `-NPV / FCFF[0]`, NPV per unit of up-front investment. Positive
  when NPV is.
- **LC** (column `LC_Mult`): the multiplier on the `<levelized_cost/>` cash
  flow's price that brings NPV to the target. Read it as "multiply the price
  you gave by this to break even".

The formulas are in `references/theory/economics.md`.

## Reading a sweep

1. Sort by `mean_NPV` (or `med_NPV`, which is less sensitive to outliers).
   Both are always written (`references/guide/uncertainty.md`). In a flat sweep, sort by `NPV`.
2. Look for the shape: where does NPV rise, flatten, and fall as each
   capacity changes? An edge optimum (best at the lowest or highest swept
   value) means the grid should be widened.
3. Compare spread against differences: if `std_NPV` is larger than the NPV
   difference between two designs, the data cannot separate them.
4. Check the total activity columns for design points that look odd: an
   unused component has near-zero total activity, which usually means
   its cost exceeds its revenue.

## Reading an optimization

- The `accepted` column is a label: `first` (the starting point), `accepted` (a step the optimizer took),
  `rejected` (evaluated, not taken), `rerun` (an `accepted` point re-evaluated
  by the optimizer), and `final` (the reported solution, repeating the last
  accepted point).
- The `final` row is the optimizer's result. Also look at how much the
  objective changed over the last several `accepted` rows, to judge convergence.
- Compare the result against the bounds. A capacity at a bound suggests the
  bound constrains the design.
- With several trajectories (`opt_soln_<trajID>.csv`), check that they agree.
  Different optima suggest a flat or noisy objective.
- Confirm by sweeping near the optimum (`references/guide/modes.md`).

## Reading debug output

| Output | What to read |
|--------|--------------|
| Dispatch plot | One panel per component or resource. Positive means supplied, negative means consumed. Check the shape against prices and demand. Legends distinguish the components. |
| Signal plot (`*_SIGNAL.png`) | The time series sampled for the realization (price, load, capacity factor). |
| Cash flow plots (`Cash_Flows_from_Project_Year_0_to_<N>.png`, `Cumulative_...png`, `Cumulative_pie_...png`) | Cash flows by component and year, their running total, and a breakdown pie. Look at size and sign. |
| `dispatch_print.csv` | Dispatch variables named `Dispatch__<component>__<tracker>__<resource>`, indexed by time, year, and cluster. Use it to check exact values. |
| `cashflows.csv`, `cashflows_<id>.csv` | One column per cash flow, named `<component>_<cashflow>_CashFlow` (depreciation columns keep their own names), indexed by `cfYears`. |

`cfYears` runs `0, 1, ..., ProjectTime`. Year 0 is the construction year,
where capex is charged, and years `1..ProjectTime` are operating years
(`references/theory/economics.md`). In debug mode `ProjectTime` is replaced
by `<macro_steps>`.

To plot or query the debug dispatch yourself, open the NetCDF database with
xarray, from the RAVEN environment:

```python
import xarray as xr
ds = xr.open_dataset('<Case name>_o/dispatch.nc')
print(ds)    # variables Dispatch__<comp>__<tracker>__<res>, signals, indices
```

Each `Dispatch__...` variable and each signal is indexed by
`(RAVEN_sample_ID, <time_variable>, <year_variable>, _ROM_Cluster)`, with
`_ROM_Cluster` of length 1 for an unclustered model. For example
`ds['Dispatch__gen__production__electricity'].isel(RAVEN_sample_ID=0, _ROM_Cluster=0)`
is a time by year array for the first realization.

## Checklist before reporting a number

- Capacities and sign conventions are what you intended.
- The dispatch plot is physically sensible for at least one realization.
- The number of realizations is large enough for the statistic quoted.
- Each reported optimum is interior to its bounds and confirmed by a sweep.
- Units of price, capacity, and time step are stated alongside the number.
- A capacity column of `-42` means the inner run was not given a capacity by
  the outer; that is a placeholder, not a design value.

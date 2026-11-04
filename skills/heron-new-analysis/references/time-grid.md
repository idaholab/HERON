# Setting the time grid from the data

Phase 4 turns each data file into a `<DataGenerators>` entry and makes
`<Case>` agree with it. HERON does not interpolate: the dispatch time grid
must match the data's time axis exactly, or the run fails with "Requested
number of steps ... does not match".

## 1. Inspect each file

The `heron` launcher activates the RAVEN environment itself:

```bash
heron inspect path/to/arma.pk
heron inspect path/to/prices.csv
heron inspect prices.csv --time-variable HOUR --year-variable YEAR   # if the CSV's columns are not named time/hour and year
```

It prints:

| Line | Use it for |
|------|-----------|
| `Type` | Which generator node: `<ARMA>` for a synthetic history ROM, `<CSV>` for a CSV, `<ROM>` for any other ROM |
| `Signals` | Allowed names in the generator's `variable` attribute, and in each component's `<ARMA variable="...">` / `<CSV variable="...">` |
| `Time variable`, `Years (...)` | `<time_variable>` and `<year_variable>` in `<time_discretization>` |
| `Years` | The limit on `<ProjectTime>` (below) |
| `Time axis` | Number of steps, first and last value, and interval of one dispatch window |
| `Clustered`, `Clusters dispatched per year` | Whether the ROM is clustered and how many representative windows are dispatched |
| `Matching <Case> settings` | A `<time_discretization>` to paste into `<Case>` |

For a synthetic history, a clustered ROM is reported once for each
`evalMode`, because the time axis is different in each:

- `clustered` (the default): one dispatch window is one cluster
  (e.g. 24 hours), and only the representative clusters are dispatched.
  Fast. Means are reliable; spreads are inflated
  (`../heron/references/guide/time-series.md`, "Clustering and higher moments").
- `full`: one dispatch window is the whole year (e.g. 8760 hours). Slower,
  with one large dispatch per year.

Use `clustered` unless the user needs spread or tail statistics. If `full`
is chosen, set `evalMode="full"` on the `<ARMA>` node and use the `full`
time discretization.

Sampling a ROM takes a few seconds to a minute. `heron inspect` reads
nothing but the file; it does not need a HERON input.

## 2. Fill in the nodes

```xml
<Case name="...">
  ...
  <time_discretization>        <!-- from "Matching <Case> settings" -->
    <time_variable>HOUR</time_variable>
    <year_variable>YEAR</year_variable>
    <start_time>0</start_time>
    <end_time>23</end_time>
    <num_steps>24</num_steps>
  </time_discretization>
  <economics>
    <ProjectTime>3</ProjectTime>   <!-- within the limit from "Years" -->
    ...
</Case>
...
<DataGenerators>
  <ARMA name="synth" variable="TOTALLOAD,WIND">path/to/arma.pk</ARMA>   <!-- from "Signals" -->
</DataGenerators>
```

Only list the signals the components use in `variable`. If several files
are used, they must all be ARMA or all CSV, every file must report the
same time variable, year variable, and time axis, and no signal name may
appear in two generators.

## 3. Check the project life

| Report says | Meaning | `<ProjectTime>` |
|-------------|---------|-----------------|
| `Years (YEAR): N, first to last` | Multi-year data. Each project year uses its own data year, in order; extra data years are dropped. | At most N. Longer is a parse-time error ("interpolates N macro steps, but project life is M"). |
| `Years: one cycle of data` | One year, repeated for every project year. | Any |

A CSV with a year column is treated as multi-year even if it holds only one
year, so its `<ProjectTime>` is limited to the years it contains.

If the user wants a longer project than the data covers, the options are a
shorter `<ProjectTime>`, retraining the ROM over more years, or a single-year
model that is repeated. Ask which; do not pick one silently.

## 4. Check the units

The inspection gives the time step in the data's own units (e.g. interval
`1` on `HOUR`). Confirm with the user what one step is in real time, and
that prices and capacities agree with it: a price in $/MWh with activity in
MW over hourly steps is consistent; the same price over 15-minute steps is
not, unless the cash flow is scaled.

## Common problems

| Report or error | Cause | Fix |
|-----------------|-------|-----|
| `Type: RAVEN ROM (not a synthetic history)` | The file is a regular ROM | Use a `<ROM>` generator, or get the synthetic history `.pk` |
| `interval not uniform` | Uneven time steps in the data | HERON needs uniform steps; resample the data or retrain |
| "Could not identify the time column" | A CSV whose time column is not named time or hour | Pass `--time-variable` |
| "has no RAVEN_sample_ID column" | CSV missing a required column | Add a constant `RAVEN_sample_ID` column (e.g. `0`) |
| "must have the same number of time steps" | A CSV whose years or samples have different lengths | Make every year the same length |

# Training data

RAVEN reads training histories into a `HistorySet` from an index CSV that
points to one history file per row.

## Layout

```text
Data.csv                         Data_2020.csv
scaling,Year,filename            Time,PRICE,LOAD
1,2020,Data_2020.csv             0,21.7,1050.2
1,2025,Data_2025.csv             1,22.2,1012.8
1,2030,Data_2030.csv             2,20.9,987.4
                                 ...
```

- **Index file** (`Data.csv`): one row per history. Its columns are the
  `HistorySet` inputs and a `filename` column. Paths in `filename` are
  relative to the index file.
- **History files**: the pivot (time) column and one column per signal, in
  any order. Any other column is ignored unless it is listed as an output.
- `scaling` is a constant 1. HERON evaluates every synthetic history model
  with `scaling = 1`, so the ROM must have `<Features>scaling</Features>`
  and the training data must provide it.

The matching `HistorySet`:

```xml
<HistorySet name="training">
  <Input>scaling, Year</Input>
  <Output>PRICE, LOAD, Time</Output>
  <options><pivotParameter>Time</pivotParameter></options>
</HistorySet>
```

loaded with an `IOStep` from a `<Files>` entry for `Data.csv`.

## Single year or multi-year

| Data | Index file | ROM |
|------|------------|-----|
| One representative year | One row; no year column. `Input` is just `scaling`. | No `<macroParameter>`. HERON repeats the year for every project year. |
| Several years | One row per year, with an integer year column | `<Segment grouping="interpolate">` with `<macroParameter>` naming the year column. Each listed year is trained separately; every year between the first and last is filled in by interpolating the trained parameters. |

For a multi-year model the years must be integers, and the span from first
to last listed year must cover the HERON project life (`<ProjectTime>`,
less the construction year). Years do not need to be evenly spaced. HERON
uses as many years as the project life and drops the rest.

## Requirements on the histories

- **Uniform time steps** in every file. HERON's dispatch grid is uniform,
  and it does not interpolate.
- **The same length and pivot values** in every file. Interpolated models
  assume the pivot is the same each year.
- **Length divisible by the segment length** when segmenting (e.g. 8760
  hours into 365 days of 24). RAVEN refuses unequal last segments. Drop
  February 29 rather than keeping 8784 hours.
- **No missing values** unless every algorithm before the gap can accept
  NaN (`references/algorithms.md`). Fill gaps before training.
- **Signals sampled together** must share one file per year. Signals that
  need different segmenting or clustering need separate ROMs, which HERON
  loads as separate `<ARMA>` generators.

## Names

The names chosen here carry through to the HERON input:

| Here | In HERON |
|------|----------|
| Pivot column (`Time`) | `<time_variable>` in `<Case>` (HERON's default is `time`, lowercase) |
| Year column (`Year`) | `<year_variable>` in `<Case>` (default `Year`) |
| Signal columns | `variable="..."` on the `<ARMA>` generator and in each component's `<ARMA>` value |

`heron inspect model.pk` reports all three after training.

## Pivot units

The pivot values set the units of every period and length in the ROM:
Fourier `periods`, `subspace pivotLength`, STL `period`. With an hourly
pivot counted 0, 1, 2, ... a daily cycle has period 24 and a daily segment
has `pivotLength="24"`. With a pivot in seconds they are 86400. Counting
the pivot in time steps is the simplest choice.

HERON's `<time_discretization>` must reproduce the pivot values the model
returns (`<end_time>` and `<num_steps>`), which `heron inspect` prints.

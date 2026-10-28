# `<DataGenerators>` node

Declares the external data sources that components refer to. Every source
has a `name`, and components refer to it by putting that name in the text of
a ValuedParam (see `valued-params.md`). The node is required, but can be
empty if nothing uses external data.

```xml
<DataGenerators>
  <ARMA name="synth" variable="PRICE,LOAD">%HERON_DATA%/TSA/my_arma.pk</ARMA>
  <CSV name="prices" variable="price">price.csv</CSV>
  <Function name="funcs">functions.py</Function>
  <ROM name="myROM">my_rom.pk</ROM>
</DataGenerators>
```

## Source types

| Node | Text | Attributes |
|------|------|------------|
| `<ARMA>` | Path to a trained, serialized RAVEN synthetic history ROM (`.pk`). | `name` (required), `variable` (required: comma-separated signal names to use), `evalMode` (optional RAVEN sampling mode, default `clustered`) |
| `<CSV>` | Path to a static history CSV. | `name` (required), `variable` (required: comma-separated column names to use) |
| `<Function>` | Path to a Python file. | `name` (required) |
| `<ROM>` | Path to a trained, serialized RAVEN ROM (`.pk`). | `name` (required) |

### `<ARMA>`

The model must be trained in RAVEN beforehand (see `../time-series.md`).
Each realization HERON draws is a new synthetic history, so the number of
realizations is set by `<num_arma_samples>` in `<Case>`. The `variable`
names must match the signal names the model was trained on.

### `<CSV>`

A fixed history, used the same way as an ARMA signal. The file must have:

- a header row with column names
- a `RAVEN_sample_ID` column (a constant such as `0` is fine)
- columns for the time and year variables, named as in `<Case>`
  `<time_discretization>`
- one column for each signal listed in `variable`

```
RAVEN_sample_ID,Year,Time,price
0,2020,0,21.5
0,2020,1,19.8
```

### `<Function>`

The Python file holds the methods named in `<Function method="...">` value
slots and in storage `<strategy>`. See `functions.md` for how to write them.

## Paths

Paths are tried in this order:

1. If the path starts with `%HERON%`: relative to the HERON installation.
2. If it starts with `%HERON_DATA%`: relative to the `HERON_DATA`
   environment variable if it is set, otherwise HERON's `data/` directory.
3. Otherwise: relative to the input file's directory, then as an absolute
   path.

## Rules

- **Unique names.** Each component reference must match exactly one source
  `name` of the right type.
- **Unique variables.** No two sources may list the same `variable`;
  duplicates are an error.
- **Consistent time and year names.** The `time_variable` and
  `year_variable` in `<Case>` must match the names used in the ARMA training
  data or the CSV columns.

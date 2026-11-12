# ValuedParams: supplying values

Any node that takes a value, such as `<capacity>`, `<reference_price>`,
`<driver>`, or `<initial_stored>`, holds exactly **one** ValuedParam node
that says where the value comes from.

```xml
<capacity resource="electricity">
  <sweep_values>100, 200, 300</sweep_values>   <!-- the ValuedParam -->
  <multiplier>-1</multiplier>                  <!-- optional add-on -->
</capacity>
```

## Types

| Node | Value comes from | Example |
|------|------------------|---------|
| `<fixed_value>` | A constant. | `<fixed_value>100</fixed_value>` |
| `<sweep_values>` | A list of values to try, one grid point each (sweep mode). | `<sweep_values>10, 20, 30</sweep_values>` |
| `<opt_bounds>` | Lower and upper bounds for the optimizer (opt mode). | `<opt_bounds>10, 50</opt_bounds>` |
| `<ARMA>` | A synthetic history variable. The text is the `<DataGenerators>` name. | `<ARMA variable="PRICE">synth</ARMA>` |
| `<CSV>` | A static history variable. The text is the `<DataGenerators>` name. | `<CSV variable="price">prices</CSV>` |
| `<Function>` | A Python function. The text is the `<DataGenerators>` name, and `method` is the function name. | `<Function method="price">funcs</Function>` |
| `<ROM>` | A trained RAVEN ROM, evaluated with inputs you give (below). | |
| `<variable>` | A named workflow variable, such as `<component>_capacity` or a `dispatch_vars` name. | `<variable>npp_capacity</variable>` |
| `<activity>` | This component's dispatched activity of a resource. Cash flows only. | `<activity>electricity</activity>` |
| `<uncertainty>` | A random variable sampled from one of the supported distributions. Cash flows only (below). | (below) |

Both `<sweep_values>` and `<opt_bounds>` accept a `debug_value` attribute,
which is the single value used when `<Case>` has `<debug>` enabled.

Match the node to the mode: `<sweep_values>` for `<mode>sweep</mode>`, and
`<opt_bounds>` for `<mode>opt</mode>`. The workflow only checks whether a
value is a list, not which of the two tags was used. So `<sweep_values>` in
opt mode is quietly treated as bounds (its minimum and maximum), and
`<opt_bounds>` in sweep mode is treated as a two-point grid.

## Which types each slot accepts

| Slot | Allowed |
|------|---------|
| `<capacity>`, `<minimum>`, `<initial_stored>`, `dispatch_vars` `<variable>` | All except `<activity>` and `<uncertainty>` |
| `<capacity_factor>` | `<ARMA>`, `<CSV>` only |
| Storage `<strategy>` | `<Function>` only |
| CashFlow `<driver>`, `<reference_price>`, `<reference_driver>`, `<scaling_factor_x>` | All, including `<activity>`. `<reference_price>` also accepts `<levelized_cost/>`. |
| ROM `<input>` | `<fixed_value>`, `<sweep_values>`, `<opt_bounds>`, `<variable>`, `<ARMA>`, `<Function>` |

## Optional add-ons

These can appear next to any ValuedParam:

- `<multiplier>`: multiplies the value. Default 1. Commonly `-1`, to flip a
  sign (e.g. positive load data used as a negative demand capacity).
- `<AdditionalInfo>`: any XML you like, passed unchecked to a `<Function>`
  as `meta['HERON']['custom_input']`, a list of XML elements.

## Details

### `<activity>`

The text is a resource this component produces or consumes. The value is
positive for produced resources and negative for consumed ones. For storage,
the `tracking` attribute selects `level` (the default), `charge`, or
`discharge`.

### `<ROM>`

```xml
<reference_price>
  <ROM rom="myROM" variable="price">
    <input name="a"><fixed_value>0</fixed_value></input>
    <input name="b"><variable>npp_capacity</variable></input>
  </ROM>
</reference_price>
```

`rom` is the `<DataGenerators>` ROM name and `variable` is the ROM output to
use. There is one `<input>` per ROM input, with `name` matching the names
the ROM was trained with.

### `<uncertainty>`

Contains one distribution node, from the list below. The value is sampled
once per inner realization. The parser accepts `<uncertainty>` in any value
slot, but the workflow only samples it for CashFlow parameters (`driver`,
`reference_price`, `reference_driver`, `scaling_factor_x`), so use it only
there.

```xml
<reference_price>
  <uncertainty>
    <Normal name="price_dist">
      <mean>10</mean>
      <sigma>0.05</sigma>
    </Normal>
  </uncertainty>
</reference_price>
```

Every distribution needs a `name` attribute (HERON replaces it). Most also
accept optional `<lowerBound>`/`<upperBound>` to truncate, and a `verbosity`
attribute; Categorical and MarkovCategorical accept neither.

| Distribution | Parameters (required in bold) |
|--------------|-------------------------------|
| `<Uniform>` | **`lowerBound`**, **`upperBound`** |
| `<Normal>` | **`mean`**, **`sigma`** |
| `<LogNormal>` | **`mean`**, **`sigma`** (of the log), `low` (shift, default 0) |
| `<Triangular>` | **`min`**, **`max`**, **`apex`** (no truncation) |
| `<Beta>` | `low` (0), `high` (1), and either `alpha`+`beta` or `peakFactor` |
| `<Gamma>` | **`alpha`**, `beta` (rate, default 1), `low` (default 0) |
| `<Exponential>` | **`lambda`**, `low` (default 0) |
| `<Weibull>` | **`k`**, **`lambda`**, `low` (default 0) |
| `<LogUniform>` | **`lowerBound`**, **`upperBound`** (exponents), **`base`** (`natural` or `decimal`) |
| `<Logistic>` | **`location`**, **`scale`** |
| `<Laplace>` | **`location`**, **`scale`** |
| `<Custom1D>` | **`functionType`** (`pdf`/`cdf`), **`dataFilename`**, **`functionID`**, **`variableID`**, `workingDir` |
| `<Poisson>` | **`mu`** (no truncation) |
| `<Binomial>` | **`n`** (integer), **`p`** (no truncation) |
| `<Bernoulli>` | **`p`** (no truncation) |
| `<Geometric>` | **`p`** (no truncation) |
| `<UniformDiscrete>` | **`strategy`** (`withReplacement`/`withoutReplacement`), and either `values` or bounds with `nPoints` or `step` |
| `<Categorical>` | one or more **`<state outcome="...">probability</state>`**, then `rtol` (order matters) |
| `<MarkovCategorical>` | one or more **`<state outcome="..." index="..."/>`**, then **`transition`** (square matrix, row order), then `workingDir` (order matters) |

Multi-dimensional distributions (`MultivariateNormal`, `NDInverseWeight`,
`NDCartesianSpline`, `NDimensionalDistributions`) are not accepted: each
`<uncertainty>` is a single value, and HERON used to sample only their first
dimension.

### `<Function>`

A Python function supplies the value. The signature, return keys, the
`meta` contents available, and the restrictions that apply inside the
dispatch optimization are all in `references/spec/functions.md`.

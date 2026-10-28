# Choosing a ValuedParam

How to decide where a value should come from. The full list of types, the
slots each accepts, and the XML for each are in `input/valued-params.md`.

Every value slot (`<capacity>`, `<driver>`, `<reference_price>`, ...) holds
exactly one ValuedParam. Two in one slot is an error ("Only one ValuedParam
type can be used per node"), and none is an error ("expected a ValuedParam
to define its value source").

## Decision guide

| The value is... | Use | Notes |
|-----------------|-----|-------|
| A known constant | `<fixed_value>` | The default choice. |
| A design choice to explore on a grid | `<sweep_values>` | Sweep mode. Add `debug_value`. |
| A design choice to optimize | `<opt_bounds>` | Opt mode. Low, high. Add `debug_value`. |
| A time series from a trained stochastic model | `<ARMA variable="..">name</ARMA>` | Needs a trained model (`time-series.md`). |
| A time series from a file | `<CSV variable="..">name</CSV>` | Fixed history; no stochasticity. |
| A formula of other signals or activity | `<Function method="..">name</Function>` | See `input/functions.md`. Keep simple. |
| The output of a trained RAVEN ROM | `<ROM rom=".." variable="..">` | One `<input>` per ROM input. |
| Another named workflow value, such as a capacity | `<variable>npp_capacity</variable>` | Capex drivers. |
| This component's dispatched activity | `<activity>resource</activity>` | Cash flows only. |
| A random parameter, drawn once per realization | `<uncertainty>` with a RAVEN distribution | Cash flow parameters only. |

## Rules that catch people

- **Match the tag to the mode.** `sweep_values` in opt mode is quietly treated
  as bounds (its min and max), and `opt_bounds` in sweep mode as a two-point
  grid. HERON does not warn.
- **One value for a fixed unit.** A capacity you do not want to vary is
  `fixed_value`, not a one-entry `sweep_values`.
- **Sign conventions.** Demand capacities and consumed-resource capacities are
  negative. Use `<multiplier>-1</multiplier>` to flip positive data. The
  multiplier works with any ValuedParam.
- **ARMA and CSV cannot be mixed.** A case cannot have both `<ARMA>` and
  `<CSV>` sources (the template driver stops with "Mixing ARMA and CSV
  sources is not yet supported"). `<ARMA>` variables used in a value slot
  must be named in an `<ARMA>` generator, `<CSV>` in a `<CSV>` generator.
- **`capacity_factor` takes only `<ARMA>` or `<CSV>`.**
- **`<uncertainty>` is the only "uncertain cash flow parameter".** A
  `<driver>`, `<reference_price>`, `<reference_driver>`, or
  `<scaling_factor_x>` that is an `<uncertainty>` is sampled once per
  realization, and its presence forces the bilevel workflow even with a
  single static history (see `modes.md`). Time series (`<ARMA>`, `<CSV>`) in
  those slots vary over time but do not count as uncertain parameters in
  that check.
- **Functions run inside the optimization.** For hourly cash flows on the
  Pyomo dispatcher the function receives symbolic activity, so only simple
  arithmetic works. Details are in `input/functions.md`.
- **A name that does not exist is an error at parse time.** The text of
  `<ARMA>`, `<CSV>`, `<Function>`, and `<ROM>` values must equal a
  `name` in `<DataGenerators>` ("Requested source ... was not found").
- **Variable names must exist in the data.** A `<CSV>` generator whose
  `variable` is not a column header fails at parse time ("Variable ... requested
  for ... but not found in ..."). An `<ARMA>` `variable` that is not a signal in
  the trained model is only caught at run time ("Expected ARMA variable ... was
  not passed to DispatchManager"), so run debug mode first.

## Worked example

Wind farm sized by opt, with a capacity factor from a trained model and a
fixed O&M cost:

```xml
<Component name="wind">
  <produces resource="electricity" dispatch="fixed">
    <capacity resource="electricity">
      <opt_bounds debug_value="100">0, 500</opt_bounds>
    </capacity>
    <capacity_factor>
      <ARMA variable="WIND">synth</ARMA>
    </capacity_factor>
  </produces>
  <economics>
    <lifetime>30</lifetime>
    <CashFlow name="capex" type="one-time" taxable="True" inflation="none">
      <driver><variable>wind_capacity</variable></driver>
      <reference_price><fixed_value>-1.5e6</fixed_value></reference_price>
    </CashFlow>
    <CashFlow name="FOM" type="repeating" period="year" taxable="True" inflation="none">
      <driver><variable>wind_capacity</variable></driver>
      <reference_price><fixed_value>-4e4</fixed_value></reference_price>
    </CashFlow>
  </economics>
</Component>
```

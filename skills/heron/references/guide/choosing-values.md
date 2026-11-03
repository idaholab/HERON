# Choosing a ValuedParam

How to decide where a value should come from. The full list of types, the
slots each accepts, and the XML for each are in `references/spec/valued-params.md`.

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
| A time series from a trained stochastic model | `<ARMA variable="..">name</ARMA>` | Needs a trained model (`references/guide/time-series.md`). |
| A time series from a file | `<CSV variable="..">name</CSV>` | Fixed history; no stochasticity. |
| A formula of other signals or activity | `<Function method="..">name</Function>` | See `references/spec/functions.md`. Keep simple. |
| The output of a trained RAVEN ROM | `<ROM rom=".." variable="..">` | One `<input>` per ROM input. |
| Another named workflow value, such as a capacity | `<variable>npp_capacity</variable>` | Capex drivers. |
| This component's dispatched activity | `<activity>resource</activity>` | Cash flows only. |
| A random parameter, drawn once per realization | `<uncertainty>` with a RAVEN distribution | Cash flow parameters only. |

## Rules that catch people

- **Match the tag to the mode** (`references/spec/valued-params.md`).
- **One value for a fixed unit.** A capacity you do not want to vary is
  `fixed_value`, not a one-entry `sweep_values`.
- **Signs.** Use `<multiplier>-1</multiplier>` (it works with any
  ValuedParam) to flip positive data into a negative demand or consumed
  capacity (`references/spec/economics.md`).
- **One kind of time series per case:** all `<ARMA>` or all `<CSV>`
  (`references/guide/time-series.md`).
- **`capacity_factor` takes only `<ARMA>` or `<CSV>`.**
- **`<uncertainty>` is the only "uncertain cash flow parameter".** Its
  presence in a cash flow forces the bilevel workflow even with a single
  static history (`references/guide/modes.md`). Time series in a cash flow
  vary over time but do not count for that check.
- **Functions run inside the optimization,** on symbolic activity, so only
  simple arithmetic works (`references/spec/functions.md`).
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

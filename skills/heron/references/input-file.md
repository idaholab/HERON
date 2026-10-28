# HERON input file

The overall structure of a HERON XML input and the minimum each part needs.
For the full set of options, read the file for that node in `input/`.

## Structure

```xml
<HERON>
  <TestInfo>...</TestInfo>              <!-- regression tests only -->
  <Case name="...">...</Case>           <!-- required: how to solve -->
  <Components>...</Components>          <!-- required: at least one <Component> -->
  <DataGenerators>...</DataGenerators>  <!-- required, may be empty -->
</HERON>
```

HERON reads `<Case>` first, then each `<Component>`, then each data
generator. It then resolves every name a component uses to refer to a data
generator (for example `<ARMA variable="PRICE">synth</ARMA>` refers to the
generator with `name="synth"`). A name that does not match is an error.

## Minimum required

**`<Case name="...">`**
- `<mode>`: `sweep` or `opt`.
- `<time_discretization>` containing `<end_time>`, plus `<num_steps>` or
  `<time_interval>`.
- `<economics>` containing `<ProjectTime>`, `<DiscountRate>`, `<tax>`, and
  `<inflation>`.
- `<dispatcher>`: usually `<pyomo/>`.

**`<Component name="...">`**, one per unit, inside `<Components>`
- Exactly one of `<produces>`, `<stores>`, or `<demands>`, with the
  `resource` and `dispatch` attributes, and a `<capacity>`. For `dispatch`,
  use only `fixed` or `independent`.
- A `<produces>` that also has `<consumes>` needs a `<transfer>` function.
- `<economics>` containing `<lifetime>`. Cash flows are optional.
- Each `<CashFlow>` needs the attributes `name`, `type`, `taxable`, and
  `inflation`, and a `<driver>`.

**`<DataGenerators>`**
- Each entry needs a `name`. `<ARMA>` and `<CSV>` also need `variable`.

**Check the signs.** Activity is positive when a component adds a resource
and negative when it removes one, so a market (`<demands>`) has negative
activity, and its sales cash flow needs `<multiplier>-1</multiplier>` on the
driver to count as revenue. This is one of the easiest mistakes to make in
HERON; read the sign conventions in `input/economics.md` before writing any
cash flow.

**`<TestInfo>`**: only in HERON's own regression tests. It holds `<name>`,
`<author>`, `<created>`, `<description>`, and `<classesTested>`.

## Where to find more

| File | Read when |
|------|-----------|
| `input/case.md` | Setting the mode, time grid, project economics, dispatcher, sampling, debug, or optimization settings |
| `input/components.md` | Writing a `<Component>`: produces, stores, demands, capacity, transfer functions, ramping, storage options |
| `input/economics.md` | Writing a component's `<economics>` and `<CashFlow>` nodes |
| `input/valued-params.md` | Filling any value slot (`fixed_value`, `sweep_values`, `opt_bounds`, `ARMA`, `CSV`, `Function`, `activity`, ...) |
| `input/functions.md` | Writing the Python functions behind `<Function>` values or a storage `<strategy>` |
| `input/data-generators.md` | Declaring `<ARMA>`, `<CSV>`, `<Function>`, or `<ROM>` sources, and file paths |
| `input/test-info.md` | Writing a regression test input for HERON itself |

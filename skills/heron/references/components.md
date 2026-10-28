# Components: modeling guide

How to turn a physical system into HERON components. For every option and
default, see `input/components.md`; for cash flows see `input/economics.md`.

## Pick the interaction

Each `<Component>` has exactly one of `<produces>`, `<stores>`, `<demands>`
(a second one is an error: "each Component can only have one interaction").

| Physical thing | Interaction | Typical `dispatch` |
|----------------|-------------|--------------------|
| Generator, reactor, electrolyzer, import from a grid | `<produces>` | `fixed` for must-run, `independent` for flexible |
| Wind or solar | `<produces>` with `<capacity_factor>` | `fixed` (takes all available output) |
| Battery, tank, thermal storage | `<stores>` | `independent` |
| Fixed load, such as a demand profile that must be met | `<demands>` | `fixed` |
| Market that buys or sells as much as is profitable | `<demands>` (sell) or `<produces>` (buy) | `independent` |

Something that converts A into B is a `<produces>` of B with `<consumes>A`
and a `<transfer>`. Do not model it as two components.

## Model the system as a resource network

Components connect only by sharing a resource name. Draw the network first:
list every resource, and for each one check that something produces it and
something consumes or demands it. A resource with only producers has nowhere
to go, so a `fixed` producer of it can make the dispatch infeasible, while a
resource with only consumers can never be supplied. Run debug mode and look
at `network.png` to confirm the connections.

## Capacities

- A capacity is a ValuedParam: `fixed_value` for known units,
  `sweep_values` or `opt_bounds` for design variables (see `valued-params.md`).
- The capacity is measured in one resource. When a component touches several
  resources (a producer with `<consumes>`), set `<capacity resource="...">`
  to say which one. Omitting it with several active resources is an error.
- Signs: capacity is positive for a produced resource and negative for a
  demand or a consumed resource. A `<demands>` capacity of `1000` gives a
  component that produces 1000 of the resource.
- Time-varying availability (wind, solar) goes in `<capacity_factor>`, which
  accepts only `<ARMA>` or `<CSV>`. Do not fold it into the capacity: the
  capacity stays the installed size (and the capex driver).
- Every capacity variable is named `<component name>_capacity` in the
  generated workflow and in the `sweep.csv` / `opt_soln.csv` columns
  (`input/components.md`). Choose component names accordingly: they become
  variable names, so avoid spaces and punctuation.

## Transfer functions

- Use `<ratio>` for fixed-efficiency conversion. HERON fixes the signs of the
  rates for you.
- `<poly>` is for nonlinear conversion; signs are not adjusted, it needs a
  nonlinear solver, and it is lightly tested. Prefer `<ratio>` and check
  results in debug mode if you use `<poly>`.
- A `<produces>` with `<consumes>` and no `<transfer>` is an error ("Any
  component that consumes a resource must have a transfer function").
- The transfer function must use all and only the resources the component
  consumes or produces. HERON errors with "Transfer function for Component
  ... has a mismatch with consumed and produced!" and lists which resources
  are unused or unrecognized.

## Storage

- Capacity is the maximum level, not a rate. Charge and discharge limits are
  fractions of capacity per time step (`max_charge_rate`, `max_discharge_rate`).
- The reported activity is the level at the start of each step. To see flows,
  look at the charge and discharge activity (`tracking`; see `input/valued-params.md`).
- `periodic_level` defaults to `True`, so each rolling window must end at its
  starting level, and `initial_stored` is then ignored. If you want a storage
  that is allowed to drain over the horizon, set `periodic_level` false and
  give `initial_stored`.
- A storage with no `<initial_stored>` gets a 0% default and a warning is
  printed, even if `periodic_level` is true (in which case it is ignored).
- Windows are `rolling_window_length` steps (default 24) in the dispatcher.
  Storage cycling longer than a window (seasonal storage) will not be seen.

## Economics attached to components

- Every component needs `<economics>` with `<lifetime>`.
- Capital costs are `one-time` cash flows driven by `<variable>X_capacity</variable>`.
- Operating costs and revenues are `repeating` cash flows driven by
  `<activity>`. The dispatch is chosen within one design point, so what
  steers it is the hourly revenue and variable cost; capex depends on the
  capacity chosen by the outer sweep or optimizer.
- Check every sign against `input/economics.md`. The most common error is a
  market whose sales count as a cost.

## Common mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Resource spelled differently on two components | Two disconnected networks; unexpected zero activity or infeasible dispatch | Match spelling exactly |
| `fixed` producer feeding a resource nothing can absorb | Infeasible dispatch | Add a market or storage that is `independent` and can absorb it |
| Positive capacity on a `<demands>` | Wrong sign everywhere downstream | Capacity negative, or `<multiplier>-1</multiplier>` on positive data |
| Sales cash flow without `<multiplier>-1</multiplier>` on the driver | Revenue counted as cost, no warning | See sign table in `input/economics.md` |
| `dispatch="dependent"` in a new input | Legacy, will be deprecated | Use `fixed` or `independent` |

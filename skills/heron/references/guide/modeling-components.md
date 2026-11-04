# Components: modeling guide

How to turn a physical system into HERON components. For every option and
default, see `references/spec/components.md`; for cash flows see `references/spec/economics.md`.

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
  `sweep_values` or `opt_bounds` for design variables (see `references/guide/choosing-values.md`).
- The capacity is measured in one resource. When a component touches several
  resources (a producer with `<consumes>`), set `<capacity resource="...">`
  to say which one. Omitting it with several active resources is an error.
- Signs: positive for a produced resource, negative for a demand or a
  consumed resource (`references/spec/economics.md`). A `<demands>` capacity
  of `1000` gives a component that produces 1000 of the resource.
- Time-varying availability (wind, solar) goes in `<capacity_factor>`, not
  in the capacity: the capacity stays the installed size (and the capex
  driver).
- Every capacity variable is named `<component name>_capacity` in the
  generated workflow and the result columns, so avoid spaces and punctuation
  in component names.

## Transfer functions

- Use `<ratio>` for fixed-efficiency conversion. HERON fixes the signs of
  the rates for you.
- Use `<poly>` only for genuinely nonlinear conversion. It is experimental:
  signs are not adjusted and it needs a nonlinear solver
  (`references/spec/components.md`).
- The transfer function must use all and only the resources the component
  consumes or produces.

## Storage

- Capacity is the maximum level, not a rate. Charge and discharge limits are
  fractions of capacity per time step.
- The reported activity is the level at the start of each step. For flows,
  use the `charge` and `discharge` tracking variables.
- `periodic_level` (default true) makes each rolling window end at its
  starting level, and `initial_stored` is then ignored. For a storage that
  may drain over the horizon, set `periodic_level` false and give
  `initial_stored`.
- Windows are `rolling_window_length` steps (default 24). Storage cycling
  longer than a window (seasonal storage) will not be seen
  (`references/theory/dispatch.md`, "Rolling windows").

## Economics attached to components

- Every component needs `<economics>` with `<lifetime>`.
- The dispatch is chosen within one design point, so what steers it is the
  hourly revenue and variable cost. Capex depends only on the capacity
  chosen by the outer sweep or optimizer, and does not affect dispatch.
- Which cash flow type to use for capex, O&M, fuel, and sales is in
  `references/guide/best-practices.md`, "Economics"; signs are in
  `references/spec/economics.md`.

## Common mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Resource spelled differently on two components | Two disconnected networks; unexpected zero activity or infeasible dispatch | Match spelling exactly |
| `fixed` producer feeding a resource nothing can absorb | Infeasible dispatch | Add a market or storage that is `independent` and can absorb it |
| Positive capacity on a `<demands>` | Wrong sign everywhere downstream | Capacity negative, or `<multiplier>-1</multiplier>` on positive data |
| Sales cash flow without `<multiplier>-1</multiplier>` on the driver | Revenue counted as cost, no warning | Sign table in `references/spec/economics.md` |

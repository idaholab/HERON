# `<Components>` node

`<Components>` holds one or more `<Component name="...">` nodes, one for each
unit in the system. The `name` is used to build variable names throughout
the workflow (for example, the capacity of component `npp` is the variable
`npp_capacity`).

Each component has:

- exactly one **interaction**: `<produces>`, `<stores>`, or `<demands>`
- one `<economics>` node, which is required even if it has no cash flows
  (see `references/spec/economics.md`)

```xml
<Component name="npp">
  <produces resource="electricity" dispatch="fixed">
    <capacity resource="electricity">
      <fixed_value>1000</fixed_value>
    </capacity>
  </produces>
  <economics>
    <lifetime>60</lifetime>
  </economics>
</Component>
```

## Resources

A resource is any string, such as `electricity`, `steam`, or `h2`.
Components are connected only through the resources they share: everything
that produces, consumes, stores, or demands `electricity` is on the same
`electricity` network, and that resource must balance at every time step.
Spelling must match exactly. A misspelled resource name creates a separate
network that nothing else connects to.

## Attributes common to all interactions

| Attribute | Meaning |
|-----------|---------|
| `resource` | The resource produced, stored, or demanded. Required. |
| `dispatch` | How flexible the unit is. Always set it, since there is no usable default. |

`dispatch` values. Use only `fixed` or `independent`:

- `fixed`: always runs at its full available capacity (capacity times
  capacity factor, if one is given), and the dispatcher cannot change it.
  Use it for must-run units and for demands that must be met in full
  (e.g. a load profile).
- `independent`: the dispatch optimization chooses its level, anywhere
  between its minimum and its capacity. Use it for anything flexible:
  dispatchable generators, storage, and markets that take or supply as
  much as is profitable.
- `dependent`: **do not use in new inputs.** It does not mean anything
  different in practice and is going to be deprecated. Older inputs and many
  of HERON's tests still use it, often for markets. If asked to update such
  an input, use `independent` for anything the dispatcher should be able to
  adjust and `fixed` for anything that must always run at capacity. Do not
  change existing inputs unless the user asks.

## Children common to all interactions

All of these take a ValuedParam (see `references/spec/valued-params.md`).

| Child | Meaning |
|-------|---------|
| `<capacity>` | Maximum activity level, in units of the resource. Attribute `resource` names which resource the capacity is measured in. It is required when the interaction involves more than one resource (e.g. a producer that consumes something). |
| `<capacity_factor>` | Available fraction of capacity at each time step, from 0 to 1. Only `<ARMA>` or `<CSV>` are allowed. Typical use: wind or solar availability. |
| `<minimum>` | Minimum operating level, as a fraction of capacity, from 0 to 1. Default 0. |

**Signs:** a `<demands>` capacity is negative, and so is a producer's
capacity measured in a resource it consumes. The full convention is in
`references/spec/economics.md`.

## `<produces>`

Makes one or more resources, possibly by consuming others.

| Child | Meaning | Default |
|-------|---------|---------|
| `<consumes>` | Comma-separated list of resources consumed. If present, `<transfer>` is required. | (none) |
| `<transfer>` | How consumed resources become produced resources (below). | (none) |
| `<ramp_limit>` | Maximum change in production between consecutive time steps, as a fraction of capacity, in (0, 1]. | `1.0` |
| `<ramp_freq>` | Minimum number of time steps between ramping events. Adds binary variables, so it needs a mixed-integer solver. | `0` |

### Transfer functions

`<ratio>` is a fixed linear ratio. To make 1 unit of electricity from 2
units of steam:

```xml
<produces resource="electricity" dispatch="independent">
  <consumes>steam</consumes>
  <capacity resource="steam">
    <fixed_value>-100</fixed_value>   <!-- consumes up to 100 steam -->
  </capacity>
  <transfer>
    <ratio>
      <rate resource="steam">2</rate>
      <rate resource="electricity">1</rate>
    </ratio>
  </transfer>
</produces>
```

For `<ratio>`, HERON sets the signs itself (negative for consumed, positive
for produced), so the signs you write in `<rate>` are ignored. `<linear>` is
a deprecated alias for `<ratio>`. It still works, but prints a warning.

`<poly>` is a polynomial balance of the form `0 = sum of terms`. Each
`<coeff>` is one term. Its `resource` attribute lists the resources in the
term and its `order` attribute gives the matching powers. HERON does **not**
adjust the signs for `<poly>`. Each term multiplies the coefficient by the
signed activities, and consumed activities are negative. Choose the
coefficient signs so the terms cancel.

For example, to make work `w` from funding `f` (consumed) and labor `l`
(consumed) with `w = 0.9 f + l + 1e-6 f l^2`, write it in terms of the signed
activities (`f` and `l` are negative, `w` is positive), giving
`0 = -0.9 f - l - 1e-6 f l^2 - w`:

```xml
<transfer>
  <poly>
    <coeff resource="funding" order="1">-0.9</coeff>
    <coeff resource="labor" order="1">-1</coeff>
    <coeff resource="funding,labor" order="1,2">-1e-6</coeff>
    <coeff resource="work" order="1">-1</coeff>
  </poly>
</transfer>
```

A nonlinear `<poly>` needs a nonlinear solver in the dispatcher (e.g.
`<solver>ipopt</solver>`).

**`<poly>` support is limited.** It is lightly tested and is being actively
developed. Prefer `<ratio>` unless the user specifically needs a nonlinear
conversion. If you do use `<poly>`, tell the user it is experimental and
check the dispatch results (debug mode) carefully.

## `<stores>`

Holds one resource and can absorb or release it. Its `<capacity>` is the
maximum stored amount (positive). Its activity is reported as the storage
**level** at the start of each time step, not as a production rate.

| Child | Meaning | Default |
|-------|---------|---------|
| `<initial_stored>` | Starting level as a fraction of capacity, from 0 to 1 (a ValuedParam). Ignored when `periodic_level` is true. | `0.0` |
| `<periodic_level>` | Require the level to end each dispatch window where it started. If true, the initial level is chosen by the optimizer. | `True` |
| `<RTE>` | Round-trip efficiency, as a fraction. | `1.0` |
| `<max_charge_rate>` | Maximum charge rate per time step, as a fraction of capacity. | `1.0` |
| `<max_discharge_rate>` | Maximum discharge rate per time step, as a fraction of capacity. | `1.0` |
| `<strategy>` | A `<Function>` that sets the storage level directly, instead of letting the dispatcher optimize it (see `references/spec/functions.md`). | (optimized) |

```xml
<stores resource="steam" dispatch="independent">
  <capacity resource="steam">
    <fixed_value>100</fixed_value>
  </capacity>
  <RTE>0.9</RTE>
</stores>
```

## `<demands>`

Consumes one resource and nothing else. Its `<capacity>` is negative, and
so is its activity (it removes the resource from the system; see the sign
conventions in `references/spec/economics.md`). Use `dispatch="fixed"` for a demand that must
always be met in full (e.g. a load profile), and `dispatch="independent"` for
a market that takes whatever is profitable, up to its capacity.

```xml
<demands resource="electricity" dispatch="fixed">
  <capacity>
    <ARMA variable="TOTALLOAD">synth</ARMA>
    <multiplier>-1</multiplier>   <!-- load data is positive; demand must be negative -->
  </capacity>
</demands>
```

# Component `<economics>` node

Each `<Component>` needs one `<economics>` node. Project-wide settings
(project length, discount rate, tax, inflation) are set once in the `<Case>`
`<economics>` node instead (see `case.md`).

| Child | Meaning |
|-------|---------|
| `<lifetime>` | Number of cycles (years) the unit operates before replacement. A `one-time` cash flow is charged again each time the unit is replaced. |
| `<CashFlow>` | Zero or more cash flows (below). |

## Sign conventions

Getting the signs wrong is one of the most common HERON mistakes. The
result is that revenues get counted as costs, or costs as revenues, and HERON
gives no warning. Check every cash flow against these rules.

**Rule 1: cash flows.** A positive `C` is revenue and a negative `C` is a
cost.

**Rule 2: activity.** A component's activity for a resource is **positive**
when it adds that resource to the system, and **negative** when it removes
it. For example:

| Component | Resource | Activity sign |
|-----------|----------|---------------|
| NPP producing heat | heat | + |
| BOP turning heat into electricity | heat | − (consumed) |
| | electricity | + (produced) |
| Electricity market (a `<demands>` component) | electricity | − (it removes electricity) |
| Fuel consumed by a gas turbine | fuel | − |

**Rule 3: capacity.** A capacity has the same sign as the activity it
limits: positive for produced resources, negative for demands and for
capacities measured in a consumed resource. So `<variable>grid_capacity</variable>`
on a demand component is a negative number, and so is the capacity variable
of a producer whose `<capacity resource="...">` names a consumed resource.
Watch for this when using such a capacity as a capex driver.

Multiply the signs together to get the sign of `C`. When the sign comes out
wrong, add `<multiplier>-1</multiplier>` to the driver (or to the price).
If `scaling_factor_x` is not 1, make sure the driver is positive (with a
multiplier if needed): a negative number raised to a fractional power is not
a real number.

**The classic case: market sales.** A market is a `<demands>` component, so
its electricity activity is negative. With a positive price, the cash flow
would be negative, which is a cost. Flip the driver to get positive revenue:

```xml
<Component name="grid">
  <demands resource="electricity" dispatch="independent">
    <capacity><fixed_value>-1000</fixed_value></capacity>
  </demands>
  <economics>
    <lifetime>1</lifetime>
    <CashFlow name="sales" type="repeating" taxable="True" inflation="none">
      <driver>
        <activity>electricity</activity>
        <multiplier>-1</multiplier>   <!-- negative activity -> positive amount sold -->
      </driver>
      <reference_price>
        <ARMA variable="PRICE">synth</ARMA>   <!-- positive $/unit -->
      </reference_price>
    </CashFlow>
  </economics>
</Component>
```

Quick reference for common cash flows:

| Cash flow | Driver (sign) | Price | Multiplier |
|-----------|---------------|-------|------------|
| Capex on a producer | `<variable>x_capacity</variable>` (+) | negative | none |
| Fixed O&M on a producer | `<variable>x_capacity</variable>` (+) | negative | none |
| Variable O&M on a producer | `<activity>` of its product (+) | negative, or positive with multiplier `-1` | as needed |
| Fuel cost on a consumer | `<activity>` of the fuel (−) | positive | none, since − × + = cost |
| Sales to a market (`<demands>`) | `<activity>` of the product (−) | positive | `-1` on the driver |
| Purchases from a market (a `<produces>` import) | `<activity>` of the product (+) | positive | `-1`, to make it a cost |

## `<CashFlow>`

Each cash flow has the form

```
C = reference_price * (driver / reference_driver) ^ scaling_factor_x
```

For a simple per-unit price, leave out `reference_driver` and
`scaling_factor_x` (both default to 1), and then `C = reference_price * driver`.

### Attributes

| Attribute | Required | Meaning |
|-----------|----------|---------|
| `name` | yes | Name of the cash flow. In outputs it appears as `<component>|<name>`. |
| `type` | yes | `one-time` (capital costs) or `repeating` (O&M, fuel, sales). |
| `taxable` | yes | `True` or `False`. |
| `inflation` | yes | How inflation applies. Almost every input uses `none`. |
| `period` | no | For `repeating` cash flows: `hour` (every time step, the default) or `year` (once per cycle, e.g. fixed O&M). |
| `npv_exempt` | no | `True` keeps the cash flow in the dispatch objective but leaves it out of the NPV calculation. This is an advanced option that changes how results should be read. Default `False`. |
| `mult_target` | no | Deprecated and has no effect. Use `<levelized_cost/>` instead. |

### Children

All except `<depreciate>` take a ValuedParam (see `valued-params.md`). In
addition to the usual types, cash flow slots also accept `<activity>`, which
is the dispatched activity of this component.

| Child | Meaning | Default |
|-------|---------|---------|
| `<driver>` | Quantity the cash flow scales with. Required. | (none) |
| `<reference_price>` | Price per `reference_driver` units of the driver. | |
| `<reference_driver>` | Number of driver units the reference price applies to. | `1` |
| `<scaling_factor_x>` | Economy-of-scale exponent. Values below 1 make each unit cheaper as the driver grows. | `1` |
| `<depreciate>` | Number of years over which the cost is depreciated (MACRS). Integer. | (none) |

### Examples

Capital cost that scales with capacity, and is paid again at each
replacement:

```xml
<CashFlow name="capex" type="one-time" taxable="True" inflation="none">
  <driver><variable>npp_capacity</variable></driver>
  <reference_price><fixed_value>-5e6</fixed_value></reference_price>
  <depreciate>15</depreciate>
</CashFlow>
```

Fixed O&M charged once per year:

```xml
<CashFlow name="FOM" type="repeating" period="year" taxable="True" inflation="none">
  <driver><variable>npp_capacity</variable></driver>
  <reference_price><fixed_value>-1e5</fixed_value></reference_price>
</CashFlow>
```

### Levelized cost

To solve for the break-even price of one cash flow, put `<levelized_cost/>`
in its `<reference_price>`:

```xml
<reference_price>
  <levelized_cost/>
</reference_price>
```

Use it together with `LC` as the economic metric (`<EconMetrics><LC/>` or
`<opt_metric>LC</opt_metric>` in `<Case>`). HERON then finds the price that
makes NPV equal to the target (default 0).

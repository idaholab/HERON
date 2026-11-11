# Component interview

Questions to ask for each component in phase 3, in order. Skip a section
when it does not apply (a market has no transfer function; a generator has
no storage options). Specs for every answer are in `../heron/references/spec/components.md`
and `../heron/references/spec/economics.md`.

Ask about units first and keep them consistent across the whole case: one
energy unit (e.g. MW for rates, MWh for amounts over a step), one currency,
and one time step length. Prices must be per the same unit the activity is
in.

## 1. Interaction and resources

- Does it make something (`<produces>`), hold something (`<stores>`), or
  take something out of the system (`<demands>`)? See
  `../heron/references/guide/modeling-components.md`.
- Which resource does it produce, store, or demand? If it also consumes
  something, which? (Use the names agreed in phase 2.)

A market the system sells into is a `<demands>`. A market or supply the
system buys from (grid imports, fuel) is a `<produces>`.

## 2. Dispatch policy

Ask: "Should the optimizer decide how much this unit runs at each time
step, or does it always run at its available capacity?"

| Answer | `dispatch` |
|--------|------------|
| The optimizer decides (dispatchable plant, storage, a market that takes what is profitable) | `independent` |
| Always at capacity (must-run unit, a load that must be met in full, a wind farm taking all its available output) | `fixed` |

Never use `dependent` in a new input.

Follow-up questions for flexible units:
- Is there a minimum stable level? (`<minimum>`, fraction of capacity)
- Is ramping limited? (`<ramp_limit>`, fraction of capacity per step;
  `<ramp_freq>` needs a mixed-integer solver, so mention the cost)

## 3. Capacity

- What is the capacity, and in which resource is it measured? For a unit
  that consumes something, ask whether it is sized by its input (e.g. a
  turbine by its steam intake) or its output.
- Is the capacity **fixed**, a **design** variable (phase 1), or **data**
  (e.g. a load profile from a time series)?
- For a design variable: the sweep values or the optimization bounds, and a
  representative `debug_value`. Pick a debug value the user would recognize
  as typical, not an extreme.
- Signs: a `<demands>` capacity is negative, and so is a capacity measured
  in a consumed resource. Data that is positive (a load profile) needs
  `<multiplier>-1</multiplier>`.
- Is the availability time-varying (wind, solar)? That is a
  `<capacity_factor>` from an `<ARMA>` or `<CSV>`, not a varying capacity.

## 4. Conversion (producers that consume)

- What is the conversion rate? Ask for it as "N units of input per 1 unit of
  output" and write a `<ratio>`. HERON sets the signs of `<rate>`s itself.
- If the user describes a nonlinear efficiency curve, suggest a linear
  approximation first. `<poly>` is experimental; say so if it is used.

## 5. Storage options (`<stores>` only)

- Capacity is the maximum stored amount (energy), positive.
- Round-trip efficiency (`<RTE>`)? Charge and discharge rate limits as a
  fraction of capacity per step?
- Should the level return to where it started at the end of each dispatch
  window (`<periodic_level>`, default true)? If not, the starting level
  (`<initial_stored>`).
- Storage needs a component that buys and a component that sells the
  stored resource; check the phase 2 graph has both.

## 6. Economics

Ask for the unit's lifetime (`<lifetime>`, in years). Then ask which costs
and revenues apply, one at a time. Typical set:

| Cash flow | `type` / `period` | Driver | Usually |
|-----------|-------------------|--------|---------|
| Capital cost | `one-time` | `<variable>comp_capacity</variable>` | per unit of capacity; ask about `<depreciate>` years and economy of scale (`<scaling_factor_x>`, `<reference_driver>`) |
| Fixed O&M | `repeating`, `period="year"` | `<variable>comp_capacity</variable>` | per unit of capacity per year |
| Variable O&M | `repeating` | `<activity>product</activity>` | per unit produced |
| Fuel | `repeating` | `<activity>fuel</activity>` | per unit consumed |
| Sales | `repeating` | `<activity>product</activity>` with `<multiplier>-1</multiplier>` | market price, often data |
| Purchases | `repeating` | `<activity>product</activity>` with `<multiplier>-1</multiplier>` | import price |

The capacity variable for a component is named `<component>_capacity`.

For each cash flow, ask and record:
1. Is it a cost or a revenue? State the expected sign of `C`.
2. The price, its unit, and its label: fixed, data (price time series),
   uncertain, or function.
3. `taxable` (usually `True`) and `inflation` (usually `none`).

Then check the sign, out loud, with the user: sign of the driver (activity
is negative for anything consumed or demanded; capacity is negative for
demands) times the sign of the price must give the sign from step 1. If not,
add `<multiplier>-1</multiplier>`. The quick reference table in
`../heron/references/spec/economics.md` covers the common cases.

### Uncertain parameters

If the user is unsure of a price or quantity, ask whether it should be
**swept** (a design-like sensitivity the user controls) or **uncertain**
(random, drawn once per realization). Uncertain parameters:
- are only allowed in cash flow `<driver>`, `<reference_price>`,
  `<reference_driver>`, and `<scaling_factor_x>`;
- need one of the supported distributions listed in
  `../heron/references/spec/valued-params.md` (ask for its shape and
  parameters: e.g. triangular with min, apex, max);
- force the bilevel workflow even with a CSV history, and need enough
  realizations to matter (`../heron/references/guide/uncertainty.md`).

### Levelized cost

If the question is "what price makes this break even?", that cash flow's
`<reference_price>` is `<levelized_cost/>` and the metric is `LC`. Only one
cash flow per case gets this.

## Record

After each component, add its rows to the decision record, write its XML,
and show both to the user before moving to the next component.

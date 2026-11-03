# Economics math

How cash flows become NPV, IRR, PI, and LC. HERON builds the yearly cash
flows (`src/DispatchManager.py`); the TEAL plugin applies tax, inflation,
depreciation, replacement, and discounting (`TEAL/src/main.py`,
`CashFlows.py`, `Amortization.py`). The theory manual's derivation is in
`doc/theory_manual/ComponentCharacterization/HERON_Cashflows.md`.

## One cash flow

$$\hat F = \alpha \left(\frac{d}{d'}\right)^{\chi}$$

$\alpha$ is `<reference_price>`, $d$ the `<driver>`, $d'$ the
`<reference_driver>` (default 1), and $\chi$ the `<scaling_factor_x>`
(default 1). The sign rules are in `references/spec/economics.md`.

## Yearly values

Project years are $y = 0, 1, \ldots, Y$ with $Y$ = `<ProjectTime>`. Year 0
is construction; years $1..Y$ operate.

| Type | Yearly value |
|------|--------------|
| `one-time` | $\hat F$ evaluated once with the capacity, in year 0, and again in each year $nL < Y$ that a replacement is built ($L$ = `<lifetime>`; no rebuild in the final year) |
| `repeating`, `period="year"` | $\hat F$ in each year $y \ge 1$ |
| `repeating`, `period="hour"` | $H_y = \sum_u m_u \sum_t \hat F_{y,u,t}$ in each year $y \ge 1$ |

In $H_y$, $t$ runs over the steps of one cluster or segment $u$, and $m_u$
is the number of segments $u$ represents (1 when unclustered). This is a
plain sum, not a time integral: a price per MWh times MW activity over
hourly steps gives dollars only because $\Delta t = 1$ hour. A single-year
time series is replayed for every $y$. `npv_exempt="True"` cash flows are
left out here (they still steer the dispatch).

## Tax and inflation

Each cash flow's yearly value $F_y$ is adjusted before discounting:

$$F'_y = F_y \,(1 - \delta_\tau \tau)\,(1 + \delta_\nu \nu)^{-y}$$

with $\tau$ = `<tax>`, $\nu$ = `<inflation>`, $\delta_\tau = 1$ when
`taxable="True"`, and $\delta_\nu = 1$ when `inflation` is `True` or `real`
(`none` or `False` turn it off). The tax is a flat multiplier on each cash
flow, revenue and cost alike. It is not a tax on net income, and losses do
not carry over.

## Depreciation (`<depreciate>`)

For a `one-time` cash flow $A$ with `<depreciate>R</depreciate>`, TEAL adds
two cash flows built from the MACRS fractions $\beta_{k}$ for recovery
period $R \in \{3, 5, 7, 10, 15, 20\}$, in years $k = 1..R+1$ after each
build:

- a tax credit $+\beta_k A$: not taxed, inflated (`<comp>_<cf>_depreciation_tax_credit`)
- a depreciation $-\beta_k A$: taxed, inflated (`<comp>_<cf>_depreciation`)

The net effect is $\beta_k \tau A$ per year, the tax shield. It only
matters when $\tau > 0$ and the capex is `taxable="True"`.

## Metrics

With the free cash flow to the firm $\text{FCFF}_y = \sum_{\text{all CF}} F'_y$
and $r$ = `<DiscountRate>`:

$$\text{NPV} = \sum_{y=0}^{Y} \frac{\text{FCFF}_y}{(1+r)^y}$$

- **IRR**: the $r$ with NPV = 0. Undefined without a sign change in FCFF.
- **PI**: $-\text{NPV} / \text{FCFF}_0$.
- **LC** (`LC_Mult`): splits the discounted cash flows into $M$, those of
  the `<levelized_cost/>` cash flow, and $O$, all others, and returns
  $(\text{target} - O)/M$, the multiplier on that cash flow's price that
  makes NPV equal the target (0). The dispatch objective then uses the
  same split, which makes it nonlinear (`src/dispatch/PyomoModelHandler.py`,
  `_compute_levelized_cashflows`).

## Across realizations

Each realization gives one NPV (and one of each other metric). The outer
loop reports statistics of those values across realizations
(`references/guide/uncertainty.md`). The design-level objective is
$\mathcal S(\{\text{NPV}_s\})$, e.g. the mean
(`references/theory/bilevel.md`).

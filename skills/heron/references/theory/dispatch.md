# Dispatch formulation

The optimization the `<pyomo/>` dispatcher solves for one window of one
cluster of one year of one realization, at fixed capacities. Read this to
explain why a dispatch looks the way it does. Source:
`src/dispatch/pyomo_dispatch.py`, `PyomoModelHandler.py`, `PyomoRuleLibrary.py`.

## Notation

| Symbol | Meaning |
|--------|---------|
| $p$, $x$, $t$ | component, resource, time step within the window |
| $D_{p,x,t}$ | activity (signed: + adds to the system, − removes) |
| $c_{p,t}$ | capacity at $t$: `<capacity>` times `<capacity_factor>` (if any), in the capacity resource |
| $m_{p,t}$ | minimum: `<minimum>` times capacity |
| $\Delta t$ | step length, $t_{k+1}-t_k$ in units of the time variable |

## Objective

$$\max_D \ \sum_{p}\sum_{t} \sum_{\text{hourly CF } f \text{ of } p} \alpha_{f,t} \left(\frac{d_{f,t}(D)}{d'_{f,t}}\right)^{\chi_f}$$

- Only hourly `repeating` cash flows (`period="hour"`, the default) are in
  the objective. `one-time` and `period="year"` cash flows do not affect
  the dispatch.
- No discounting, tax, inflation, or cluster weighting is applied here:
  each step of the window counts equally. Those apply afterwards, in the
  economics (`references/theory/economics.md`).
- A $\chi \ne 1$ on an activity driver, a `<poly>` transfer, or the `LC`
  metric makes the problem nonlinear.

## Constraints

**Resource balance** (`<x>_conservation`), for every resource and step:

$$\sum_{p \text{ using } x} D_{p,x,t} = 0$$

Storage enters as charge + discharge. There is no slack: any imbalance is
infeasible. That is why a `fixed` component needs somewhere to send its
resource.

**Capacity and minimum** apply only to the capacity resource
$x_c$ (`<capacity resource="...">`). For positive capacities,

$$m_{p,t} \le D_{p,x_c,t} \le c_{p,t}$$

and for negative ones (demands, consumed resources) the bounds swap, so
$c_{p,t} \le D_{p,x_c,t} \le m_{p,t}$. They are variable bounds, not named
constraints.

- `dispatch="fixed"`: the minimum is replaced by the capacity, so both
  bounds equal $c_{p,t}$. A fixed producer always makes capacity times
  capacity factor, and a fixed demand always takes its full (negative)
  capacity.
- `dispatch="independent"`: free between the bounds.

**Transfer functions**, for a producer with `<consumes>`:

- `<ratio>`, one equality per non-reference resource
  (`<comp>_<res>_<ref>_transfer`): $D_{p,x,t} = (\rho_x/\rho_{ref}) D_{p,ref,t}$,
  with the rates signed by HERON (consumed negative). Other resources than
  $x_c$ are bounded only through this.
- `<poly>` (`<comp>_transfer_func`): $\sum_k a_k \prod_x D_{p,x,t}^{o_{k,x}} = 0$
  with the signs as written.

**Ramping** (`<ramp_limit>` $\ell$), for $t > 0$:
$|D_{p,x_c,t} - D_{p,x_c,t-1}| \le \ell\, c_p$. The limit uses
available capacity (times capacity factor), not installed. It is skipped at
the first step of each window, so a unit can jump between windows. `<ramp_freq>`
adds binary up/down/steady variables per step and limits ramp events, so
the problem becomes a MILP.

## Storage

Variables: level $L_t \in [0, c]$, charge $C_t \in [-r_c c, 0]$, and
discharge $E_t \in [0, r_d c]$, with $r_c$, $r_d$ the
`max_charge_rate` and `max_discharge_rate`. With $\eta$ the round-trip
efficiency `<RTE>`:

$$L_t = L_{t-1} - \sqrt{\eta}\, C_t\, \Delta t - \frac{E_t}{\sqrt{\eta}}\, \Delta t$$

The loss is split evenly between charging and discharging. The balance sees
$C_t + E_t$, the net flow into the system.

- $L_{-1}$ is the initial level: `initial_stored` times capacity, or, when
  `periodic_level` is true (default), the window's own final level
  $L_{T-1}$. That makes each window cyclic and ignores `initial_stored`.
- Simultaneous charge and discharge is allowed (the binary that would forbid
  it is disabled in the code). With $\eta < 1$ it wastes energy, so an
  optimal solution only does it when it must dump a surplus.
- A storage `<strategy>` replaces all of this with a fixed level profile,
  from which charge and discharge are derived, iterated up to 10 times to
  converge.

## Rolling windows

The time steps of one cluster of one year are split into consecutive
windows of `rolling_window_length` steps (default 24), solved one after
another with perfect foresight inside each window and none across them.

- Each window starts storage from its initial level again: with
  `periodic_level` true, each window is a closed cycle. Nothing carries
  over between windows. So storage cannot move energy between days with
  24-step windows, and seasonal storage needs windows as long as the season.
- The last window is shorter if the steps do not divide evenly. A final
  window of one step is an error.
- Longer windows give a better dispatch but larger problems.

## After the solve

A `<validator>`, if present, checks the solution and adds production limits,
and the window is solved again (`references/spec/case.md`). The solved
activity of all windows is stitched together, and the hourly cash flows are
evaluated again with numbers for the economics.

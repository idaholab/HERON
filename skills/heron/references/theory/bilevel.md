# Bilevel structure

The problem HERON's standard workflow solves, and how the outer and inner
RAVEN runs split it. Source: `templates/bilevel_templates.py`.

## The problem

$$\underset{\mathbf c}{\text{opt}} \ \ \mathcal S\Big(\big[\ \mathcal T\big(\mathbf c,\ D^\star(\mathbf c, W_s)\big)\ :\ s = 1..S\ \big]\Big)$$

| Symbol | Meaning | Input |
|--------|---------|-------|
| $\mathbf c$ | capacities (and `dispatch_vars`) | `sweep_values` / `opt_bounds` |
| $W_s$ | realization $s$: one synthetic history over all years, plus one draw of every `<uncertainty>` | `<ARMA>`, `<uncertainty>` |
| $S$ | number of realizations | `<num_arma_samples>` |
| $D^\star$ | optimal dispatch for every window, cluster, and year of $W_s$ at capacities $\mathbf c$ | `references/theory/dispatch.md` |
| $\mathcal T$ | economic metric of the resulting cash flows | `references/theory/economics.md` |
| $\mathcal S$ | statistic across realizations | `<stats_metric>` |

"opt" is max or min (`<type>`) in opt mode, or evaluation at each grid
point in sweep mode, where several statistics and metrics are reported.

## Where each piece runs

| Level | RAVEN run | Does |
|-------|-----------|------|
| Outer | `outer.xml`, working dir `<Case>_o` | Chooses $\mathbf c$: a Grid sampler (sweep) or an optimizer (opt). Launches the inner run once per $\mathbf c$ and collects statistics. |
| Inner | `inner.xml`, working dir `<Case>_i` | A Monte Carlo sampler draws $S$ realizations. For each, HERON's `DispatchManager` dispatches every year and cluster and computes the metrics. Then the inner computes $\mathcal S$ of each metric and returns it. |

## Consequences

- The dispatch does not know about capex or the outer objective. It
  maximizes hourly operating profit at given capacities, window by window.
  So the dispatch can be economically sensible while the design is a bad
  one, and the reverse.
- The dispatch has perfect foresight within each window, so its profit is
  an upper bound on what a real operator would earn.
- The outer objective is noisy, because it is an estimate from $S$
  realizations. Gradient-based optimization (`GradientDescent`) is
  especially sensitive to that. Increase $S$ before tightening convergence.
- Every outer evaluation is $S$ full dispatches over all years, so the
  total cost multiplies (`references/guide/modes.md`, "Run cost").
- Design variables enter only through capacities and `dispatch_vars`.
  Anything else (prices, efficiencies) is fixed or random, not optimized.

## Other workflows

- **Flat sweep** (only static histories, a single sample, sweep mode, no
  `<uncertainty>`): $S = 1$, so there is no inner run; one dispatch per grid
  point and no statistics.
- **Debug**: one $\mathbf c$ (the `debug_value`s), a few realizations and
  years, and plots.
- `<workflow>` `MOPED`, `combined`, `DISPATCHES`: single monolithic
  optimizations that choose capacities and dispatch together. They do not
  follow this structure.

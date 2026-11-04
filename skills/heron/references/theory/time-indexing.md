# Time indexing

How a realization's time axis is split into years, segments, clusters, and
time steps, and how results are added back up. Source: `src/DispatchManager.py`.
What this means for which statistics to trust is in
`references/guide/time-series.md`, "Clustering and higher moments".

## Indices

| Index | Range | Name in outputs |
|-------|-------|-----------------|
| Realization $s$ | $1..S$ (`<num_arma_samples>`) | `RAVEN_sample_ID` |
| Year $y$ | project years $1..Y$ | `<year_variable>` |
| Cluster $u$ | representative segments $1..U$ | `_ROM_Cluster` |
| Step $t$ | steps within one segment, $1..T_U$ | `<time_variable>` |

A year of $T_Y$ steps is divided into $N$ segments of $T_U$ steps each
($T_Y = N T_U$; e.g. 8760 hours into 365 days of 24). The model is trained
per segment and can cluster the $N$ segments into $U$ groups. Cluster $u$
represents the set $\mathbb N_u$ of segments, and its multiplicity is
$m_u = |\mathbb N_u|$, with $\sum_u m_u = N$.

| Model | $U$ | $m_u$ |
|-------|-----|-------|
| Unsegmented | 1 | 1 |
| Segmented, not clustered | $N$ | 1 each |
| Clustered (default `evalMode`) | $U < N$ | segments per cluster |

`<time_discretization>` describes one segment (one cluster), not the whole
year: `<num_steps>` must equal $T_U$.

## Dispatch and aggregation

The dispatcher runs once per $(s, y, u)$, over the $T_U$ steps of that
cluster, in rolling windows (`references/theory/dispatch.md`). For each
realization:

- **Hourly cash flows** for year $y$: $H_y = \sum_u m_u \sum_t \hat F_{y,u,t}$
  (`references/theory/economics.md`).
- **Total activity** columns: $\sum_y \sum_u m_u \int D_{y,u}(t)\,dt$, with the
  integral by the trapezoid rule, undiscounted.

## Years

- A **single-year** model is replayed for every project year: the same
  $(u, t)$ values, with a new draw for each realization but the same draw for
  each year within it.
- An **interpolated** (multi-year) model gives each year its own values. It
  must cover at least $Y$ years; extra years are dropped.
- Year values in outputs come from the model (e.g. `2020, 2021, ...`) or count
  up from the model's first year.

## Why multiplicity inflates spread

Within one realization, one dispatched cluster stands in for $m_u$
segments, which then contribute $m_u$ identical copies. Their real
counterparts would vary independently around the cluster, so their sum
would vary less. The mean is unbiased, but the variance across realizations
of any sum is too large by up to a factor of about $m_u$ for the part
contributed by within-cluster variation.

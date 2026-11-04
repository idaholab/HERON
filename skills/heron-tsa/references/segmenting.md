# Segmenting, clustering, and multi-year models

`<Segment>` inside the ROM splits each history along the pivot into equal
segments (usually days) and trains a separate set of per-segment
algorithms on each. Optionally it clusters similar segments, and for
multi-year data it interpolates between trained years. These choices set
the time grid HERON dispatches on, so they decide HERON's run cost as much
as the fit.

## `<Segment grouping="...">`

| `grouping` | What it does | Needs |
|------------|--------------|-------|
| `segment` (default) | One trained model per segment; samples are the segments back to back. | `<subspace>` |
| `cluster` | Trains each segment, then groups segments with similar trained parameters. Each cluster is represented by one segment's model. | `<subspace>`, `<Classifier>` |
| `interpolate` | Trains each listed year (each as `cluster`), then fills in the missing years by linear interpolation of the trained parameters. | `<subspace>`, `<Classifier>`, `<macroParameter>` |

## Children

| Child | Meaning |
|-------|---------|
| `<subspace>` | Text: the pivot name. One of the attributes `pivotLength` (length of each segment in pivot units, e.g. `24`) or `divisions` (number of equal segments). The data length must divide evenly. Attribute `shift`: `zero` restarts the pivot at 0 in each segment, `first` at the first segment's start; values are restored when sampling. |
| `<Classifier class="Models" type="PostProcessor">` | Name of a `DataMining` PostProcessor that clusters the segments (below). |
| `<macroParameter>` | `interpolate` only: the year variable, as in the training data. |
| `<evalMode>` | `clustered`, `truncated` (RAVEN's default), or `full`. How a clustered model is sampled (below). Use `full`. |
| `<evaluationClusterChoice>` | Which member's model represents each cluster when sampling: `first` (default; least memory), `random` (a random member each sample), or `centroid`. |
| `<clusterFeatures>` | Which trained parameters to cluster on. Default: all clusterable parameters (`references/algorithms.md`). Leave it out unless there is a reason. |

## The classifier

```xml
<PostProcessor name="classifier" subType="DataMining">
  <KDD labelFeature="labels" lib="SciKitLearn">
    <Features>PRICE, LOAD</Features>
    <SKLtype>cluster|KMeans</SKLtype>
    <n_clusters>8</n_clusters>
    <tol>1E-12</tol>
    <init>k-means++</init>
    <random_state>3</random_state>
  </KDD>
</PostProcessor>
```

RAVEN replaces `<Features>` with the trained per-segment parameters, each
z-normalized across segments before clustering. The text there must still
be valid but does not choose what is clustered. `<n_clusters>` is the main
choice. Set `random_state` so the clusters are reproducible.

Only per-segment algorithms that are characterizers produce cluster
features. If every algorithm is global, segments have nothing to cluster
on.

## Multi-year without segmenting

`interpolate` needs a clustered model per year. For one model per year
with no daily segments, use a single segment and one cluster:

```xml
<Segment grouping="interpolate">
  <macroParameter>Year</macroParameter>
  <Classifier class="Models" type="PostProcessor">classifier</Classifier>
  <subspace divisions="1">Time</subspace>
  <evalMode>full</evalMode>
</Segment>
```

with `<n_clusters>1</n_clusters>` in the classifier.

## What HERON's time grid becomes

HERON samples the ROM in its own `evalMode` (set on the HERON `<ARMA>`
node, default `clustered`) and dispatches one window per segment of the
sample.

| ROM | HERON `evalMode="clustered"` (default) | HERON `evalMode="full"` |
|-----|----------------------------------------|-------------------------|
| No `<Segment>` | One window: the whole history | Same |
| `grouping="segment"` | One window: the whole history | Same |
| `cluster` or `interpolate`, $U$ clusters of segment length $T_U$ | $U$ windows of $T_U$ steps per year; each cluster's results are weighted by the number of segments it represents | One window: the whole year, reassembled from cluster representatives |

`heron inspect model.pk` prints the time axis and cluster count for each
mode and the `<time_discretization>` to match. HERON's `full` mode uses
the `<evalMode>` the ROM was trained with, so train with `full`.

Run cost scales with the number of windows dispatched and their length:
$U$ short windows (clustered) are far cheaper than one 8760-step window.

## Choosing the number of clusters

- **Fewer clusters** are faster in HERON and smooth out unusual days.
- **More clusters** keep more of the variety between days, including rare
  extreme days, at a higher run cost.
- Each cluster stands for $m_u$ segments. In HERON's `clustered` mode one
  dispatched day is counted $m_u$ times, so within a realization those
  days are perfectly correlated. Means are preserved; spread statistics
  across realizations are inflated. HERON's guide covers what this means
  for results (`../heron/references/guide/time-series.md`, "Clustering
  and higher moments").
- To choose $U$, train with several values and compare the validation
  statistics (`references/validation.md`), the duration curve in
  particular. Stop increasing $U$ when they stop improving. The k-means
  within-cluster sum of squares,
  $W(U) = \sum_{u=1}^{U}\sum_{i \in u}\lVert z_i - \bar z_u\rVert^2$ over the
  normalized features $z_i$, is a rough guide: look for the $U$ where it
  stops falling quickly.
- Typical values for a year of daily segments are 4 to 20.

## Segments are sampled independently

Each segment's stochastic model starts from its own initial state, so the
noise is not continuous across segment boundaries, and autocorrelation
beyond one segment comes only from global algorithms. Consequences:

- Multi-day persistence (a week of low wind) is not reproduced by
  per-segment `arma` alone. A global `fourier` adds back the slow cycles
  but not random persistence.
- Check the autocorrelation at lags past one segment
  (`references/validation.md`). If multi-day persistence matters for the
  study (long-duration storage), use longer segments or no segmenting.

## Multi-year interpolation

- Each listed year is trained and clustered on its own. A missing year $y$
  between trained years $y_a < y < y_b$ gets parameters
  $\theta(y) = \theta_a + \frac{y-y_a}{y_b-y_a}(\theta_b - \theta_a)$,
  segment by segment.
- Interpolation is linear between the listed years and covers only the
  years from the first to the last listed. It does not extrapolate. The
  span must cover the HERON project life.
- Parameters are interpolated, not data, so an interpolated year is only
  as good as the linear change in the trained parameters. Train more
  years if the change is not smooth.

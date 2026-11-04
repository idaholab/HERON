# Choosing algorithms and transforms

A synthetic history model splits each signal into a deterministic part
(cycles, trend), a change of distribution, and a stationary stochastic part:

$$x_t = g\big(m(t) + y_t\big), \qquad y_t \text{ a stationary ARMA process}$$

where $m(t)$ is the deterministic structure (`fourier`, `PolynomialRegression`,
`STL`) and $g$ is the inverse of the distribution transforms (`gaussianize`,
`logtransformer`, ...). Training removes these layers from the outside in;
the stochastic model sees only what is left. Most bad fits come from leaving
structure in the residual or choosing a transform that does not suit the
data, not from the ARMA order.

## The usual chain

```xml
<preserveCDF   target="..."/>              <!-- optional: keep each signal's distribution -->
<fourier       target="..."> <periods>...</periods> </fourier>   <!-- cycles -->
<gaussianize   target="..."/>              <!-- residual to normal -->
<arma          target="..." seed="42"> <P>..</P> <Q>..</Q> </arma>
```

Order rules:

1. Bounds and output-only transforms (`outtruncation`, `preserveCDF`) go
   first, so that in sampling they act on the final output.
2. `zerofilter` goes before anything that would turn zeros into nonzeros.
3. Deterministic structure (`fourier`, `PolynomialRegression`, `STL`)
   before distribution transforms. A cycle is additive in the original
   units, not after gaussianizing.
   Exception: a log transform for a multiplicative signal comes first
   (below).
4. `gaussianize` immediately before `arma` or `varma`.
5. `autoarma`, if used, after the same transforms as the `arma` it feeds,
   followed by `<arma auto_select="True">`.

## Matching data features to algorithms

| What the data shows | Use |
|---------------------|-----|
| Daily, weekly, or yearly cycles | `fourier` with those periods and their first few harmonics (24, 12, 8 for daily in hours). Periods longer than a segment in a separate `global="True"` node. |
| A cycle whose shape changes through the year | Segment by day and cluster (`references/segmenting.md`), so each cluster fits its own daily shape. |
| Smooth trend within the training window | `PolynomialRegression` of low degree, or `STL`. |
| Growth or change between years | Multi-year training data with `grouping="interpolate"`, not a trend term. |
| Zeros for long periods (solar at night) | `zerofilter` before the rest. Pair with `outtruncation domain="positive"` first in the chain. |
| Must not go negative (load, wind, prices with a floor) | `outtruncation domain="positive"` first in the chain, or a log transform. |
| Bounded on both sides (capacity factor in [0, 1]) | `bounddomain` with the bounds, or `sigmoidtransformer` after scaling to (0, 1). |
| Skewed, heavy-tailed, or spiky (scarcity prices) | `gaussianize` before `arma`. If spikes cluster in time, consider `MarkovAR` with 2 regimes. |
| Variability that scales with level | `logtransformer` (positive data) or `arcsinhtransformer` (data near or below zero) first, then the rest on the transformed signal. |
| Random-walk behavior (residual autocorrelation decays very slowly) | `differencing` with `<order>1</order>`, then `arma`. Check that nothing after it needs NaN-free data. |
| Signals that move together (load and price; wind at nearby sites) | `varma` on those signals, or `arma` per signal if correlation does not matter for the study. |
| Distinct regimes (calm and stormy, normal and scarcity) | `MarkovAR`. Needs NaN-free input. |

## Correlated signals

`arma` with several targets fits an independent model to each. Their
realizations share deterministic structure (cycles), so they remain
correlated through $m(t)$, but the stochastic parts are independent. This
underestimates joint extremes, such as high load with low wind.

`varma` models the stochastic parts jointly with a full noise covariance
$\Sigma$ and cross-lag terms. Use it when the economics depend on
coincidence between signals (a price that responds to load, renewables
against demand). Its cost grows as $k^2(P+Q)$ parameters for $k$ signals,
so keep orders low ($P \le 2$, $Q \le 1$) and $k$ small.

Check by comparing the cross-correlation of samples and data
(`references/validation.md`).

## Choosing P and Q

The ARMA orders control how long memory the stochastic part has. After the
deterministic structure and transforms are removed:

| Residual autocorrelation (ACF) | Partial autocorrelation (PACF) | Suggests |
|--------------------------------|--------------------------------|----------|
| Decays geometrically | Cuts off after lag $p$ | AR($p$): $P = p$, $Q = 0$ |
| Cuts off after lag $q$ | Decays geometrically | MA($q$): $P = 0$, $Q = q$ |
| Both decay | Both decay | ARMA: start with $P = 1..2$, $Q = 1$ |
| Peaks at a fixed lag (24, 168) | | A cycle is left: add that period to `fourier` rather than raising P |

Or let `autoarma` choose by information criterion:

$$\text{AIC} = 2k - 2\ln\hat L, \qquad \text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}, \qquad \text{BIC} = k\ln n - 2\ln\hat L$$

for $k$ parameters, $n$ data, and maximized likelihood $\hat L$. BIC
penalizes parameters most and picks smaller models; AICc (the default)
corrects AIC for short series. `autoarma` always runs on the whole signal,
so with segmenting it picks one order for all segments.

Guidance:

- Small orders usually suffice once cycles are removed. $P \le 3$ and
  $Q \le 2$ cover most hourly energy data.
- With daily segments of 24 points, each segment has very few data for
  fitting. Keep $P + Q$ small (1 to 3), or the coefficients will vary
  wildly between segments.
- Raising P to capture a daily pattern is a sign the `fourier` periods are
  incomplete.
- Check the residual of `arma` itself (`saveResiduals`): it should be white
  noise (`references/validation.md`).

## Per-segment fits see little data

Every per-segment algorithm is fitted to one segment at a time: 24 points
for a daily segment with hourly measurements. That is enough for a few
Fourier terms and a low-order ARMA, but not for a distribution. A
per-segment `gaussianize` builds its quantile map from those 24 values
(RAVEN warns that `n_quantiles` was reduced). Its inverse is bounded by
the training values, so in sampling each segment's values stay within the
range that segment had in training. No sample can exceed the most extreme
day in its cluster.

When the spread or tails matter, prefer a global `preserveCDF` (which maps
samples to the whole training distribution), or longer segments. Inspect
the upper tail of the duration curve either way (`references/validation.md`).

## Fourier periods

- Start from the physical cycles: 24 h (daily), 168 h (weekly), 8760 h
  (yearly), and harmonics of each ($24/2$, $24/3$, ...).
- A periodogram of the signal, $I(f) = \frac{1}{n}\left|\sum_t x_t e^{-2\pi i f t}\right|^2$,
  shows which periods carry power; peaks at $f = 1/k$ are candidates.
- Each period adds two parameters per segment. Too many periods overfit:
  the deterministic part starts to reproduce the training noise, and the
  samples look too much like the training data with too little spread.
- With daily segments, only periods up to 24 h can be fitted per segment.
  Weekly and yearly periods must be global.

## When to keep it simple

The model only needs to reproduce what drives the HERON result. For
dispatch economics that is usually the daily shape, the level by season,
the spread of values, and how long high or low periods last (the duration
curve and autocorrelation). A simpler model that gets these right is
better than a complex one that overfits. Validate against those features
(`references/validation.md`) before adding algorithms.

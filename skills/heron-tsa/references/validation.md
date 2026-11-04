# Judging goodness of fit

A synthetic history model is good enough when its samples reproduce the
features of the data that drive the HERON result, and the variation between
samples is plausible. No single number says so. Use the checks below,
numerically or by eye, in roughly this order. The user decides which to
compute and how; each can be done by hand, in a spreadsheet, or with a
short script.

Notation: training data $x_t$, $t = 1..n$; samples $\tilde x^{(s)}_t$,
$s = 1..S$, from `synthetic.csv` (`assets/train_synthetic_history.xml`
writes $S = 10$). Compare each statistic of the data with its spread over
the $S$ samples, not with one sample.

## What the training run writes

| Output | Contents |
|--------|----------|
| `romMeta.xml` | Trained parameters per algorithm, target, and cluster (Fourier amplitudes and phases, ARMA coefficients and variance); cluster labels and the segments each cluster represents |
| `romMeta.xml`, `<residuals>` | With `<saveResiduals>True</saveResiduals>` on an **unsegmented** ROM: the residual after each algorithm, per target, as comma-separated values, at `<residuals>/<algorithm>/<target>` (e.g. `<residuals><ARMA><PRICE>`). Not written for segmented ROMs. |
| `synthetic.csv` | The samples, indexed by sample, year, and time |

## 1. Residuals of the stochastic model

The residual left after the last algorithm (`arma` or `varma`) should be
white noise: zero mean, constant variance, no autocorrelation, and roughly
normal after `gaussianize`. Structure left here is structure the model does
not reproduce.

`saveResiduals` writes residuals only for an unsegmented ROM. For a
segmented model, train a copy of the same chain without `<Segment>` (or on
one representative stretch of data) with `saveResiduals` on, and check its
residuals; the per-segment fits use the same algorithms.

- **Autocorrelation.** For residual $e_t$ with mean $\bar e$,

  $$\hat\rho(k) = \frac{\sum_{t=1}^{n-k}(e_t-\bar e)(e_{t+k}-\bar e)}{\sum_{t=1}^{n}(e_t-\bar e)^2}$$

  For white noise $\hat\rho(k)$ falls inside $\pm 1.96/\sqrt n$ for about 95%
  of lags. Spikes at lags 24, 168, ... mean a cycle is left: add that
  period to `fourier`. A slow decay over many lags means the AR order is
  too low, or the signal needs `differencing`.
- **Ljung-Box test.** Combines the first $h$ lags:

  $$Q = n(n+2)\sum_{k=1}^{h}\frac{\hat\rho(k)^2}{n-k} \;\sim\; \chi^2_{h - P - Q}$$

  A $p$-value below 0.05 rejects white noise. With $n$ in the thousands
  this test rejects small, unimportant structure; use it alongside the ACF
  plot, not instead of it.
- **Normality.** A Q-Q plot of the residual against $N(0,1)$ quantiles
  should be close to a straight line. Heavy tails mean extremes will be
  under-generated: revisit the distribution transform.
- **Constant variance.** Plot $e_t^2$ or a rolling variance through the
  year. A seasonal pattern means the spread changes with season: segment
  and cluster, or use a log or arcsinh transform.

## 2. Distribution

Compare the distribution of all sample values with the data.

- **Moments.** Mean $\bar x$, standard deviation $s$, skewness
  $\frac{1}{n}\sum (x_t-\bar x)^3/s^3$, and the 1st, 5th, 50th, 95th, and
  99th percentiles. The data's value should fall inside the range over
  samples.
- **CDF distance.** With empirical CDFs $F$ (data) and $\tilde F$
  (samples), the Kolmogorov-Smirnov distance
  $D = \sup_x \lvert F(x) - \tilde F(x)\rvert$, or the area between them,
  $A = \int \lvert F(x) - \tilde F(x)\rvert\,dx$ (in the signal's units,
  so it can be compared with the signal's spread). RAVEN's
  `CDFAreaDifference` and `PDFCommonArea` metrics compute these within a
  RAVEN run if wanted. As with Ljung-Box, a KS test with large $n$ rejects
  trivial differences; read $D$ or $A$ as a size, not a test.
- **Bounds.** Count sample values outside the physical range (negative
  load, capacity factor above 1). Any is a problem: add `outtruncation` or
  `bounddomain`.

`preserveCDF` forces each sample to the training distribution, so this
check passes by construction when it is used. Check the autocorrelation
and duration of extremes instead.

## 3. Autocorrelation of the signal

Compare $\hat\rho(k)$ of the data with that of each sample, for lags up to
at least two cycle lengths (48 h for a daily cycle, 336 h for weekly).

- The sample ACFs should bracket the data's ACF at every lag.
- Short lags (1 to a few steps) reflect how quickly the signal moves, and
  so ramping and storage cycling in the dispatch.
- Lags at the cycle period reflect the strength of the cycle.
- Lags beyond one segment show whether multi-day persistence is reproduced.
  Per-segment models lose it (`references/segmenting.md`).

For several signals, compare the cross-correlation too:

$$\hat\rho_{xy}(k) = \frac{\sum_t (x_t-\bar x)(y_{t+k}-\bar y)}{\sqrt{\sum_t (x_t-\bar x)^2 \sum_t (y_t-\bar y)^2}}$$

At $k = 0$ this is the ordinary correlation between, for example, load and
price. Independent `arma` models keep only the part of it carried by
shared cycles; `varma` is needed for the rest
(`references/choosing-models.md`).

## 4. Duration curve

Sort each year's values from largest to smallest and plot them against
rank (or fraction of the year): $x_{(1)} \ge x_{(2)} \ge \dots \ge x_{(n)}$
against $i/n$. This is the inverse of the exceedance function
$P(X > x)$: the fraction of time the signal exceeds each level.

- It is the distribution read as hours: how many hours prices exceed a
  given level, or wind is below a level. Revenue for a dispatchable unit or
  storage depends directly on it.
- The ends matter most. Compare the top and bottom 1 to 5% of the curve:
  peak prices, minimum net load.
- Plot the sample curves as a band and the data on top. The data should
  fall inside the band along its whole length.

The duration curve ignores time order. Also compare **run lengths**: for a
threshold $c$, the lengths of consecutive runs with $x_t > c$ (or $< c$).
The distribution of run lengths in the samples should match the data's.
This decides how much energy storage can shift and whether a low-wind
spell outlasts it.

## 5. Average profiles

Average by hour of day, $\bar x_h = \operatorname{mean}\{x_t : \text{hour}(t) = h\}$,
and by month or season. Compare data and samples on one plot, with the
sample spread as a band. Check also the standard deviation by hour, which
shows whether the spread changes through the day as in the data.

For a clustered model, compare per cluster as well: the samples of each
cluster against the training days in it (cluster labels are in
`romMeta`).

## 6. By eye

Plot the data and two or three samples over the same week, for a summer
and a winter week. They should be indistinguishable in character: the same
roughness, the same size of swings, the same daily shape. Common problems
are easy to see this way:

| Looks like | Likely cause |
|------------|--------------|
| Samples identical to each other | No stochastic algorithm, or its variance near zero |
| Samples identical to the data | Too many Fourier periods; the deterministic part has fitted the noise |
| Too smooth | AR order too high with a small variance, or the residual variance was absorbed by too many periods |
| Too noisy | Cycle missing from `fourier`; variance is carrying structure |
| Steps or jumps at midnight | Segment boundaries; expected with per-segment models, a problem if the study is sensitive to overnight ramps |
| Negative or out-of-range values | Missing bounds transform |
| Every day the same shape | Too few clusters, or no segmenting with a strong seasonal change |

## Holding out data

Statistics computed on the training data reward overfitting. When there is
enough data, train on all years but one and compare the samples with the
held-out year using the checks above. An interpolated model can be checked
the same way on an intermediate year left out of the training list.

## What matters for HERON

Prioritize the checks by what drives the study:

| Study driven by | Check most carefully |
|-----------------|----------------------|
| Energy arbitrage, storage | Daily profile, short-lag autocorrelation, run lengths, top and bottom of the price duration curve |
| Capacity for peak demand | Upper tail and duration curve of load or net load; joint extremes with renewables |
| Renewable firming | Run lengths of low output, cross-correlation with load, multi-day persistence |
| Average economics (mean NPV) | Mean and daily and seasonal profiles; tails matter less |

A model that passes the checks for the features that drive the economics is
good enough, even if it fails others.

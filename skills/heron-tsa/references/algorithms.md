# TSA algorithms in a `SyntheticHistory` ROM

Each algorithm is a child node of `<ROM subType="SyntheticHistory">`.
Node names are case-sensitive and are listed exactly as RAVEN expects them.

## How the chain works

- **Training.** Algorithms with `global="True"` are fitted first, on the
  whole signal, then the others are fitted on each segment. Within each
  group the order is top to bottom. Each algorithm is fitted to the
  residual left by the ones before it. A transformer replaces the signal
  with its residual; a characterizer-only algorithm (`autoarma`) leaves it
  unchanged.
- **Sampling.** The order reverses. Per-segment algorithms run bottom to
  top: the stochastic model generates noise, then each transformer above it
  inverts its step (adds its cycle back, undoes its transform). The global
  algorithms then run bottom to top on the reassembled signal.
- **Targets.** Each node acts only on the signals in its `target`. Signals
  can have different chains.

## Attributes on every node

| Attribute | Meaning |
|-----------|---------|
| `target` | Required. Comma-separated signal names. |
| `seed` | Random seed, for algorithms with random elements. |
| `global` | `True`: fit on the whole signal, not per segment. Default `False`. Only matters when the ROM is segmented. |

Optional ROM-level nodes: `<pivotParameter>` (default `time`) and
`<saveResiduals>` (default `False`; writes each algorithm's residual into
the ROM metadata, for `references/validation.md`). `saveResiduals` only
works on an unsegmented ROM: the per-segment copies RAVEN makes when
segmenting do not keep the setting, and no residuals are written.

## Properties

| Node | Kind | Stochastic | Accepts NaN | Leaves NaN | Per segment | Clusterable features |
|------|------|:---:|:---:|:---:|:---:|------|
| `fourier` | cycle | | yes | | yes | `sin`, `cos`, `intercept` |
| `PolynomialRegression` | trend | | yes | | yes | |
| `STL` | trend + seasonal | | no | | yes | |
| `wavelet` | decomposition | | no | | yes | |
| `arma` | stochastic | yes | yes | | yes | `ar`, `ma`, `var`, `const` |
| `varma` | stochastic, joint | yes | yes | | yes | `ar`, `ma`, `cov`, `const` |
| `MarkovAR` | stochastic, regimes | yes | no | | yes | |
| `autoarma` | order selection | | yes | | **no** (always global) | |
| `zerofilter` | filter | | yes | yes | yes | |
| `differencing` | transform | | yes | yes | yes | |
| `gaussianize`, `quantiletransformer` | distribution | | yes | | yes | |
| `preserveCDF` | distribution | | yes | | yes | |
| `logtransformer`, `arcsinhtransformer`, `tanhtransformer`, `sigmoidtransformer` | function | | yes | | yes | |
| `outtruncation`, `bounddomain` | bounds | | yes | | yes | |
| `standardscaler`, `minmaxscaler`, `maxabsscaler`, `robustscaler` | scaling | | yes | | yes | see below |

"Leaves NaN" means the residual passed down the chain contains NaN, so
every later algorithm on that target must accept NaN.

## Deterministic structure

### `fourier`

Fits $\sum_k \left[a_k \sin(2\pi t/k) + b_k \cos(2\pi t/k)\right] + c$ by
least squares, for the listed periods $k$ only (not an FFT). The fitted
waves are subtracted; sampling adds them back unchanged, so every
realization has the same cycles.

| Child | Meaning |
|-------|---------|
| `<periods>` | Required. Comma-separated periods, in pivot units. |

Periods longer than a segment cannot be resolved inside it; put them in a
separate `fourier` node with `global="True"`. If the periods are nearly
collinear over the data (condition number $\ge 30$), RAVEN fits them one
at a time instead of jointly and says so in the log.

### `PolynomialRegression`

Fits a polynomial in the pivot, $\sum_{i=0}^{d} \beta_i t^i$, by least
squares and subtracts it.

| Child | Meaning |
|-------|---------|
| `<degree>` | Required. Polynomial degree, at least 1. |

The trend is fitted to the training pivot range. It is reproduced, not
extrapolated, so it describes a within-year trend, not growth between
years (use `interpolate` for that).

### `STL`

Seasonal-trend decomposition by LOESS (statsmodels `STL`). Removes the
trend and seasonal components, leaving the remainder. Sampling adds back
the same trend and seasonal curves. Cannot accept NaN.

| Child | Meaning |
|-------|---------|
| `<period>` | Period of the seasonal component, in time steps. |
| `<seasonal>` | Length of the seasonal smoother (odd). statsmodels default if omitted. |
| `<trend>` | Length of the trend smoother (odd). statsmodels default if omitted. |

### `wavelet`

Single-level discrete wavelet transform (PyWavelets `dwt`); sampling
reconstructs the same signal with `idwt`. Cannot accept NaN. Requires
`pywavelets`.

| Child | Meaning |
|-------|---------|
| `<family>` | Required. Wavelet, e.g. `haar`, `db4`, `sym5`, `coif3`. |

## Stochastic models

### `arma`

$$A_t = c + \sum_{i=1}^P \phi_i A_{t-i} + \epsilon_t + \sum_{j=1}^Q \theta_j \epsilon_{t-j}, \qquad \epsilon_t \sim N(0, \sigma^2)$$

fitted independently for each target (statsmodels `ARIMA`). Sampling draws
a new initial state from the model's stationary distribution and new
shocks $\epsilon_t$, so each realization is different.

| Child or attribute | Meaning |
|--------------------|---------|
| `<P>` | AR order. One value for all targets, or one per target in `target` order. |
| `<Q>` | MA order, same form. |
| `auto_select` | `True`: take P and Q from a preceding `autoarma`; omit `<P>` and `<Q>`. Default `False`. |
| `reduce_memory` | `True`: lower memory use during training. Default `False`. Do not combine with algorithms after `arma`: with it on, `arma` passes its input through instead of its residual. |

Use a `gaussianize` node before `arma` rather than the deprecated
`gaussianize="True"` attribute on `arma`.

Targets in one `arma` node are modeled independently: the realizations of
different signals are uncorrelated except through any shared deterministic
structure. Use `varma` for correlated noise.

### `varma`

The vector form of `arma`: $\phi_i$ and $\theta_j$ are $k \times k$
matrices for $k$ targets, and $\epsilon_t \sim N(0, \Sigma)$ with a full
covariance matrix (statsmodels `VARMAX`). Captures correlation between
signals, including lagged effects.

| Child | Meaning |
|-------|---------|
| `<P>` | AR order (one value). |
| `<Q>` | MA order (one value). |

The number of parameters grows as $k^2(P+Q)$; keep $k$ and the orders
small. Training is much slower than `arma`.

### `MarkovAR`

Markov-switching autoregression (statsmodels `MarkovAutoregression`): a
hidden state $S_t \in \{1..r\}$ follows a Markov chain, and the mean, AR
coefficients, and noise variance can depend on it. For signals that switch
between regimes (normal and scarcity prices, calm and gusty wind).

| Child or attribute | Meaning |
|--------------------|---------|
| `<P>` | AR order. |
| `<MarkovStates>` | Number of regimes $r$. |
| `switching_ar` | AR coefficients depend on the regime. Default `True`. |
| `switching_variance` | Noise variance depends on the regime. Default `True`. |
| `switching_trend` | Mean depends on the regime. Default `True`. |

Cannot accept NaN. Fitting can fail to converge or produce NaN parameters;
RAVEN stops with an error naming the parameter. Reduce `<MarkovStates>`,
`<P>`, or the number of switching terms if it does.

### `autoarma`

Chooses P and Q for each target by minimizing an information criterion
(`statsforecast` `AutoARIMA`, non-seasonal, stationary). It does not
transform the signal. Always fitted on the whole signal, before
segmenting. Requires `statsforecast`.

| Child | Meaning |
|-------|---------|
| `<P_upper>` | Largest P to try. Default 5. |
| `<Q_upper>` | Largest Q to try. Default 5. |
| `<criterion>` | `aic`, `aicc` (default), or `bic`. |
| `<use_approximation>` | Faster stepwise search. Default `False`. Results have differed between operating systems with it on. |

Follow it with `<arma auto_select="True">` on the same targets, after the
same transforms.

## Transforms

### Distribution

- **`gaussianize`**: maps the data to a standard normal through its
  empirical quantiles; the inverse maps back. Put it just before `arma` or
  `varma`, whose noise is Gaussian. Attribute `nQuantiles` (default 1000,
  capped at the number of data).
- **`quantiletransformer`**: as `gaussianize`, with attribute
  `outputDistribution` = `normal` (default) or `uniform`.
- **`preserveCDF`**: does nothing in training. In sampling, maps the
  generated values through their own empirical CDF and then the training
  data's inverse CDF, so each sample has the training distribution. Put it
  first in the chain (as global, if other algorithms are). It changes the
  distribution of every sample to match, which hides distribution errors
  from validation; check the autocorrelation instead.

### Function transforms

Applied to each value, inverted on sampling.

| Node | Forward | Inverse | Domain |
|------|---------|---------|--------|
| `logtransformer` | $\ln x$ | $e^x$ | $x > 0$; raises an error otherwise |
| `arcsinhtransformer` | $\sinh^{-1} x$ | $\sinh x$ | all $x$; log-like for large $\lvert x\rvert$ |
| `tanhtransformer` | $\tanh x$ | $\tanh^{-1} x$ | all $x$ |
| `sigmoidtransformer` | $1/(1+e^{-x})$ | logit | all $x$ |

### Bounds and filters

- **`outtruncation`**: attribute `domain` = `positive` or `negative`
  (required). Does nothing in training; in sampling reflects out-of-range
  values back ($\lvert x\rvert$ or $-\lvert x\rvert$). For signals that cannot be
  negative. Put it first, so it acts on the final output.
- **`bounddomain`**: children `<lowerBound>` and `<upperBound>` (default
  $\mp\infty$). Clips to $[L, U]$ in both training and sampling. For
  bounded signals such as capacity factors.
- **`zerofilter`**: masks values with $\lvert x\rvert <$ `tol` (default
  $10^{-8}$) so later algorithms ignore them, and puts them back in
  sampling at the same time steps. For solar at night. Attribute `fill`:
  `drop` (default; masked values become NaN) or a number to substitute.
  The zeros come back at the training positions in every sample.
- **`differencing`**: child `<order>` $n$. Replaces the signal with its
  $n$-th difference; sampling integrates back from the training initial
  values. For signals with a unit root (random-walk behavior). Leaves $n$
  NaN values at the end of the residual.

### Scaling

`standardscaler` ($(x-\mu)/s$), `minmaxscaler` (to $[0,1]$),
`maxabsscaler` (to $[-1,1]$), `robustscaler` (median and interquartile
range). They are linear, so they do not change the shape of the
distribution. Their parameters can be cluster features (`mean`, `scale`,
`dataMin`, `dataMax`, `center`), which lets clustering group segments by
level and spread.

## Not for HERON

- **`rwd`**: a characterizer only (sliding-window features). It generates
  nothing.
- **`filterbankdwt`**: only for RAVEN's `MultiResolutionTSA` ROM, not
  `SyntheticHistory`.

---
name: heron-tsa
description: Train, check, and save a RAVEN SyntheticHistory ROM (time series model) for use as a HERON <ARMA> data generator. Use when the user has historical time series data (prices, loads, wind, solar) and needs a stochastic synthetic history model before building a HERON analysis; is choosing TSA algorithms or transformations (fourier, arma, varma, gaussianize, zerofilter, ...); is setting up segmenting, clustering, or multi-year interpolation; or is judging how well a trained model reproduces the data. For using a trained model in a HERON input, use the heron skill.
---

# Training synthetic history models for HERON

HERON does not train time series models. RAVEN does, in a separate run
before any HERON analysis. The result is a serialized `SyntheticHistory`
ROM (a `.pk` file) that HERON samples through an `<ARMA>` data generator,
drawing a new random history for each realization.

The model has two jobs: reproduce the structure of the data that drives the
economics, and give realistic variability between realizations. Spend
effort on checking it, because every HERON result inherits its errors.

## Workflow

1. **Look at the data.** Plot each signal. Note the time step, the number
   of years, cycles (daily, weekly, yearly), trends, zeros (solar at night),
   bounds (capacity factors in [0, 1]), spikes, and how the signals move
   together. Put the data in RAVEN's layout (`references/training-data.md`).
2. **Choose the algorithm chain.** Deterministic parts first (cycles,
   trend), then distribution transforms, then a stochastic model of what is
   left. See `references/choosing-models.md`; every node and option is in
   `references/algorithms.md`.
3. **Choose the segmenting.** Daily segments clustered into a few
   representative days is the usual choice for HERON; multi-year data uses
   `grouping="interpolate"`. See `references/segmenting.md`.
4. **Write the training input.** Start from `assets/train_synthetic_history.xml`.
   It loads the data, trains, writes the model and its metadata, and
   samples a few histories for checking.
5. **Train:** `raven_framework train.xml`.
6. **Check the fit.** Compare the samples with the data and inspect the
   residuals (`references/validation.md`). Go back to step 2 or 3 until the
   features that matter for the study are reproduced.
7. **Hand off to HERON.** Run `heron inspect model.pk`. It prints the
   signals, the time and year variables, the years covered, and the
   `<time_discretization>` to put in `<Case>` for each `evalMode`. Using
   the model in a HERON input is covered by the heron skill
   (`../heron/references/guide/time-series.md`).

## What HERON requires of the ROM

| Requirement | Why |
|-------------|-----|
| `subType="SyntheticHistory"` | The supported synthetic history ROM. |
| `<Features>scaling</Features>`, exactly | HERON evaluates the model with `scaling = 1`, and only treats a ROM whose sole feature is `scaling` as a synthetic history. |
| The pivot (time variable) listed in `<Target>` | RAVEN requires it. Its name becomes `<time_variable>` in HERON. |
| Uniform time steps | HERON's dispatch grid is uniform and does not interpolate. |
| `<evalMode>full</evalMode>` in a clustered `<Segment>` | HERON's default `evalMode="clustered"` overrides the trained mode. HERON's `evalMode="full"` does not; it uses the trained mode, and RAVEN's default (`truncated`) is not one HERON supports. Training with `full` keeps both options open. |
| Multi-year: integer years in `<macroParameter>`, covering at least the project life | HERON truncates a longer model and rejects a shorter one. A single-year model is repeated for every project year. |
| Signal names | Each name in the HERON `<ARMA variable="...">` must be a target of the ROM. |

## Gotchas

- **Order is the model.** Algorithms are fitted top to bottom, each on what
  the previous ones left (the residual). Sampling runs bottom to top. So a
  transform listed early is applied to the final output last.
- **Deterministic vs. stochastic.** Only `arma`, `varma`, and `MarkovAR` add
  randomness. A chain without one of them gives the same history every
  realization.
- **Missing values.** `zerofilter` and `differencing` leave NaN in the
  residual. `MarkovAR`, `STL`, and `wavelet` cannot accept NaN and raise an
  error if they come after either.
- **Fourier periods are in pivot units.** With an hourly pivot counted in
  hours, a daily cycle is `24`; with a pivot in seconds, it is `86400`.
- **Periods longer than a segment** cannot be fitted per segment. Put those
  in a `fourier` node with `global="True"`.
- **Each segment is sampled independently.** Correlation across segment
  boundaries is lost unless a global algorithm carries it (see
  `references/segmenting.md`).
- **Seeds.** Set `seed` on each stochastic node for reproducible training
  and samples.

## References

| File | Read when |
|------|-----------|
| `references/training-data.md` | Preparing the training CSVs and the `HistorySet` |
| `references/algorithms.md` | Writing any TSA node: every algorithm, its options, and its properties |
| `references/choosing-models.md` | Deciding which algorithms and transforms to use, in what order, and choosing P and Q |
| `references/segmenting.md` | `<Segment>`: segments, clusters, multi-year interpolation, `evalMode`, and what HERON's time grid becomes |
| `references/validation.md` | Judging goodness of fit: residual checks, distribution, autocorrelation, duration curves, profiles |
| `assets/train_synthetic_history.xml` | Starting a training input |

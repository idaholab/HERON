# Error catalog

Each entry: symptom (message text, abbreviated), likely cause, fix. Messages
are taken from HERON's source; the file that raises each is given so you can
read the surrounding code. Where RAVEN raises the error, the entry says so.

Search this file for a fragment of the message; do not read it whole.

This is not an exhaustive list. For an error that is not here, find the
message in HERON's source (`../heron/references/source-map.md`) and read the
check that raises it.

## Parsing errors (`heron input.xml`)

| Symptom | Cause and fix |
|---------|---------------|
| `<Case> node is missing from HERON input file!` (also `<Components>`, `<DataGenerators>`, `No <Component> nodes were found`) | Required top-level node missing or misspelled, or the root is not `<HERON>`. `input_loader.py` |
| `No <mode> node was provided in the <Case> node!` | Add `<mode>sweep</mode>` or `<mode>opt</mode>`. `Cases.py` |
| `No <dispatch> node was provided in the <Case> node!` | Text says `<dispatch>` but the node is `<dispatcher>` (add `<dispatcher><pyomo/></dispatcher>`). `Cases.py` |
| `<time_discretization> node was not provided` | Add it. `Cases.py` |
| `Invalid time discretization choices! Must specify any of the following pairs...` | Give two of `end_time`, `num_steps`, `time_interval`. `Cases.py` |
| `<economics> node missing from component "X"!` | Every component needs `<economics>` (with `<lifetime>`). `Components.py` |
| `Currently each Component can only have one interaction` | A component has more than one of `produces`/`stores`/`demands`. Split into two components. `Components.py` |
| `Component "X" node <capacity> expected a ValuedParam to define its value source, but none was found! Options include: ...` | The node is empty or has an unsupported child. Add one ValuedParam. `ValuedParamHandler.py` |
| `Only one ValuedParam type can be used per node` | Two value nodes in one slot (e.g. `fixed_value` and `sweep_values`). `ValuedParamHandler.py` |
| `If multiple resources are active, "capacity" requires a "resource" specified!` | Add `resource="..."` to `<capacity>` for a producer with `<consumes>`. `Components.py` |
| `Any component that consumes a resource must have a transfer function describing the production process!` | Add `<transfer>`. `Components.py` |
| `Transfer function for Component "X" has a mismatch with consumed and produced!` followed by lists | A resource is consumed or produced but not in the transfer function, or the reverse. The message lists which. `TransferFuncs/TransferFunc.py` |
| `Unrecognized transfer function for component "X"` | Use `<ratio>` (or deprecated `<linear>`) or `<poly>`. `TransferFuncs/Ratio.py` |
| `"linear" has been deprecated` (warning) | Rename `<linear>` to `<ratio>`. |
| `Ramp limit must be (0, 1]` | `<ramp_limit>` out of range. `Components.py` |
| `Value for <max_charge_rate> must be greater than 0 and less than or equal to 1` (also `max_discharge_rate`) | Fractions of capacity per step. `Components.py` |
| `Multiple storage resources given for component "X"` | A storage holds exactly one resource. |
| `Initial storage level for "X" was not provided! Defaulting to 0%.` (warning) | Informational. Add `<initial_stored>` if you need a different start (ignored while `periodic_level` is true). |
| `No <driver> node provided for CashFlow X!` / `No <reference_price> node provided for CashFlow X!` | Both are required. `Economics.py` |
| `Unrecognized "CashFlow" node: ...` | A child node that is not part of CashFlow. Check spelling. `Economics.py` |
| `Unrecognized DataGenerator: ...` | Use `ARMA`, `CSV`, `Function`, or `ROM`. `input_loader.py` |
| `Multiple sources in <DataGenerators> node specify values for variables [...]` | Two generators list the same `variable`. Each variable may appear once. `input_loader.py` |
| `Requested source "X" for component "Y" was not found!` | The text of a `<ARMA>`/`<CSV>`/`<Function>`/`<ROM>` value does not match any generator `name`. `input_loader.py` |
| `File not found for <DataGenerator><TYPE> named "X". Looked in: ... Given location: ...` | Path wrong. See path rules in `../heron/references/spec/data-generators.md`. `Placeholders.py` |
| `ERROR trying to find %HERON_DATA%...` / `path ... found in HERON_DATA environment variable does not seem to exist` | `%HERON_DATA%` could not be resolved. It points to HERON's workshop data, which is not part of a normal install. Replace the prefix with a plain path to the file. `Placeholders.py` |
| `Variable "V" requested for "X" but not found in "<file>"! Found: [...]` | The CSV has no such column header. `Placeholders.py` |
| `Module "..." for function "X" was not found!` | The `<Function>` Python file path is wrong. `Placeholders.py` |
| `Levelized Cost metric was selected, but no <levelized_cost> node was found in component Cash Flows!` | Put `<levelized_cost/>` in one cash flow's `<reference_price>`. `Cases.py` |
| `Levelized Cost metric requires a nonlinear optimization in the inner step...` | Set a nonlinear `<solver>` such as `ipopt`. `Cases.py` |
| `Multiple optimization algorithms found, only first algorithm provided...will be used` (warning) | Put one algorithm in `<optimization_settings>`. |
| `System may be overloaded and greatly increase run time!` (warning) | `<parallel>` requests more processes than the machine has. Lower them. |
| `HERON config file not found at ...! Has HERON been installed as a plugin in a RAVEN installation?` | Register HERON with RAVEN (`install_plugins.py`). `_utils.py` |
| RAVEN input-spec error naming an unknown node or a bad value (printed while parsing) | A node is not part of HERON's input. Check spelling, nesting, and allowed values in `../heron/references/spec/*.md`. |

## Workflow generation errors (`heron input.xml`, after parsing)

| Symptom | Cause and fix |
|---------|---------------|
| `Mixing ARMA and CSV sources is not yet supported!` | Use only one kind of time series source. `templates/template_driver.py` |
| `raven_framework not in PATH and not at ...` | RAVEN's executable could not be found. Activate the RAVEN environment or fix the plugin registration. `templates/bilevel_templates.py` |
| `Template does not recognize optimization strategy ...` | `<algorithm>` must be `BayesianOpt` or `GradientDescent`. `templates/bilevel_templates.py` |
| `Nothing that requires a sampler was found.` (debug mode) | The case has no ARMA, CSV, sampled capacity, or uncertain parameter. Debug mode needs at least one. `templates/debug_template.py` |
| `ValueError` or `AttributeError` raised from `templates/` with no message about your input | Often an unsupported combination. Reduce the input to the smallest case that still fails, and compare it against `../heron/assets/minimal_sweep.xml` or `minimal_opt.xml` to find the feature that triggers it. |

## RAVEN errors (`raven_framework outer.xml`)

| Symptom | Cause and fix |
|---------|---------------|
| File not found for `inner.xml`, `heron.lib`, or `../write_inner.py` | Run `raven_framework` on the generated `outer.xml` from the directory `heron` wrote to. Do not move `outer.xml` away from `inner.xml` and `heron.lib`. Re-run `heron` to regenerate. |
| `RAVENInterface WARNING: Could not find .../.ravenStatus, assuming failed RAVEN run.` | The inner run died. Open `generalOut` in that run's subfolder of `<Case name>_o/` for the real error (`SKILL.md`, "Inner runs"). RAVEN |
| Outer run finishes but results are missing or the output database is empty (no specific message) | The inner runs died before writing results. Same as above. |
| Python `ModuleNotFoundError` for `HERON`, `TEAL`, or `pyomo` | The environment is not the RAVEN one, or the plugin is not installed. |
| Pickle or ROM load errors on an ARMA `.pk` | The model was trained with a different RAVEN version, or the file is corrupted. Retrain with the same RAVEN (heron-tsa skill). |

## Dispatch errors (inside the inner run or debug run)

| Symptom | Cause and fix |
|---------|---------------|
| `Solve was unsuccessful, see log file located at ".../constraint_violations.log"... Status: warning Termination: infeasible` | The dispatch constraints cannot all hold. Read the infeasible constraints printed to the console (the log file is often empty; see `references/infeasible-solves.md`). Usual causes: a `fixed` producer whose output has nowhere to go; a `fixed` demand larger than total capacity; a resource with no producer or no consumer (check spelling); ramp limits or `minimum` that conflict with a fixed signal; a storage `periodic_level` with a forced draw. Add an `independent` market or storage to absorb or supply the resource, or fix the sign or capacity. `dispatch/pyomo_dispatch.py` |
| `Solve was unsuccessful ... Status: warning Termination: unbounded` | The solver reports that the objective can improve without limit. HERON raises this for any result that is not status `ok` with termination `optimal`. Usually a variable with no upper bound appears in the objective: check component signs and capacities, and any `<Function>` price or driver, in the printed Pyomo model. |
| Missing solver: `RuntimeError: Attempting to use an unavailable solver. The SolverFactory was unable to create the solver "X"...` | The name in `<solver>` is not an installed solver in the RAVEN environment. Install the solver or set `<solver>` to one that is (default `cbc`, `glpk` on Windows). |
| `Tolerance setting not available for solver 'X'` | `<tol>` is not supported for that solver. Remove `<tol>`. `dispatch/pyomo_dispatch.py` |
| `Window length of 1 detected, which is not supported.` | The number of time steps leaves a final window of one step with `rolling_window_length`. Pick a window length that does not leave a remainder of 1. `dispatch/pyomo_dispatch.py` |
| `Exceeded validation attempt limit!` | A `<validator>` kept reporting violations for 100 re-solves of one window. The limits it imposes conflict with the rest of the dispatch; loosen the validator's settings (e.g. `<delta>`, `<tolerance>`). `dispatch/pyomo_dispatch.py` |
| `Convergence not reached after 10 iterations.` | A storage `<strategy>` function is governing the dispatch and the iteration with the optimizer is not settling. Simplify the function or the coupling. `dispatch/pyomo_dispatch.py` |
| `Requested number of steps (N) does not match "time" history provided (M)!` | `<time_discretization>` differs from the time series. Run `heron inspect <file>` and use the `<time_discretization>` it prints for the `evalMode` in use. `DispatchManager.py` |
| `Requested end time (T) is greater than time variable ...` / `Requested start time ... is less than ...` | Case longer than the history. `DispatchManager.py` |
| `Expected ARMA variable "V" was not passed to DispatchManager!` | `variable` not in the model's signals (`heron inspect <file>` lists them). `DispatchManager.py` |
| `Provided macro variable "Year" is length N, but expected project life is M!` | The history's number of years differs from `<ProjectTime>` (less one construction year). `DispatchManager.py` |
| `An interpolated ARMA ROM was used, but there are less interpolated years ...` / `"X" interpolates N macro steps, but project life is M!` | Model covers fewer years than the project. Retrain or lower `<ProjectTime>`. `DispatchManager.py`, `Placeholders.py` |
| `Synthetic histories are not of consistent shape!` (with `History Shapes` and `Index Map` printed) | Signals from different models have different lengths. Retrain them on the same time axis. `DispatchManager.py` |
| `variable "V" was not found among the RAVEN variables!` | A `<ARMA>`/`<CSV>` value names a variable that did not reach the dispatch. Check `variable` against the generator's list. `ValuedParams/SyntheticHistory.py`, `StaticHistory.py` |
| `Attempted to access variable "V" beyond the end of its length!` | Signal shorter than the dispatch window. Match time grids. |
| `From Function "F" method "m" expected ... to return with form (results_dict, meta_dict)` | A `<Function>` must return `(data, meta)`, both dicts. See `../heron/references/spec/functions.md`. `Placeholders.py` |
| `While calculating minimum operating level ... an invalid percent was provided` / `...initial storage level ... invalid percent` | `<minimum>` or `<initial_stored>` evaluated outside 0 to 1. `Components.py` |
| `Some variables were missing or None!` | A component's evaluation lacks a needed value (see the warnings printed just above). `Components.py` |
| A cash flow `Function` fails with a Pyomo error about expressions, `if`, `min`, `max`, or NumPy | The function received symbolic activity. Use only plain arithmetic on activity. See `../heron/references/spec/functions.md`. |

## Suspicious results

No error message. Treat each row as a hypothesis to check in debug mode.

| Symptom | Likely cause and check |
|---------|------------------------|
| NPV very large or very negative, same magnitude in every design | Sign error in a cash flow driver or price (`../heron/references/spec/economics.md`); units mismatch between price and activity. |
| Revenue shows up as a cost | Market sales without `<multiplier>-1</multiplier>` on the driver. |
| Capex does not change when capacity changes | Capex driven by a constant rather than `<variable>X_capacity</variable>`. |
| Optimum on the edge of `opt_bounds` | Bound is limiting. Widen it. |
| Every design has the same NPV | The swept variable is not used (wrong name), or the dispatch ignores the component (disconnected resource). |
| Storage level looks like it resets every day | The rolling window (default 24 steps) forces periodic level each window. See `rolling_window_length`. |
| Results change a lot between runs | Few realizations (`<num_arma_samples>`) or a tail statistic as the objective. Raise the sample count. |
| Optimizer stops early or wanders | Noisy objective, `persistence` too low, or `limit` hit. Look at the optimization path and `accepted` column in `opt_soln*.csv`. |
| `mean_NPV` column missing | You are looking at a flat sweep (single static history, no uncertain cash flow parameters), which writes plain `NPV` with no statistics. Bilevel runs always write `mean_`, `std_`, and `med_` columns. |
| `NaN` in results | A solve failed for some realization, or a negative number was raised to a fractional `scaling_factor_x` (the driver must be positive; use a multiplier). |
| Flat sweep and bilevel sweep disagree | Different column naming (`NPV` vs. `mean_NPV`); the flat workflow has no averaging. Compare like with like. |

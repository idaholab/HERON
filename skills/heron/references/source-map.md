# Source map

These references were written from HERON's source. When they disagree with
the code, the code is right: check it and tell the user about the mismatch.

## Where HERON is installed

Paths below are relative to the HERON directory, which depends on how HERON
was installed:

| Install | HERON directory | TEAL (economics plugin) |
|---------|-----------------|-------------------------|
| pip | `<site-packages>/HERON/` | `<site-packages>/TEAL/` |
| Source checkout, registered as a RAVEN plugin | the HERON checkout (often `raven/plugins/HERON/`) | the TEAL checkout (often `raven/plugins/TEAL/`) |

For a pip install, find `<site-packages>` in the environment HERON is
installed in with:

```bash
python -c "import HERON, os; print(os.path.dirname(HERON.__file__))"
```

## Topics

| Question | Source |
|----------|--------|
| What a `<Case>` node accepts, defaults, and metric/statistic names | `src/Cases.py` (`get_input_specs`, `stats_metrics_meta`, `economic_metrics_meta`) |
| What a component, producer, storage, or demand accepts | `src/Components.py` |
| Cash flow rules | `src/Economics.py`, `src/CashFlowUser.py` |
| ValuedParam types and where they are allowed | `src/ValuedParams/`, `src/ValuedParamHandler.py` |
| Transfer functions | `src/TransferFuncs/` |
| Data generators, path resolution, ARMA/CSV checks | `src/Placeholders.py`, `src/input_loader.py` |
| Time structure of a ROM or CSV (`heron inspect`) | `src/inspect_data.py`, `src/_utils.py` (`get_synthhist_structure`, `sample_synthhist`, `get_csv_structure`) |
| Which RAVEN workflow gets written and why | `templates/template_driver.py` |
| What the generated workflows contain | `templates/*.py`, `templates/xml/` |
| How dispatch is solved and failures are raised | `src/dispatch/pyomo_dispatch.py`, `PyomoModelHandler.py`, `PyomoRuleLibrary.py` |
| How results are computed per run | `src/DispatchManager.py` |
| Tax, inflation, depreciation, discounting | TEAL: `src/main.py`, `src/CashFlows.py`, `src/Amortization.py` |

`heron --definition` prints the input definition as HERON itself sees it.

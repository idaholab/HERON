# Source map

These references were written from HERON's source. When they disagree with
the code, the code is right: check it and tell the user about the mismatch.

| Question | Source |
|----------|--------|
| What a `<Case>` node accepts, defaults, and metric/statistic names | `src/Cases.py` (`get_input_specs`, `stats_metrics_meta`, `economic_metrics_meta`) |
| What a component, producer, storage, or demand accepts | `src/Components.py` |
| Cash flow rules | `src/Economics.py`, `src/CashFlowUser.py` |
| ValuedParam types and where they are allowed | `src/ValuedParams/`, `src/ValuedParamHandler.py` |
| Transfer functions | `src/TransferFuncs/` |
| Data generators, path resolution, ARMA/CSV checks | `src/Placeholders.py`, `src/input_loader.py` |
| Which RAVEN workflow gets written and why | `templates/template_driver.py` |
| What the generated workflows contain | `templates/*.py`, `templates/xml/` |
| How dispatch is solved and failures are raised | `src/dispatch/pyomo_dispatch.py`, `PyomoModelHandler.py`, `PyomoRuleLibrary.py` |
| How results are computed per run | `src/DispatchManager.py` |
| Theory (formulation, cash flows, workflows, time indexing) | `doc/theory_manual/` |
| Working examples | `tests/integration_tests/` (small, one feature each) and `tests/workshop/` (full studies) |

`src/`, `templates/`, `doc/`, and `tests/` exist only in a source checkout
of HERON, not in a pip install. `heron --definition` prints the input
definition as HERON itself sees it, and works in either.

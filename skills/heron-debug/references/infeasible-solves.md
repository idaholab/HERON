# Infeasible or failed solves

When a window's solve does not return `optimal`, HERON prints to the console
of that run (`generalOut` for an inner run):

1. `DEBUGG model pieces:`, the full Pyomo model: objective, then each variable
   with its bounds, then each constraint.
2. `Resource Map:`, which index each resource has for each component.
3. The infeasible constraints from Pyomo's `log_infeasible_constraints`, with
   their expressions and variable values, logged as
   `INFO:pyomo.util.infeasible:CONSTR <name>: ...`.
4. `Solve was unsuccessful, see log file located at ".../constraint_violations.log"`.

The log file is often empty: HERON configures the log file only after the
first logging call, so the first failure in a process is not written to it.
Read the console output instead.

How to read it:

- Constraint names (`src/dispatch/PyomoModelHandler.py`) say what each
  enforces. The index in brackets is the time step within the window.

  | Name | Enforces |
  |------|----------|
  | `<resource>_conservation` | Resource balance at each step |
  | `<comp>_<res>_capacity_constr`, `<comp>_<res>_minprod_constr` | Capacity (times capacity factor) and minimum |
  | `<comp>_ramp_up_constr`, `<comp>_ramp_down_constr`, `<comp>_ramp_freq_*` | Ramping |
  | `<comp>_<res>_..._transfer`, `<comp>_transfer_func` | Transfer function |
  | `<comp>_..._level_constr`, `..._charge_constr`, `..._discharge_constr` | Storage level and rates |
  | `..._vld_limit_constr_<i>` | A limit added by a `<validator>` |

  A violation at the first or last step of a window points to storage
  periodicity or initial level; one in the middle points to a signal value
  at that step.
- Activity variables are named `<comp>_production` (storage:
  `<comp>_level`, `_charge`, `_discharge`), printed with their bounds per
  step. A `fixed` component's variable has equal lower and upper bounds.
  For a `_conservation` violation, add up the `fixed` activity on that
  resource at that step by hand. If it cannot balance with every
  `independent` component at its bounds, the case is infeasible as written.
- Then use the dispatch causes in `references/error-catalog.md`, "Dispatch
  errors", and the formulation in `../heron/references/theory/dispatch.md`.

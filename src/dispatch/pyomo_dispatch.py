"""
  pyomo-based dispatch strategy
"""

import os
import sys
from functools import partial

import numpy as np
import pyomo.environ as pyo

from .Dispatcher import Dispatcher
try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils


class Pyomo(Dispatcher):
  """
    Dispatches using rolling windows in Pyomo
  """
  naming_template = {'comp prod': '{comp}|{res}|prod',
                     'comp transfer': '{comp}|{res}|trans',
                     'comp max': '{comp}|{res}|max',
                     'comp ramp up': '{comp}|{res}|rampup',
                     'comp ramp down': '{comp}|{res}|rampdown',
                     'conservation': '{res}|consv',
                    }

  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('Dispatcher', ordered=False, baseNode=None)
    # TODO specific for pyomo dispatcher
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'PyomoDispatcher' # identifying name
    self._window_len = 24         # time window length to dispatch at a time # FIXME user input

  def read_input(self, specs):
    """
      Read in input specifications.
      @ In, specs, RAVEN InputData, specifications
      @ Out, None
    """
    print('DEBUGG specs:', specs)

  def dispatch(self, case, components, sources, variables):
    """
      Performs dispatch.
      @ In, case, HERON Case, Case that this dispatch is part of
      @ In, components, list, HERON components available to the dispatch
      @ In, sources, list, HERON source (placeholders) for signals
      @ In, variables, dict, additional variables passed through
      @ Out, disp, DispatchScenario, resulting dispatch
    """
    time = np.arange(100) # FIXME
    resources = sorted(list(hutils.get_all_resources(components))) # list of all active resources
    # TODO rolling window?
    start_index = 0
    final_index = len(time)
    # TODO make dispatch scenario?
    while start_index < final_index:
      print('DEBUGG starting index')
      end_index = start_index + self._window_len
      if end_index > final_index:
        end_index = final_index
      subdisp = self.dispatch_window(start_index, end_index,
                                     case, components, sources, resources,
                                     variables)
      # TODO update dispatch
      start_index = end_index
      CRASHME

  def dispatch_window(self, start_index, end_index,
                      case, components, sources, resources,
                      variables):
    # build the Pyomo model
    m = pyo.ConcreteModel()
    # indices
    C = np.arange(0, len(components), dtype=int) # indexes component
    R = np.arange(0, len(resources), dtype=int) # TODO indexes resources
    T = np.arange(start_index, end_index, dtype=int) # TODO indexes resources


    m.C = pyo.Set(initialize=C)
    m.R = pyo.Set(initialize=R)
    m.T = pyo.Set(initialize=T)

    # properties
    m.Case = case
    m.Components = components
    m.Resources = resources


    # variables
    ## component productions, as a single matrix indexed by component, resources, time
    ## TODO it saves memory if this gets split up
    ## -> but then the conservation constraint is more complicated, perhaps?
    m.Production = pyo.Var(m.C, m.R, m.T, domain=pyo.Reals)

    # objective
    ## cashflow eval
    ### XXX DEBUGG just max production
    obj = lambda m: - m.Production[0, 1, 2]
    m.obj = pyo.Objective(rule=obj)

    # constraints
    for c, comp in enumerate(components):
      name = comp.name

      # capacity limits
      cap_res = comp.get_capacity_var()
      res = resources.index(cap_res)
      # production is always lower than capacity
      ## get_capacity returns (data, meta) and data is dict
      cap = comp.get_capacity(None, None, None, None)[0][cap_res]
      rule = partial(self.capacity_rule, res, cap) #lambda m, c, t: m.Production[c, res, t] < cap
      constr = pyo.Constraint(m.C, m.T, rule=rule)
      setattr(m, '{c}_{r}_capacity_constr'.format(c=name, r=cap_res), constr)
      # production is always greater than minimum # FIXME using 0 for now
      ## not true! it's negative if it's an input!
      # rule = partial(self.prod_min_rule, res)
      # constr = pyo.Constraint(m.C, m.T, rule=rule)
      # setattr(m, '{c}_{r}_prod_min_constr'.format(c=name, r=cap_res), constr)

      # ramp rates TODO
      ## INCLUDING previous-time boundary condition TODO

      # transfer functions
      ## e.g. 2A + 3B -> 1C + 2E
      ## get linear coefficients
      print('DEBUGG comp:', comp)
      interaction = comp.get_interaction()
      print('DEBUGG   inter:', interaction)
      print('DEBUGG   transfer:', interaction.get_transfer())
      transfer = comp.get_interaction().get_transfer()
      if transfer is not None:
        coeffs = transfer._coefficients
        print('DEBUGG   coeffs:', coeffs)

    ## conservation of resources TODO
    for res, resource in enumerate(resources):
      print('DEBUGG resource conservation:', resource)
      rule = partial(self.conservation_rule, res, 1e-14) #lambda m, c, t: abs(np.sum(m.Production[c, res, t])) <=1e-14 # TODO zero tolerance value?
      constr = pyo.Constraint(m.T, rule=rule)
      setattr(m, '{r}_conservation'.format(r=resource), constr)

    # DEBUGG
    print('/' + '='*80)
    print('DEBUGG model pieces:')
    print('  -> objective:')
    print('     ', m.obj.pprint())
    print('  -> variables:')
    for var in m.component_objects(pyo.Var):
      print('     ', var.pprint())
    print('  -> constraints:')
    for constr in m.component_objects(pyo.Constraint):
      print('     ', constr.pprint())
    print('\\' + '='*80)
    print('')
    # DEBUGG XXX TESTING
    import pyutilib.subprocess.GlobalData
    pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
    soln = pyo.SolverFactory('cbc').solve(m)
    soln.write()
    print('DEBUGG solution:')
    for c, comp in enumerate(components):
      print('  component:', c, comp)
      for r, res in enumerate(resources):
        print('    resource:', r, res)
        for t, time in enumerate(T):
          print('      time:', t, time, m.Production[c, r, t].value)

  def capacity_rule(self, res, cap, m, c, t):
    """ Constructs capacity constraints. TODO"""
    if cap > 0:
      return m.Production[c, res, t] <= cap
    else:
      return m.Production[c, res, t] >= cap

  def prod_min_rule(self, res, m, c, t):
    """ Constructs minumum production constraints. TODO"""
    if cap > 0:
      return m.Production[c, res, t] >= 0
    else:
      return m.Production[c, res, t] <= 0

  def conservation_rule(self, res, tol, m, t):
    """ Constructs conservation constraints. TODO """
    return sum(m.Production[c, res, t] for c in m.C) == 0 #<= tol TODO which is better?

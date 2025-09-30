# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  pyomo-based dispatch strategy
"""
import os
import sys
import time as time_mod
import numpy as np
import pyutilib.subprocess.GlobalData
import logging
from copy import deepcopy

from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints
from ravenframework.utils import InputData, InputTypes

from . import putils
from .Dispatcher import Dispatcher, DispatchError
from .DispatchState import NumpyState

try:
  import dove.core as dv
except ImportError:
  # TODO: temporary solution that works when DOVE and HERON dirs share a parent dir
  sys.path.append(
    os.path.abspath(
      os.path.join(__file__, os.pardir, os.pardir, os.pardir, os.pardir, 'DOVE', 'src')
    )
  )
  import dove.core as dv

# allows pyomo to solve on threaded processes
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

# different solvers express "tolerance" for converging solution in different
# ways. Further, they mean different things for different solvers. This map
# just tracks the "nominal" argument that we should pass through pyomo.
SOLVER_TOL_MAP = {
  'ipopt': 'tol',
  'cbc': 'primalTolerance',
  'glpk': 'mipgap',
}

class DispatchError(Exception):
    """
      Custom exception for dispatch errors.
    """
    pass

class Pyomo(Dispatcher):
  """
    Dispatches using rolling windows in Pyomo
  """
  ### INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory(
      'pyomo', ordered=False, baseNode=None,
      descr=r"""The \texttt{pyomo} dispatcher uses analytic modeling and rolling
      windows to solve dispatch optimization with perfect information via the
      pyomo optimization library."""
    )

    specs.addSub(
      InputData.parameterInputFactory(
        'rolling_window_length', contentType=InputTypes.IntegerType,
        descr=r"""Sets the length of the rolling window that the Pyomo optimization
        algorithm uses to break down histories. Longer window lengths will minimize
        boundary effects, such as nonoptimal storage dispatch, at the cost of slower
        optimization solves. Note that if the rolling window results in a window
        of length 1 (such as at the end of a history), this can cause problems for pyomo.
        \default{24}"""
      )
    )

    specs.addSub(
      InputData.parameterInputFactory(
        'debug_mode', contentType=InputTypes.BoolType,
        descr=r"""Enables additional printing in the pyomo dispatcher.
        Highly discouraged for production runs. \default{False}."""
      )
    )

    specs.addSub(
      InputData.parameterInputFactory(
        'solver', contentType=InputTypes.StringType,
        descr=r"""Indicates which solver should be used by pyomo. Options depend
        on individual installation. \default{'glpk' for Windows, 'cbc' otherwise}."""
      )
    )

    specs.addSub(
      InputData.parameterInputFactory(
        'tol', contentType=InputTypes.FloatType,
        descr=r"""Relative tolerance for converging final optimal dispatch solutions.
        Specific implementation depends on the solver selected. Changing this value
        could have significant impacts on the dispatch optimization time and quality.
        \default{solver dependent, often 1e-6}."""
      )
    )
    # TODO specific for pyomo dispatcher
    return specs


  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.name = 'PyomoDispatcher' # identifying name
    self.debug_mode = False       # whether to print additional information
    self.solve_options = {}       # options passed from Pyomo to the solver
    self._window_len = 24         # time window length to dispatch at a time # FIXME user input
    self._solver = None           # overwrite option for solver
    self._picard_limit = 10       # iterative solve limit


  def read_input(self, specs) -> None:
    """
      Read in input specifications.
      @ In, specs, RAVEN InputData, specifications
      @ Out, None
    """
    super().read_input(specs)

    window_len_node = specs.findFirst('rolling_window_length')
    if window_len_node is not None:
      self._window_len = window_len_node.value

    debug_node = specs.findFirst('debug_mode')
    if debug_node is not None:
      self.debug_mode = debug_node.value

    solver_node = specs.findFirst('solver')
    if solver_node is not None:
      self._solver = solver_node.value

    tol_node = specs.findFirst('tol')
    if tol_node is not None:
      solver_tol = tol_node.value
    else:
      solver_tol = None

    self._solver = putils.check_solver_availability(self._solver)

    if solver_tol is not None:
      key = SOLVER_TOL_MAP.get(self._solver, None)
      if key is not None:
        self.solve_options[key] = solver_tol
      else:
        raise ValueError(f"Tolerance setting not available for solver '{self._solver}'.")


  def get_solver(self):
    """
      Retrieves the solver information (if applicable)
      @ In, None
      @ Out, solver, str, name of solver used
    """
    return self._solver


  def dispatch(self, case, components, sources, meta):
    """
      Performs dispatch.
      @ In, case, HERON Case, Case that this dispatch is part of
      @ In, components, list, HERON components available to the dispatch
      @ In, sources, list, HERON source (placeholders) for signals
      @ In, meta, dict, additional variables passed through
      @ Out, disp, DispatchScenario, resulting dispatch
    """
    t_start, t_end, t_num = self.get_time_discr()
    time = np.linspace(t_start, t_end, t_num)
    dispatch = NumpyState()
    dispatch.initialize(components, meta['HERON']['resource_indexer'], time)

    start_index = 0
    final_index = len(time)

    if case.use_levelized_inner:
      raise NotImplementedError("This version of HERON cannot yet handle cases with levelized cost")

    while start_index < final_index:
      end_index = min(start_index + self._window_len, final_index)
      if end_index - start_index == 1:
        raise DispatchError("Window length of 1 detected, which is not supported.")

      specific_time = time[start_index:end_index]
      print(f"Start: {start_index} End: {end_index}")
      subdisp, solve_time = self._handle_dispatch_window_solve(specific_time, start_index, components, meta)
      print(f'DEBUGG solve time: {solve_time} s')

      # Store Results of optimization into dispatch container
      for comp in components:
        for tag in comp.get_tracking_vars():
          for res, values in subdisp[comp.name][tag].items():
            dispatch.set_activity_vector(comp, res, values, tracker=tag, start_idx=start_index, end_idx=end_index)
      start_index = end_index

    return dispatch


  def _handle_dispatch_window_solve(self, specific_time, start_index, components, meta):
    """
      Set up convergence criteria and collect results from a dispatch window solve.
      @ In, specific_time, np.array, value of time to evaluate.
      @ In, start_index, int, index of the start of the window.
      @ In, components, list, HERON components available to the dispatch.
      @ In, meta, dict, additional variables passed through.
      @ Out, subdisp, dict, results of window dispatch.
    """
    start = time_mod.time()
    subdisp = self._dispatch_window(specific_time, start_index, components, meta)

    if self._needs_convergence(components):  # Should always be False in this version of HERON
      conv_counter = 0
      converged = False
      previous = None

      while not converged and conv_counter < self._picard_limit:
        conv_counter += 1
        print(f'DEBUGG iteratively solving window, iteration {conv_counter}/{self._picard_limit} ...')
        subdisp = self._dispatch_window(specific_time, start_index, components, meta)
        converged = self._check_if_converged(subdisp, previous, components)
        previous = subdisp

      if conv_counter >= self._picard_limit and not converged:
        raise DispatchError(f"Convergence not reached after {self._picard_limit} iterations.")

    end = time_mod.time()
    solve_time = end - start
    return subdisp, solve_time


  def _dispatch_window(self, time, time_offset, components, meta):
    """
      Dispatches one part of a rolling window.
      @ In, time, np.array, value of time to evaluate
      @ In, time_offset, int, offset of the time index in the greater history
      @ In, case, HERON Case, Case that this dispatch is part of
      @ In, components, list, HERON components available to the dispatch
      @ In, sources, list, HERON source (placeholders) for signals
      @ In, resources, list, sorted list of all resources in problem
      @ In, initial_storage, dict, initial storage levels if any
      @ In, meta, dict, additional variables passed through
      @ Out, result, dict, results of window dispatch
    """
    model = self._build_dove_dispatch(time, time_offset, components, meta)
    result = self._solve_dispatch(model, components)
    return result


  def _build_dove_dispatch(self, time, time_offset, components, meta):
    """
      Build dispatch model using DOVE.
      @ In, time, np.array, value of time to evaluate
      @ In, time_offset, int, offset of the time index in the greater history
      @ In, components, list, HERON components available to the dispatch
      @ In, meta, dict, additional variables passed through
      @ Out, model, pyo.ConcreteModel, the pyomo model built by DOVE
    """
    context = deepcopy(meta)

    heron_res_map = context['HERON']['resource_indexer']
    dove_res_map = self._create_dove_resources(heron_res_map)

    dove_comp_list = []
    for comp in components:
      caps = []  # Time dependent capacity values
      mins = []  # Time dependent minimum values
      cf_map = {} # dict keyed by cashflow names with values that are dicts containing economic info

      for t in range(len(time)):
        # update time index in context for capacity/minimum evaluation
        context['HERON']['time_index'] = t + time_offset
        caps.append(comp.get_capacity(context)[0][comp.get_capacity_var()]) # get capacity for this component
        mins.append(comp.get_minimum(context)[0][comp.get_capacity_var()]) # get minimum for this component

        # We have to spoof activity to get the cashflow params
        recurring_cfs = [cf for cf in comp.get_cashflows() if cf.get_type() == 'repeating' and cf.get_period() != "year"]
        for cf in recurring_cfs:
          if cf.name not in cf_map.keys():
            cf_map[cf.name] = {"alphas": [], "dprimes": [], "scaling_factors": [], "d_multipliers": [], "costs": []}
            # Costs only used for identifying the sign of the cashflow
          # Set capacity to 1 in order to read the multiplier on the driver
          context["HERON"]["activity"] = {cf.get_driver()._vp._tracking_var: {comp.get_capacity_var(): 1}}
          params = cf.calculate_params(context)

          cf_map[cf.name]["alphas"].append(params["alpha"])
          cf_map[cf.name]["dprimes"].append(params["ref_driver"])
          cf_map[cf.name]["scaling_factors"].append(params["scaling"])
          cf_map[cf.name]["d_multipliers"].append(params["driver"])
          cf_map[cf.name]["costs"].append(params["cost"])

      comp._capacity_vector_t = caps
      comp._minimum_vector_t = mins
      comp._cfs = cf_map
      comp._r = comp.get_capacity_var()

      if comp.get_interaction().is_type("Storage"):
        comp._initial_storage = comp.get_interaction().get_initial_level(context)
        comp._max_charge, comp._max_discharge = comp.get_interaction().get_charge_rate_limits(context)
        comp._periodic_level = comp.get_interaction().apply_periodic_level

      if comp.is_governed():
        # comp._activity = comp.get_interaction().get_strategy().evaluate(self.meta)[0]['level']
        raise NotImplementedError(f"{comp.name}: This version of HERON cannot yet handle governed components.")

      dove_comp = self._create_dove_component(comp, dove_res_map)
      dove_comp_list.append(dove_comp)

    dove_system = dv.System(
      components=dove_comp_list,
      resources=list(dove_res_map.values()),
      dispatch_window=np.arange(0, len(time), dtype=int)
    )

    model = dove_system.build(model_type="price_taker")
    return model


  def _create_dove_resources(self, heron_res_map):
    dove_res_map = {}
    for indexer_per_comp in heron_res_map.values():
      for r_name in indexer_per_comp.keys():
        if r_name not in dove_res_map.keys():
          dove_res_map[r_name] = dv.Resource(name=r_name)
    return dove_res_map


  def _create_dove_component(self, heron_comp, dove_res_map):
    interaction = heron_comp.get_interaction()
    if interaction.is_type("Producer"):
      if interaction.get_transfer() is None:
        dove_comp = self._create_dove_source(heron_comp, dove_res_map)
      else:
        dove_comp = self._create_dove_converter(heron_comp, dove_res_map)
    elif interaction.is_type("Storage"):
      dove_comp = self._create_dove_storage(heron_comp, dove_res_map)
    elif interaction.is_type("Demand"):
      dove_comp = self._create_dove_sink(heron_comp, dove_res_map)
    else:
      raise IOError(f"{heron_comp.name}: Unrecognized interaction type. Please use 'Producer', 'Storage', or 'Demand'.")

    return dove_comp


  def _create_dove_source(self, heron_comp, dove_res_map):
    init_kwargs = {}
    init_kwargs["name"] = heron_comp.name
    init_kwargs["installed_capacity"] = max(heron_comp._capacity_vector_t)
    if not all(cap_val == init_kwargs["installed_capacity"] for cap_val in heron_comp._capacity_vector_t):
      init_kwargs["capacity_factor"] = [
        cap_val / init_kwargs["installed_capacity"] for cap_val in heron_comp._capacity_vector_t
      ]
    if heron_comp.is_dispatchable() == "fixed":
      init_kwargs["flexibility"] = "fixed"
    elif init_kwargs["installed_capacity"] > 0:
      # The minimum would be overridden by fixed flexibility, so only worry about it if it's flexible
      init_kwargs["min_capacity_factor"] = [
        min_val / init_kwargs["installed_capacity"] for min_val in heron_comp._minimum_vector_t
      ]

    if heron_comp.ramp_limit is not None:
      init_kwargs["ramp_limit"] = heron_comp.ramp_limit()
    if heron_comp.ramp_freq is not None:
      init_kwargs["ramp_freq"] = heron_comp.ramp_freq()

    if heron_comp._cfs:
      init_kwargs["cashflows"] = []
      for cf_name, cf_data in heron_comp._cfs.items():
        init_kwargs["cashflows"].append(self._create_dove_cashflow(cf_name, cf_data, 1))
    init_kwargs["produces"] = dove_res_map[heron_comp._r]

    return dv.Source(**init_kwargs)


  def _create_dove_sink(self, heron_comp, dove_res_map):
    init_kwargs = {}
    init_kwargs["name"] = heron_comp.name
    init_kwargs["demand_profile"] = [abs(cap_at_t) for cap_at_t in heron_comp._capacity_vector_t]
    if heron_comp.is_dispatchable() == "fixed":
      init_kwargs["flexibility"] = "fixed"
    else:
      # The minimum would be overridden by fixed flexibility, so only worry about it if it's flexible
      init_kwargs["min_demand_profile"] = [abs(min_val) for min_val in heron_comp._minimum_vector_t]
    if heron_comp._cfs:
      init_kwargs["cashflows"] = []
      for cf_name, cf_data in heron_comp._cfs.items():
        init_kwargs["cashflows"].append(self._create_dove_cashflow(cf_name, cf_data, -1))
    init_kwargs["consumes"] = dove_res_map[heron_comp._r]

    return dv.Sink(**init_kwargs)


  def _create_dove_converter(self, heron_comp, dove_res_map):
    init_kwargs = {}
    init_kwargs["name"] = heron_comp.name

    init_kwargs["installed_capacity"] = max(abs(cap_val) for cap_val in heron_comp._capacity_vector_t)
    if not all(cap_val == heron_comp._capacity_vector_t[0] for cap_val in heron_comp._capacity_vector_t):
      init_kwargs["capacity_factor"] = [
        abs(cap_val) / init_kwargs["installed_capacity"] for cap_val in heron_comp._capacity_vector_t
      ]
    if heron_comp.is_dispatchable() == "fixed":
      init_kwargs["flexibility"] = "fixed"
    elif init_kwargs["installed_capacity"] > 0:
      # The minimum would be overridden by fixed flexibility, so only worry about it if it's flexible
      init_kwargs["min_capacity_factor"] = [
        abs(min_val) / init_kwargs["installed_capacity"] for min_val in heron_comp._minimum_vector_t
      ]

    init_kwargs["consumes"] = [dove_res_map[res_name] for res_name in list(heron_comp.get_inputs())]
    init_kwargs["produces"] = [dove_res_map[res_name] for res_name in list(heron_comp.get_outputs())]
    init_kwargs["capacity_resource"] = dove_res_map[heron_comp._r]

    heron_transfer_fn = heron_comp.get_interaction().get_transfer()
    coeffs = heron_comp.get_interaction().get_transfer().get_coefficients()
    match heron_transfer_fn.type:
      case "Ratio":
        tf_inputs = {dove_res: abs(coeffs[dove_res.name]) for dove_res in init_kwargs["consumes"]}
        tf_outputs = {dove_res: abs(coeffs[dove_res.name]) for dove_res in init_kwargs["produces"]}
        init_kwargs["transfer_fn"] = dv.RatioTransfer(input_resources=tf_inputs, output_resources=tf_outputs)
      case "Polynomial":
        # TODO
        # HERON has everything on the lhs of the equation (0 on the right)
        # DOVE is incompatible as is (it has the output on the right)
        raise NotImplementedError(f"{heron_comp.name}: This version of HERON does not support Polynomial transfer functions.")
      case _:
        raise IOError(f"{heron_comp.name}: Unrecognized transfer function type.")

    if heron_comp.ramp_limit is not None:
      init_kwargs["ramp_limit"] = heron_comp.ramp_limit()
    if heron_comp.ramp_freq is not None:
      init_kwargs["ramp_freq"] = heron_comp.ramp_freq()

    if heron_comp._cfs:
      init_kwargs["cashflows"] = []
      for cf_name, cf_data in heron_comp._cfs.items():
        cf_default_sign = 1 if init_kwargs["capacity_resource"] in init_kwargs["produces"] else -1
        init_kwargs["cashflows"].append(self._create_dove_cashflow(cf_name, cf_data, cf_default_sign))

    return dv.Converter(**init_kwargs)


  def _create_dove_storage(self, heron_comp, dove_res_map):
    init_kwargs = {}
    init_kwargs["name"] = heron_comp.name

    init_kwargs["installed_capacity"] = max(abs(cap_val) for cap_val in heron_comp._capacity_vector_t)
    if init_kwargs["installed_capacity"] > 0:
      if not all(cap_val == heron_comp._capacity_vector_t[0] for cap_val in heron_comp._capacity_vector_t):
        init_kwargs["capacity_factor"] = [
          abs(cap_val) / init_kwargs["installed_capacity"] for cap_val in heron_comp._capacity_vector_t
        ]
      init_kwargs["min_capacity_factor"] = [
          abs(min_val) / init_kwargs["installed_capacity"] for min_val in heron_comp._minimum_vector_t
        ]
    if heron_comp.is_dispatchable() == "fixed":
      init_kwargs["flexibility"] = "fixed"
    if heron_comp._cfs:
      init_kwargs["cashflows"] = []
      for cf_name, cf_data in heron_comp._cfs.items():
        init_kwargs["cashflows"].append(self._create_dove_cashflow(cf_name, cf_data, 1))

    init_kwargs["resource"] = dove_res_map[heron_comp._r]
    init_kwargs["rte"] = (heron_comp.get_interaction().get_sqrt_RTE())**2
    if heron_comp._max_charge is not None:
      init_kwargs["max_charge_rate"] = heron_comp._max_charge
    if heron_comp._max_discharge is not None:
      init_kwargs["max_discharge_rate"] = heron_comp._max_discharge
    if heron_comp._periodic_level:
      init_kwargs["periodic_level"] = True
    else:
      init_kwargs["initial_stored"] = heron_comp._initial_storage / init_kwargs["installed_capacity"]
      init_kwargs["periodic_level"] = False

    return dv.Storage(**init_kwargs)


  def _create_dove_cashflow(self, cf_name, cf_data, default_sign):
    '''
    default_sign: 1 if activity causes revenue from this cashflow; -1 if activity causes expense
    '''
    # HERON:  cost  =                      a(t) * [D(activity) / D'(t)]^x
    #  DOVE: |cost| = | price_profile(t) * a    * [ activity   / D'   ]^x |
    # We need to rearrange the HERON equation to:
    # (1) find the absolute value of the cashflow (we'll handle the sign later)
    # (2) move all t-dependencies into the price_profile term
    # (3) convert D(activity) to activity
    cf_kwargs = {"name": cf_name}
    price_profile = []

    # Handle alpha
    if all(alpha_val == cf_data["alphas"][0] for alpha_val in cf_data["alphas"]):
      # HERON a(t) is constant with time
      cf_kwargs["alpha"] = abs(cf_data["alphas"][0])
    else:
      # HERON a(t) varies with time
      cf_kwargs["alpha"] = 1
      price_profile = [abs(alpha_at_t) for alpha_at_t in cf_data["alphas"]]

    # Handle scalex
    if all(sf_val == cf_data["scaling_factors"][0] for sf_val in cf_data["scaling_factors"]):
      # HERON scaling factor is constant with time
      cf_kwargs["scalex"] = cf_data["scaling_factors"][0]
    else:
      # HERON scaling factor varies with time
      raise ValueError(f"{cf_name}: scaling factor must be constant with respect to time.")

    # Handle dprime
    if all(dprime_val == cf_data["dprimes"][0] for dprime_val in cf_data["dprimes"]):
      # HERON D'(t) is constant with time
      cf_kwargs["dprime"] = abs(cf_data["dprimes"][0])
    else:
      # HERON D'(t) varies with time
      # cost = a * [D/D']^x => cost = D'^[-x] * [a * D^x]
      cf_kwargs["dprime"] = 1.0
      if price_profile:
        # The new price_profile has a max length such that there's available data for itself and D'
        allowed_len = min(len(price_profile), len(cf_data["dprimes"]))
        price_profile = [
          price_profile[t] * abs(cf_data["dprimes"][t])**(-1 * cf_kwargs["scalex"]) for t in range(allowed_len)
        ]
      else:
        price_profile = [
          abs(cf_data["dprimes"][t])**(-1 * cf_kwargs["scalex"]) for t in range(len(cf_data["d_primes"]))
        ]

    # Handle driver
    # TODO: Any chance the Driver isn't activity-based? Would need to consider differently.
    # Need to consider cases where HERON D(activity) = multiplier * activity
    # cost = a * [D/D']^x; D = d_mult*activity => cost = d_mult^x * [a * [activity/D']^x]
    if price_profile:
      # The new price_profile has a max length such that there's available data for itself and d_mult
      allowed_len = min(len(price_profile), len(cf_data["d_multipliers"]))
      price_profile = [
        price_profile[t] * abs(cf_data["d_multipliers"][t])**cf_kwargs["scalex"] for t in range(allowed_len)
      ]
    else:
      price_profile = [
        abs(cf_data["d_multipliers"][t])**cf_kwargs["scalex"] for t in range(len(cf_data["d_multipliers"]))
      ]

    # Set price_profile
    if price_profile:
      cf_kwargs["price_profile"] = price_profile

    if all(cost*default_sign >= 0 for cost in cf_data["costs"]):
      return dv.Revenue(**cf_kwargs)
    elif all(cost*default_sign <= 0 for cost in cf_data["costs"]):
      return dv.Cost(**cf_kwargs)
    else:
      raise ValueError(f"{cf_name}: Sign of cashflow must be either always positive or always negative.")


  def _check_if_converged(self, new, old, components, tol=1e-4):
    """
      Checks convergence of consecutive dispatch solves
      @ In, new, dict, results of dispatch # TODO should this be the model rather than dict?
      @ In, old, dict, results of previous dispatch
      @ In, components, list, HERON component list
      @ In, tol, float, optional, tolerance for convergence
      @ Out, converged, bool, True if convergence is met
    """
    if old is None:
      return False

    for comp in components:
      intr = comp.get_interaction()
      if intr.is_governed(): # by "is_governed" we mean "isn't optimized in pyomo"
        # check activity L2 norm as a differ
        # TODO this may be specific to storage right now
        name = comp.name
        tracker = comp.get_tracking_vars()[0]
        res = intr.get_resource()
        scale = np.max(old[name][tracker][res])
        # Avoid division by zero
        if scale == 0:
          diff = np.linalg.norm(new[name][tracker][res] - old[name][tracker][res])
        else:
          diff = np.linalg.norm(new[name][tracker][res] - old[name][tracker][res]) / scale
        if diff > tol:
          return False
    return True


  def _needs_convergence(self, components):
    """
      Determines whether the current setup needs convergence to solve.
      @ In, components, list, HERON component list
      @ Out, needs_convergence, bool, True if iteration is needed
    """
    # NOTE: Since we can't handle governed components in this version of HERON, should always return False
    return any(comp.get_interaction().is_governed() for comp in components)


  def _retrieve_dove_solution(self, components, results_df):
    """
    Extracts results from the pandas dataframe into a dict.
    @ In, results_df, pd.Dataframe, the results of the dispatch
    @ Out, results_heron_dict, dict, dict of numpy arrays for dispatch results
    """
    results_dove_dict = results_df.to_dict(orient="list")
    results_heron_dict = {comp.name: {} for comp in components}
    for comp in components:
      for tracking_var in comp.get_tracking_vars():
        if tracking_var == "production":
          expected_substr = comp.name + "_" + comp.get_capacity_var()
          for col_name, values in results_dove_dict.items():
            if expected_substr in col_name:
              results_heron_dict[comp.name].update(
                {"production": {comp.get_capacity_var(): np.array(values)}}
              )
              break
        if tracking_var == "level":
          expected_col_name = comp.name + "_SOC"
          results_heron_dict[comp.name].update(
            {"level": {comp.get_capacity_var(): np.array(results_dove_dict[expected_col_name])}}
          )
        if tracking_var == "charge":
          expected_col_name = comp.name + "_charge"
          results_heron_dict[comp.name].update(
            {"charge": {comp.get_capacity_var(): np.array(results_dove_dict[expected_col_name])}}
          )
        if tracking_var == "discharge":
          expected_col_name = comp.name + "_discharge"
          values = np.array([-val for val in results_dove_dict[expected_col_name]])
          results_heron_dict[comp.name].update(
            {"discharge": {comp.get_capacity_var(): values}}
          )

    return results_heron_dict


  def _solve_dispatch(self, builder, components):
    """
      Solves the dispatch problem.
      @ In, builder, PriceTakerBuilder, dove builder object
      @ In, meta, dict, additional variables passed through
      @ Out, result, dict, results of solved dispatch
    """
    # start a solution search
    done_and_checked = False
    attempts = 0
    # DEBUGG show variables, bounds
    # if self.debug_mode:
    #   putils.debug_pyomo_print(builder.model)

    while not done_and_checked:
      attempts += 1
      print(f'DEBUGG using solver: {self._solver}')
      print(f'DEBUGG solve attempt {attempts} ...:')
      soln = builder.solve(solver=self._solver, options=self.solve_options)

      # check solve status
      if soln.solver.status == SolverStatus.ok and soln.solver.termination_condition == TerminationCondition.optimal:
        print('DEBUGG ... solve was successful!')
      else:
        putils.debug_pyomo_print(builder.model)
        log_infeasible_constraints(builder.model, log_expression=True, log_variables=True)
        log_name = 'constraint_violations.log'
        logging.basicConfig(filename=log_name, encoding='utf-8', level=logging.INFO)
        raise DispatchError(
          f'Solve was unsuccessful, see log file located at: "{os.getcwd()}/{log_name}" for more details! Status: {soln.solver.status} Termination: {soln.solver.termination_condition}'
        )

      # try validating
      # print('DEBUGG ... validating ...')
      # validation_errs = self.validate(builder.model.Components, builder.model.Activity, builder.model.Times, meta)
      # if validation_errs:
      #   done_and_checked = False
      #   print('DEBUGG ... validation concerns raised:')
      #   for e in validation_errs:
      #     print(f"DEBUGG ... ... Time {e['time_index']} ({e['time']}) \n" +
      #           f"Component \"{e['component'].name}\" Resource \"{e['resource']}\": {e['msg']}")
      #     # builder._create_production_limit(e)
      #   # TODO go back and solve again
      #   # raise DispatchError('Validation failed, but idk how to handle that yet')
      # else:
      #   print('DEBUGG Solve successful and no validation concerns raised.')
      #   done_and_checked = True
      done_and_checked = True

      # In the words of Charles Bukowski, "Don't Try [too many times]"
      if attempts > 100:
        raise DispatchError('Exceeded validation attempt limit!')

    # if self.debug_mode:
    #   # soln.write()
    #   putils.debug_print_soln(builder.model)

    # return dict of numpy arrays
    dove_results_df = builder.extract_results()
    result = self._retrieve_dove_solution(components, dove_results_df)

    return result

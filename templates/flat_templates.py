# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Templates for workflows which can be "flat" RAVEN workflows (no need for RAVEN-runs-RAVEN)

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-12-23
"""
import os

from .heron_types import HeronCase, Component, Source

from .raven_template import RavenTemplate

from .snippets.dataobjects import PointSet
from .snippets.runinfo import RunInfo
from .snippets.samplers import CustomSampler, EnsembleForward, Grid
from .snippets.steps import MultiRun
from .snippets.variablegroups import VariableGroup

from .naming_utils import get_capacity_vars


class FlatMultiConfigTemplate(RavenTemplate):
  """
  A template for RAVEN workflows which do not consider uncertainty from sources which affect the system dispatch (one
  time history, no uncertain variable costs for dispatchable components). Many system configurations may be considered.
  """
  template_name = "flat_multi_config.xml"
  write_name = "outer.xml"

  # With static histories, the stats that are used shouldn't require multiple samples. Therefore, we drop the
  # sigma and variance default stat names for the static history case here.
  DEFAULT_STATS_NAMES = {
    "opt": ["expectedValue", "median"],
    "sweep": ["maximum", "minimum", "percentile", "samples"]
  }

  def createWorkflow(self, **kwargs) -> None:
    """
    Create a workflow for the specified Case and its components and sources
    @ In, kwargs, dict, keyword arguments
    @ Out, None
    """
    super().createWorkflow(**kwargs)
    case = kwargs["case"]
    components = kwargs["components"]
    sources = kwargs["sources"]

    case_name = self.namingTemplates["jobname"].format(case=case.name, io="o")
    self._set_case_name(case_name)
    self._initialize_runinfo(case)

    # Set up some helpful variable groups
    capacities_vargroup = self._template.find("VariableGroups/Group[@name='GRO_capacities']")  # type: VariableGroup
    capacities_vars = list(get_capacity_vars(components, self.namingTemplates["variable"]))
    capacities_vargroup.variables.extend(capacities_vars)

    results_vargroup = self._template.find("VariableGroups/Group[@name='GRO_results']")  # type: VariableGroup
    results_vars = self._get_deterministic_results_vars(case, components)
    results_vargroup.variables.extend(results_vars)

    # Define a sampler for handling the static history
    static_hist_sampler = self._template.find("Samplers/EnsembleForward/CustomSampler")  # type: CustomSampler
    self._configure_static_history_sampler(static_hist_sampler, case, sources, scaling=None)

    # Define a grid sampler, a data object to store the sweep results, and an outstream to print those results
    grid_sampler = self._template.find("Samplers/EnsembleForward/Grid")  # type: Grid
    grid_results = self._template.find("DataObjects/PointSet[@name='grid']")  # type: PointSet

    variables, consts = self._create_sampler_variables(case, components)
    for sampled_var, vals in variables.items():
      grid_sampler.add_variable(sampled_var)
      sampled_var.use_grid(construction="custom", kind="value", values=sorted(vals))

    ensemble_sampler = self._template.find("Samplers/EnsembleForward")  # type: EnsembleForward
    for var_name, val in consts.items():
      ensemble_sampler.add_constant(var_name, val)

    # If there are any case labels, make a variable group for those and add it to the "grid" PointSet.
    # These labels also need to get added to the sampler as constants.
    labels = case.get_labels()
    if labels:
      vargroup = self._create_case_labels_vargroup(labels)
      self._add_snippet(vargroup)
      grid_results.outputs.append(vargroup.name)
    self._add_labels_to_sampler(grid_sampler, labels)

    # Use a MultiRun to run to the model over the grid points
    multirun = self._template.find("Steps/MultiRun[@name='sweep']")  # type: MultiRun
    for func in self._get_function_files(sources):
      multirun.add_input(func)

    # Update the parallel settings based on the number of sampled variables if the number of outer parallel runs
    # was not specified before.
    if case.outerParallel == 0 and case.useParallel:
      sampler = self._template.find("Samplers/Grid")
      run_info = self._template.find("RunInfo")
      case.outerParallel = sampler.num_sampled_vars + 1
      run_info.batch_size = case.outerParallel
      run_info.internal_parallel = True

  def _initialize_runinfo(self, case: HeronCase) -> None:
    """
    Initializes the RunInfo node of the workflow
    @ In, case, Case, the HERON Case object
    @ Out, None
    """
    run_info = self._template.find("RunInfo")  # type: RunInfo

    # parallel
    batch_size = min(case.outerParallel, 1) * min(case.innerParallel, 1)
    run_info.use_internal_parallel = batch_size > 1

    if case.useParallel:
      # Fills in parallel settings for template RunInfo from case. Also appliespre-sets for known
      # hostnames (e.g. sawtooth, bitterroot), as specified in the HERON/templates/parallel/*.xml files.
      run_info.set_parallel_run_settings(case.parallelRunInfo)

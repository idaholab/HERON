# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  A template for HERON's debug mode

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-12-23
"""
from .raven_template import RavenTemplate

from .snippets.models import EnsembleModel
from .snippets.outstreams import HeronDispatchPlot, TealCashFlowPlot
from .snippets.runinfo import RunInfo
from .snippets.samplers import MonteCarlo, CustomSampler
from .snippets.variablegroups import VariableGroup

from .heron_types import HeronCase, Component, Source
from .naming_utils import get_capacity_vars, get_component_activity_vars, get_cashflow_names
from .xml_utils import find_node


class DebugTemplate(RavenTemplate):
  """ Sets up a flat RAVEN run for debug mode """
  template_name = "debug.xml"
  write_name = "outer.xml"

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

    # RunInfo initialization
    case_name = self.namingTemplates["jobname"].format(case=case.name, io="o")
    self._set_case_name(case_name)
    self._initialize_runinfo(case)

    # First, handle aspects of the workflow common across all debug runs
    self._update_vargroups(case, components, sources)
    self._update_dataset_indices(case)

    # Add optional plots
    debug_iostep = self._template.find("Steps/IOStep[@name='debug_output']")
    if case.debug["dispatch_plot"]:
      disp_plot = self._make_dispatch_plot(case)
      self._add_snippet(disp_plot)
      debug_iostep.add_output(disp_plot)

    if case.debug["cashflow_plot"]:
      cashflow_plot = self._make_cashflow_plot()
      self._add_snippet(cashflow_plot)
      debug_iostep.add_output(cashflow_plot)

    # Then, figure out how things will be sampled for the debug run.
    # We need to handle 3 separate sampler configurations:
    #   1. MonteCarlo sampler: Used for sampling from time series ROM and distributions for swept/optimized
    #      capacities without a debug value and uncertain cashflow parameters.
    #   2. CustomSampler sampler: Used for getting a static history from a CSV file.
    #   3. EnsembleForward sampler: Used for combining the two where needed.
    # If there is a MonteCarlo sampler being used, we'll add any constants (like capacities with debug values)
    # there. Otherwise, if only a CustomSampler is used, we'll add them to the CustomSampler.

    # What time series sources does our case have?
    has_arma_source = any(s.is_type("ARMA") for s in sources)
    has_csv_source = any(s.is_type("CSV") for s in sources)

    # What variables need to be sampled and which are constants?
    cap_vars, cap_consts = self._create_sampler_variables(case, components)  # capacities
    has_sampled_capacities = len(cap_vars) > 0

    # Are there any uncertain cashflow parameters which need to get sampled?
    cashflow_vars, cashflow_dists = self._get_uncertain_cashflow_params(components)
    has_uncertain_cashflows = len(cashflow_vars) > 0

    # Create the Monte Carlo sampler, if needed
    if any([has_arma_source, has_sampled_capacities, has_uncertain_cashflows]):
      # Okay, we know we need it, so let's make it.
      monte_carlo = MonteCarlo("mc")

      # Set number of samples for sampler
      monte_carlo.denoises = case.get_num_samples()
      monte_carlo.init_limit = case.get_num_samples()

      # Set up case to use synthetic history ROM
      if has_arma_source:
        self._use_time_series_rom(monte_carlo, case, sources)

      # Add capacities to sampler
      for sampled_var in cap_vars:
        monte_carlo.add_variable(sampled_var)
      for var_name, val in cap_consts.items():
        monte_carlo.add_constant(var_name, val)

      # Add uncertain cashflow parameters
      if has_uncertain_cashflows:
        vg_econ_uq = find_node(self._template, "VariableGroups/Group[@name='GRO_UQ']")  # type: VariableGroup
        self._template.find("VariableGroups/Group[@name='GRO_dispatch_in_scalar']").variables.append(vg_econ_uq.name)
        self._template.find("VariableGroups/Group[@name='GRO_timeseries_in_scalar']").variables.append(vg_econ_uq.name)
        # Add the SampledVariable and Distribution nodes to the appropriate locations
        for samp_var, dist in zip(cashflow_vars, cashflow_dists):
          self._add_snippet(dist)
          vg_econ_uq.variables.append(samp_var.name)
          monte_carlo.add_variable(samp_var)
    else:
      monte_carlo = None

    # The CustomSampler is only needed if there is a static history from CSV
    if has_csv_source:
      custom_sampler = CustomSampler("static_hist_sampler")
      self._configure_static_history_sampler(custom_sampler, case, sources)
    else:
      custom_sampler = None

    # If only a CustomSampler if being used, the capacity constants need to be added to the custom sampler
    if monte_carlo is None and custom_sampler is not None:
      for var_name, val in cap_consts.items():
        custom_sampler.add_constant(var_name, val)

    # If we need both the MonteCarlo sampler and the CustomSampler, add them both to an EnsembleForward sampler so they
    # can be used together.
    multirun_step = self._template.find("Steps/MultiRun[@name='debug']")
    if monte_carlo and custom_sampler:
      ensemble_sampler = self._create_ensemble_forward_sampler([monte_carlo, custom_sampler], name="ensemble_sampler")
      self._add_snippet(ensemble_sampler)
      multirun_step.add_sampler(ensemble_sampler)
    elif monte_carlo is not None:
      self._add_snippet(monte_carlo)
      multirun_step.add_sampler(monte_carlo)
    elif custom_sampler is not None:
      self._add_snippet(custom_sampler)
      multirun_step.add_sampler(custom_sampler)
    else:
      raise ValueError("Nothing that requires a sampler was found.")

    # Add the model and file inputs to the main multirun step
    multirun = self._template.find("Steps/MultiRun[@name='debug']")
    model = self._template.find("Models/EnsembleModel") or self._template.find("Models/ExternalModel")
    multirun.add_model(model)
    for func in self._get_function_files(sources):
      multirun.add_input(func)

  def _initialize_runinfo(self, case: HeronCase) -> None:
    """
    Initializes the RunInfo node of the workflow
    @ In, case_name, str, optional, the case name
    @ Out, None
    """
    run_info = self._template.find("RunInfo")  # type: RunInfo

    # Use the outer parallel settings for flat run modes
    batch_size = min(case.outerParallel, 1) * min(case.innerParallel, 1)
    run_info.use_internal_parallel = batch_size > 1

    if case.useParallel:
      # Fills in parallel settings for template RunInfo from case. Also applies pre-sets for known
      # hostnames (e.g. sawtooth, bitterroot), as specified in the HERON/templates/parallel/*.xml files.
      run_info.set_parallel_run_settings(case.parallelRunInfo)

  def _use_time_series_rom(self, sampler: MonteCarlo, case: HeronCase, sources: list[Source]) -> None:
    """
    Sets the workflow up to sample a time history from a PickledROM model
    @ In sampler, MonteCarlo, a MonteCarlo sampler snippet
    @ In, case, HeronCase, the HERON case
    @ In, sources, list[Source], external models, data, and functions
    @ Out, None
    """
    # If a time series PickledROM is being used, it will need to be combined with the HERON.DispatchManager external
    # model with an EnsembleModel.
    ensemble = EnsembleModel("sample_and_dispatch")
    self._add_snippet(ensemble)

    # Load the time series ROM(s) from file and add to the ensemble model.
    # Includes steps to load and print metadata for the pickled time series ROMs
    self._add_time_series_roms(ensemble, case, sources)

    # Fetch the dispatch model and add it to the ensemble. The model and associated data objects already exist
    # in the template XML, so we find those and add them to the model.
    dispatcher = self._template.find("Models/ExternalModel[@subType='HERON.DispatchManager']")
    dispatcher_assemb = dispatcher.to_assembler_node("Model")
    # FIXME: I don't know why this is the case with RAVEN, but the dispatch_placeholder data object MUST come before
    # any function <Input> nodes, or it errors out. This is bad XML practice, which should be independent of order!
    disp_placeholder = self._template.find("DataObjects/PointSet[@name='dispatch_placeholder']")
    dispatcher_assemb.append(disp_placeholder.to_assembler_node("Input"))  # THIS COMES FIRST
    for func in self._get_function_files(sources):
      dispatcher_assemb.append(func.to_assembler_node("Input"))  # THEN ADD THESE
    disp_eval = self._template.find("DataObjects/DataSet[@name='dispatch_eval']")
    dispatcher_assemb.append(disp_eval.to_assembler_node("TargetEvaluation"))
    ensemble.append(dispatcher_assemb)

    # A scaling constant needs to be added to the MonteCarlo sampler for the
    sampler.add_constant("scaling", 1.0)

  def _update_vargroups(self, case: HeronCase, components: list[Component], sources: list[Source]) -> None:
    """
    Updates existing variable group nodes with index and variable names
    @ In, case, HeronCase, the HERON case object
    @ In, components, list[Component], the case components
    @ In, sources, list[Source], the case data sources
    @ Out, None
    """
    # Fill out capacities vargroup
    capacities_vargroup = self._template.find("VariableGroups/Group[@name='GRO_capacities']")
    capacities_vars = list(get_capacity_vars(components, self.namingTemplates["variable"], debug=True))
    capacities_vargroup.variables.extend(capacities_vars)

    # Add time indices to GRO_time_indices
    self._template.find("VariableGroups/Group[@name='GRO_time_indices']").variables = [
      case.get_time_name(),
      case.get_year_name()
    ]

    # Dispatch variables
    dispatch_vars = get_component_activity_vars(components, self.namingTemplates["dispatch"])
    self._template.find("VariableGroups/Group[@name='GRO_full_dispatch']").variables.extend(dispatch_vars)

    # Cashflows
    cfs = get_cashflow_names(components)
    self._template.find("VariableGroups/Group[@name='GRO_cashflows']").variables.extend(cfs)

    # Time history sources
    group = self._template.find("VariableGroups/Group[@name='GRO_debug_synthetics']")  # type: VariableGroup
    for source in filter(lambda x: x.type in ["ARMA", "CSV"], sources):
      synths = source.get_variable()
      group.variables.extend(synths)

    # Figure out which econ metrics are being used for the case
    activity_vars = get_component_activity_vars(components, self.namingTemplates["tot_activity"])
    econ_vars = case.get_econ_metrics(nametype="output")
    output_vars = econ_vars + activity_vars
    self._template.find("VariableGroups/Group[@name='GRO_dispatch_out']").variables.extend(output_vars)
    self._template.find("VariableGroups/Group[@name='GRO_timeseries_out_scalar']").variables.extend(output_vars)

  def _update_dataset_indices(self, case: HeronCase) -> None:
    """
    Update the Index node variables for all DataSet nodes to correctly reflect the provided macro, micro, and cluster
    index names
    @ In, case, HeronCase, the HERON case
    @ Out, None
    """
    # Configure dispatch DataSet indices
    time_name = case.get_time_name()
    year_name = case.get_year_name()
    cluster_name = self.namingTemplates["cluster_index"]

    for time_index in self._template.findall(".//DataSet/Index[@var='Time']"):
      time_index.set("var", time_name)

    for year_index in self._template.findall(".//DataSet/Index[@var='Year']"):
      year_index.set("var", year_name)

    for cluster_index in self._template.findall(".//DataSet/Index[@var='_ROM_Cluster']"):
      cluster_index.set("var", cluster_name)

  def _make_dispatch_plot(self, case: HeronCase) -> HeronDispatchPlot:
    """
    Make a HERON dispatch plot
    @ In, case, HeronCase, the HERON case
    @ Out, disp_plot, HeronDispatchPlot, the dispatch plot node
    """
    disp_plot = HeronDispatchPlot("dispatchPlot")
    dispatch_dataset = self._template.find("DataObjects/DataSet[@name='dispatch']")
    disp_plot.source = dispatch_dataset
    disp_plot.macro_variable = case.get_year_name()
    disp_plot.micro_variable = case.get_time_name()
    disp_plot.signals.append("GRO_debug_synthetics")
    return disp_plot

  def _make_cashflow_plot(self) -> TealCashFlowPlot:
    """
    Make a TEAL cashflow plot
    @ In, None,
    @ Out, cashflow_plot, TealCashFlowPlot, the cashflow plot node
    """
    cashflow_plot = TealCashFlowPlot("cashflow_plot")
    cashflows = self._template.find("DataObjects/HistorySet[@name='cashflows']")
    cashflow_plot.source = cashflows
    return cashflow_plot

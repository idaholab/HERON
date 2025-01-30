# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Templates for bilevel RAVEN workflows (i.e. RAVEN-runs-RAVEN workflows)

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-12-23
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

from .imports import RAVEN_LOC
from .heron_types import HeronCase, Component, Source
from .naming_utils import get_capacity_vars, get_component_activity_vars, get_opt_objective

from .raven_template import RavenTemplate
from .snippets.runinfo import RunInfo
from .snippets.variablegroups import VariableGroup
from .snippets.databases import NetCDF
from .snippets.dataobjects import PointSet
from .snippets.distributions import Distribution
from .snippets.models import RavenCode
from .snippets.outstreams import PrintOutStream, OptPathPlot
from .snippets.samplers import SampledVariable, Sampler, Grid, MonteCarlo, CustomSampler
from .snippets.steps import MultiRun


class BilevelTemplate(RavenTemplate):
  """ Coordinates information between inner and outer templates for bilevel workflows """

  def __init__(self, mode: str, has_static_history: bool, has_synthetic_history: bool) -> None:
    """
    Constructor
    @ In, case, HeronCase, the HERON case object
    @ In, source, list[Source], sources
    """
    super().__init__()
    if has_static_history and has_synthetic_history:
      raise ValueError("Bilevel HERON workflows expect either a static history source (<CSV>) or a synthetic history "
                       "source (<ARMA>) but not both! Check your input file.")

    self.inner = InnerTemplateStaticHistory() if has_static_history else InnerTemplateSyntheticHistory()

    if mode == "sweep":
      self.outer = OuterTemplateSweep()
    elif mode == "opt":
      self.outer = OuterTemplateOpt()
    else:
      raise ValueError(f"Unsupported case mode '{mode}' in Bilevel workflow template.")

  @property
  def template_name(self) -> dict[str, str]:
    """
      Getter property for the template template_name attributes. Allows for compatibility with acces strategy for
      template classes which use only one template file.
      @ In, None
      @ Out, dict[str, str], inner and outer template names
    """
    return {"inner": self.inner.template_name, "outer": self.outer.template_name}

  @property
  def write_name(self) -> dict[str, str]:
    """
      Getter property for the template write_name attributes. Allows for compatibility with acces strategy for template
      classes which use only one template file.
      @ In, None
      @ Out, dict[str, str], inner and outer template write names
    """
    return {"inner": self.inner.write_name, "outer": self.outer.write_name}

  def loadTemplate(self, filename: dict[str, str], path: str) -> None:
    """
      Loads template file statefully.
      @ In, filename, dict[str, str], dictionary of names of files to load (xml)
      @ In, path, str, path to file relative to HERON/templates/
      @ Out, None
    """
    self.inner.loadTemplate(filename["inner"], path)
    self.outer.loadTemplate(filename["outer"], path)

  def createWorkflow(self, **kwargs) -> dict[str, ET.Element]:
    """
    Create a workflow for the specified Case and its components and sources
    @ In, kwargs, dict, keyword arguments
    @ Out, dict[str, ET.Element], modified template XML
    """
    self.inner.createWorkflow(**kwargs)

    # set the path to the inner sampler so the outer knows where to send aliased variables
    self.outer.inner_sampler = self.inner.get_sampler_path()
    self.outer.createWorkflow(**kwargs)

    # Coordinate across templates. The outer workflow needs to know where the inner workflow will save the dispatch
    # results to perform additional processing.
    case = kwargs["case"]
    disp_results_name = self.inner.get_dispatch_results_name()
    self.outer.set_inner_data_name(disp_results_name, case.data_handling["inner_to_outer"])

    return {"inner": self.inner.template_xml, "outer": self.outer.template_xml}

  def writeWorkflow(self, template: dict[str, ET.Element], destination: dict[str, str], run: bool = False) -> None:
    """
      Writes a template to file.
      @ In, template, dict[str, ET.Element], XML trees to write
      @ In, destination, dict[str, str], paths to write the templates to
      @ In, run, bool, optional, if True then run the workflow after writing? good idea?
      @ Out, errors, int, 0 if successfully wrote [and run] and nonzero if there was a problem
    """
    for name, xml in template.items():
      super().writeWorkflow(xml, destination[name], run)

    # copy "write_inner.py", which has the denoising and capacity fixing algorithms
    conv_filename = "write_inner.py"
    write_inner_dir = Path(__file__).parent
    dest_dir = Path(next(iter(destination.values()))).parent
    conv_src = write_inner_dir / conv_filename
    conv_file = dest_dir / conv_filename
    shutil.copyfile(str(conv_src), str(conv_file))
    print(f"Wrote '{conv_filename}' to '{destination}'")

  @property
  def template_xml(self) -> dict[str, ET.Element]:
    """
      Getter property for the template XML ET.Element tree
      @ In, None
      @ Out, _templates, dict[str, ET.Element], the XML trees
    """
    return {"inner": self.inner.template_xml, "outer": self.outer.template_xml}

  def get_write_path(self, dest_dir: str) -> dict[str, str]:
    """
      Get the path of to write the template to
      @ In, dest_dir, str, the directory to write the file to
      @ Out, paths, dict[str, str], the path (directory + file name) to write to for each template
    """
    return {"inner": self.inner.get_write_path(dest_dir), "outer": self.outer.get_write_path(dest_dir)}

class OuterTemplate(RavenTemplate):
  """ Base class for modifying the outer workflow in bilevel workflows """
  write_name = "outer.xml"

  def __init__(self) -> None:
    """
    Constructor
    @ In, None
    @ Out, None
    """
    super().__init__()
    self.inner_sampler = None

  def createWorkflow(self, **kwargs) -> None:
    """
    Create a workflow for the specified Case and its components and sources
    @ In, kwargs, dict, keyword arguments
    @ Out, None
    """
    super().createWorkflow(**kwargs)

    case = kwargs["case"]
    case_name = self.namingTemplates["jobname"].format(case=case.name, io="o")
    self._set_case_name(case_name)
    self._initialize_runinfo(case)

    components = kwargs["components"]

    # Configure the RAVEN model
    self._configure_raven_model(case, components)
    # Populate the capacities and outer_results variable groups
    self._configure_variable_groups(case, components)

  def set_inner_data_name(self, name: str, inner_to_outer: str) -> None:
    """
    Give the RAVEN code model the place to look for the inner's data
    @ In, name, str, the name of the data objectw
    @ In, inner_to_outer, str, the type of file used to pass the data ("csv" or "netcdf")
    @ Out, None
    """
    model = self._template.find("Models/Code[@subType='RAVEN']")  # type: RavenCode
    model.set_inner_data_handling(name, inner_to_outer)

  def _initialize_runinfo(self, case: HeronCase) -> None:
    """
    Initializes the RunInfo node of the workflow
    @ In, case, Case, the HERON Case object
    @ Out, None
    """
    run_info = self._template.find("RunInfo")  # type: RunInfo

    # parallel
    if case.outerParallel > 0:
      # set outer batchsize and InternalParallel
      run_info.batch_size = case.outerParallel
      run_info.use_internal_parallel = True
    else:
      run_info.batch_size = 1

    if case.useParallel:
      # Fills in parallel settings for template RunInfo from case. Also appliespre-sets for known
      # hostnames (e.g. sawtooth, bitterroot), as specified in the HERON/templates/parallel/*.xml files.
      run_info.set_parallel_run_settings(case.parallelRunInfo)

    if case.innerParallel:
      run_info.num_mpi = case.innerParallel

  def _configure_raven_model(self, case: HeronCase, components: list[Component]) -> RavenCode:
    """
    Configures the inner RAVEN code. The bilevel outer template MUST have a <Code subType="RAVEN"> node defined.
    @ In, case, HeronCase, the HERON case
    @ In, components, list[Component], the case components
    @ Out, raven, RavenCode, the RAVEN code node
    """
    raven = self._template.find("Models/Code[@subType='RAVEN']")  # type: RavenCode

    # Find the RAVEN executable to use
    exec_path = RAVEN_LOC / "raven_framework"
    if exec_path.resolve().exists():
      executable = str(exec_path.resolve())
    elif shutil.which("raven_framework") is not None:
      executable = "raven_framework"
    else:
      raise RuntimeError(f"raven_framework not in PATH and not at {exec_path}")
    raven.executable = executable

    # custom python command for running raven (for example, "coverage run")
    if cmd := case.get_py_cmd_for_raven():
      raven.python_command = cmd

    # Add alias for the number of denoises
    raven.add_alias("denoises", loc=self.inner_sampler)

    # Add variable aliases for Inner
    for component in components:
      raven.add_alias(component.name, suffix="capacity", loc=self.inner_sampler)

    # Add label aliases for Inner
    for label in case.get_labels():
      raven.add_alias(label, suffix="label", loc=self.inner_sampler)

    return raven

  def _configure_variable_groups(self, case: HeronCase, components: list[Component]) -> None:
    """
    Fills out variable groups with capacity and results names
    @ In, case, HeronCase, the HERON case
    @ In, components, list[Component], the case components
    @ Out, None
    """
    # Set up some helpful variable groups
    capacities_vargroup = self._template.find("VariableGroups/Group[@name='GRO_capacities']")
    capacities_vars = list(get_capacity_vars(components, self.namingTemplates["variable"]))
    capacities_vargroup.variables.extend(capacities_vars)

    results_vargroup = self._template.find("VariableGroups/Group[@name='GRO_outer_results']")
    results_vars = self._get_statistical_results_vars(case, components)
    results_vargroup.variables.extend(results_vars)

  def _set_batch_size(self, batch_size: int, case: HeronCase) -> None:
    """
    Sets the batch size in RunInfo and sets internalParallel to True
    @ In, batch_size, int, the batch size
    @ In, case, HeronCase, the HERON case
    @ Out, None
    """
    case.outerParallel = batch_size
    run_info = self._template.find("RunInfo")
    run_info.batch_size = batch_size
    run_info.internal_parallel = True


class OuterTemplateOpt(OuterTemplate):
  """ Sets up the outer workflow for optimization mode """
  template_name = "outer_opt.xml"

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

    # Define XML blocks for optimization: optimizer, sampler, ROM, etc.
    opt_strategy = case.get_opt_strategy()
    if opt_strategy == "BayesianOpt":
      optimizer = self._create_bayesian_opt(case, components)
    elif opt_strategy == "GradientDescent":
      optimizer = self._create_gradient_descent(case, components)
    else:
      raise ValueError(f"Template does not recognize optimization strategy {opt_strategy}.")

    # Set optimizer <TargetEvaluation> data object
    results_data = self._template.find("DataObjects/PointSet[@name='opt_eval']")
    optimizer.target_evaluation = results_data

    # Set optimizer objective function
    objective = get_opt_objective(case)
    optimizer.objective = objective
    results = self._template.find("VariableGroups/Group[@name='GRO_outer_results']")  # type: VariableGroup
    if objective not in results.variables:
      results.variables.insert(0, objective)

    # Add case labels to the optimizer
    self._add_labels_to_sampler(optimizer, case.get_labels())

    # Add the optimizer and any custom function files to the main MultiRun step
    multirun = self._template.find("Steps/MultiRun[@name='optimize']")  # type: MultiRun
    for func in self._get_function_files(sources):
      multirun.add_input(func)
    multirun.add_optimizer(optimizer)

    # Add the optimization objective to the opt_path plot variables
    opt_path_plot = self._template.find("OutStreams/Plot[@subType='OptPath']")  # type: OptPathPlot
    opt_path_plot.variables.append(objective)

    # Update the parallel settings based on the number of sampled variables if the number of outer parallel runs
    # was not specified before.
    if case.outerParallel == 0 and case.useParallel:
      batch_size = optimizer.num_sampled_vars + 1
      self._set_batch_size(batch_size, case)


class OuterTemplateSweep(OuterTemplate):
  """ Sets up the outer workflow for sweep mode """
  template_name = "outer_sweep.xml"

  def createWorkflow(self, **kwargs) -> None:
    """
    Create a workflow for the specified Case and its components and sources
    @ In, kwargs, dict, keyword arguments
    @ Out, None
    """
    super().createWorkflow(**kwargs)
    case = kwargs["case"]
    components = kwargs["components"]

    # Populate the sampled and constant capacities in the Grid sampler
    sampler = self._template.find("Samplers/Grid")  # type: Grid
    variables, consts = self._create_sampler_variables(case, components)
    for sampled_var, vals in variables.items():
      sampler.add_variable(sampled_var)
      sampled_var.use_grid(construction="custom", kind="value", values=sorted(vals))
    for var_name, val in consts.items():
      sampler.add_constant(var_name, val)

    # Number of "denoises" for the sampler is the number of samples it should take
    sampler.denoises = case.get_num_samples()

    # If there are any case labels, make a variable group for those and add it to the "grid" PointSet.
    # These labels also need to get added to the sampler as constants.
    grid_results = self._template.find("DataObjects/PointSet[@name='grid']")  # type: PointSet
    labels = case.get_labels()
    if labels:
      vargroup = self._create_case_labels_vargroup(labels)
      self._add_snippet(vargroup)
      grid_results.outputs.append(vargroup.name)
    self._add_labels_to_sampler(sampler, labels)

    # Update the parallel settings based on the number of sampled variables if the number of outer parallel runs
    # was not specified before.
    if case.outerParallel == 0 and case.useParallel:
      batch_size = sampler.num_sampled_vars + 1
      self._set_batch_size(batch_size, case)


class InnerTemplate(RavenTemplate):
  """ Template for the inner workflow of a bilevel problem """
  write_name = "inner.xml"

  def __init__(self) -> None:
    """
    Constructor
    @ In, None
    @ Out, None
    """
    super().__init__()
    self._dispatch_results_name = ""  # str, keeps track of the name of the Database or OutStream used pass dispatch
                                      #      data to the outer workflow

  def createWorkflow(self, **kwargs) -> None:
    """
    Create a workflow for the specified Case and its components and sources
    @ In, kwargs, dict, keyword arguments
    @ Out, None
    """
    super().createWorkflow(**kwargs)
    case = kwargs["case"]
    components = kwargs["components"]

    case_name = self.namingTemplates["jobname"].format(case=case.name, io="i")
    self._set_case_name(case_name)
    self._initialize_runinfo(case)

    # Set the index variable names for the time index names
    self._set_time_vars(case.get_time_name(), case.get_year_name())

    # Figure out econ metrics are being used for the case
    #   - econ metrics (from case obj), total activity variables (assembled from components list)
    #   - add to output groups GRO_dispatch_out, GRO_timeseries_out_scalar
    #   - add to metrics data object (arma_metrics PointSet)
    activity_vars = get_component_activity_vars(components, self.namingTemplates["tot_activity"])
    econ_vars = case.get_econ_metrics(nametype="output")
    output_vars = econ_vars + activity_vars
    self._template.find("VariableGroups/Group[@name='GRO_dispatch_out']").variables.extend(output_vars)
    self._template.find("VariableGroups/Group[@name='GRO_timeseries_out_scalar']").variables.extend(output_vars)
    self._template.find("DataObjects/PointSet[@name='arma_metrics']").outputs.extend(output_vars)

    # Figure out what result statistics are being used
    vg_final_return = self._template.find("VariableGroups/Group[@name='GRO_metrics_stats']")
    results_vars = self._get_statistical_results_vars(case, components)
    vg_final_return.variables.extend(results_vars)

    # Fill out the econ postprocessor statistics
    econ_pp = self._template.find("Models/PostProcessor[@name='statistics']")
    for stat, variable in self._get_stats_for_econ_postprocessor(case, econ_vars, activity_vars):
      econ_pp.append(stat.to_element(variable))

    # Work out how the inner results should be routed back to the outer
    self._handle_data_inner_to_outer(case)

  def get_sampler_path(self) -> str:
    """
    Getter for the pipe-delimited path to the first sampler in the workflow
    @ In, None
    @ Out, path, str, the path to the sampler
    """
    sampler = self._template.find("Samplers")[0]
    path = f"Samplers|{sampler.tag}@name:{sampler.name}"
    return path

  def _handle_data_inner_to_outer(self, case: HeronCase) -> None:
    """
    Set up either a Database or DataObject for the outer workflow to read
    @ In, case, HeronCase, the HERON case object
    @ Out, None
    """
    # Work out how the inner results should be routed back to the outer
    metrics_stats = self._template.find("DataObjects/PointSet[@name='metrics_stats']")
    write_metrics_stats = self._template.find("Steps/IOStep[@name='database']")
    self._dispatch_results_name = "disp_results"
    data_handling = case.data_handling["inner_to_outer"]
    if data_handling == "csv":
      disp_results = PrintOutStream(self._dispatch_results_name)
      disp_results.source = metrics_stats
    else:  # default to NetCDF handling
      disp_results = NetCDF(self._dispatch_results_name)
      disp_results.read_mode = "overwrite"
    self._add_snippet(disp_results)
    write_metrics_stats.add_output(disp_results)

  def get_dispatch_results_name(self) -> str:
    """
    Gets the name of the Database or OutStream used to export the dispatch results to the outer workflow
    @ In, None
    @ Out, disp_results_name, str, the name of the dispatch results object
    """
    if not self._dispatch_results_name:
      raise ValueError("No dispatch results object name has been set! Perhaps the inner workflow hasn't "
                       "been created yet?")
    return self._dispatch_results_name

  def _initialize_runinfo(self, case: HeronCase) -> None:
    """
    Initializes the RunInfo node of the workflow
    @ In, case, Case, the HERON Case object
    @ Out, None
    """
    run_info = self._template.find("RunInfo")

    # parallel settings
    if case.innerParallel > 0:
      run_info.use_internal_parallel = True
      run_info.batch_size = case.innerParallel
    else:
      run_info.batch_size = 1

  def _add_case_labels_to_sampler(self, case_labels: dict[str, str], sampler: Sampler) -> None:
    """
    Adds case labels to relevant variable groups
    @ In, case_labels, dict[str, str], the case labels
    @ In, sampler, Sampler, the sampler to add labels to
    @ Out, None
    """
    if not case_labels:
      return

    vg_case_labels = VariableGroup("GRO_case_labels")
    self._add_snippet(vg_case_labels)
    self._template.find("VariableGroups/Group[@name='GRO_timeseries_in_scalar']").variables.append(vg_case_labels.name)
    self._template.find("VariableGroups/Group[@name='GRO_dispatch_in_scalar']").variables.append(vg_case_labels.name)
    for k, label_val in case_labels.items():
      label_name = self.namingTemplates["variable"].format(unit=k, feature="label")
      vg_case_labels.variables.append(label_name)
      sampler.add_constant(label_name, label_val)

  def _set_time_vars(self, time_name: str, year_name: str) -> None:
    """
    Update variable groups and data objects to have the correct time variable names.
    @ In, time_name, str, name of time variable
    @ In, year_name, str, name of year variable
    @ Out, None
    """
    group = self._template.find("VariableGroups/Group[@name='GRO_dispatch']")
    group.variables.extend([time_name, year_name])

    for time_index in self._template.findall("DataObjects/DataSet/Index[@var='Time']"):
      time_index.set("var", time_name)

    for year_index in self._template.findall("DataObjects/DataSet/Index[@var='Year']"):
      year_index.set("var", year_name)

  def _add_uncertain_econ_params(self,
                                 sampler: Sampler,
                                 variables: list[SampledVariable],
                                 distributions: list[Distribution]) -> VariableGroup:
    """
    Add uncertain economic parameter variables to the sampler and appropriate variable groups
    @ In, sampler, Sampler, the sampler
    @ In, variables, list[SampledVariable], variables to be sampled
    @ In, distributions, list[Distribution], distributions to be sampled from
    @ Out, vg_econ_uq, VariableGroup, a VariableGroup with the economic parameter names
    """
    vg_econ_uq = self._template.find("VariableGroups/Group[@name='GRO_UQ']")
    if vg_econ_uq is None:
      vg_econ_uq = VariableGroup("GRO_UQ")
      self._add_snippet(vg_econ_uq)
    # Add the SampledVariable and Distribution nodes to the appropriate locations
    for samp_var, dist in zip(variables, distributions):
      self._add_snippet(dist)
      vg_econ_uq.variables.append(samp_var.name)
      sampler.add_variable(samp_var)
    return vg_econ_uq

  def _add_constant_caps_to_sampler(self, sampler: Sampler, components: list[Component]) -> None:
    """
    Add capacity values to the sampler as constants
    @ In, sampler, Sampler, the sampler
    @ In, components, list[Component], the case components
    @ Out, None
    """
    capacities_vargroup = self._template.find("VariableGroups/Group[@name='GRO_capacities']")  # type: VariableGroup
    capacities_vars = get_capacity_vars(components, self.namingTemplates["variable"])
    capacities_vargroup.variables.extend(list(capacities_vars))
    for k, v in capacities_vars.items():
      val = "" if isinstance(v, list) else v  # empty string is overwritten by capacity from outer in write_inner.py
      sampler.add_constant(k, val)


class InnerTemplateSyntheticHistory(InnerTemplate):
  """ Template for the inner workflow of a bilevel problem that uses a static history source """
  template_name = "inner_synth.xml"

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

    # Add ARMA ROMs to ensemble model
    ensemble_model = self._template.find("Models/EnsembleModel")
    self._add_time_series_roms(ensemble_model, case, sources)

    # Determine which variables are sampled by the Monte Carlo sampler
    mc = self._template.find("Samplers/MonteCarlo[@name='mc_arma_dispatch']")  # type: MonteCarlo
    # default sampler init
    mc.init_seed = 42
    mc.init_limit = 3
    # Add capacities as constants to the sampler
    self._add_constant_caps_to_sampler(mc, components)

    # Add case labels to sampler and variable groups, if any labels have been provided
    self._add_case_labels_to_sampler(case.get_labels(), mc)

    # See if there are any uncertain cashflow parameters that need to get added to the sampler
    sampled_vars, distributions = self._get_uncertain_cashflow_params(components)
    if len(sampled_vars) > 0:
      # Create a VariableGroup for the uncertain econ parameters
      vg_econ_uq = self._add_uncertain_econ_params(mc, sampled_vars, distributions)
      self._template.find("VariableGroups/Group[@name='GRO_dispatch_in_scalar']").variables.append(vg_econ_uq.name)
      self._template.find("VariableGroups/Group[@name='GRO_timeseries_in_scalar']").variables.append(vg_econ_uq.name)


class InnerTemplateStaticHistory(InnerTemplate):
  """ Template for the inner workflow of a bilevel problem """
  template_name = "inner_static.xml"

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

    # Create the custom sampler used to provide static history data to the model
    custom_sampler = CustomSampler("static_history_sampler")
    self._configure_static_history_sampler(custom_sampler, case, sources)

    # Add case labels to the sampler
    self._add_case_labels_to_sampler(case.get_labels(), custom_sampler)

    # Add the outer capacities as constants here
    #   - component capacities (constants)
    #     - add variables to GRO_capacities
    capacities_vargroup = self._template.find("VariableGroups/Group[@name='GRO_capacities']")  # type: VariableGroup
    capacities_vars = get_capacity_vars(components, self.namingTemplates["variable"])
    capacities_vargroup.variables.extend(list(capacities_vars))
    for k, v in capacities_vars.items():
      val = "" if isinstance(v, list) else v  # empty string is overwritten by capacity from outer in write_inner.py
      custom_sampler.add_constant(k, val)

    # See if there are any uncertain cashflow parameters. If so, we need to create a MonteCarlo sampler to sample
    # from those distributions and tie the MonteCarlo and CustomSampler samplers together with an EnsembleForward
    # sampler.
    sampled_vars, distributions = self._get_uncertain_cashflow_params(components)
    if len(sampled_vars) > 0:
      # Create a MonteCarlo sampler
      mc = MonteCarlo("mc")
      mc.init_seed = 42
      mc.init_limit = case.get_num_samples()
      # Create a VariableGroup for the uncertain econ parameters
      vg_econ_uq = self._add_uncertain_econ_params(mc, sampled_vars, distributions)
      self._template.find("VariableGroups/Group[@name='GRO_dispatch_in_scalar']").variables.append(vg_econ_uq.name)
      self._template.find("VariableGroups/Group[@name='GRO_timeseries_in_scalar']").variables.append(vg_econ_uq.name)

      # Combine the MonteCarlo and CustomSampler samplers in an EnsembleForward sampler.
      ensemble_sampler = self._create_ensemble_forward_sampler(custom_sampler, mc)
      self._add_snippet(ensemble_sampler)

      sampler = ensemble_sampler
    else:
      self._add_snippet(custom_sampler)
      sampler = custom_sampler

    # Set the sampler to be used in the main MultiRun
    multirun = self._template.find("Steps/MultiRun[@name='arma_sampling']")
    multirun.add_sampler(sampler)

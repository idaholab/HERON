# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  RAVEN workflow templates

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-10-29
"""
import os
import re
import glob
from pathlib import Path
import itertools as it
import xml.etree.ElementTree as ET

from .imports import xmlUtils, Template
from .heron_types import HeronCase, Component, Source, ValuedParam
from .naming_utils import get_result_stats, get_component_activity_vars, get_opt_objective, get_statistics, Statistic
from .xml_utils import add_node_to_tree, stringify_node_values

from .snippets.base import RavenSnippet
from .snippets.runinfo import RunInfo
from .snippets.steps import Step, IOStep
from .snippets.samplers import Sampler, SampledVariable, Grid, Stratified, CustomSampler, EnsembleForward
from .snippets.optimizers import BayesianOptimizer, GradientDescent
from .snippets.models import GaussianProcessRegressor, PickledROM, EnsembleModel
from .snippets.distributions import Distribution, Uniform
from .snippets.outstreams import PrintOutStream
from .snippets.dataobjects import DataObject, PointSet, DataSet
from .snippets.variablegroups import VariableGroup
from .snippets.files import File
from .snippets.factory import factory as snippet_factory


# NOTE: Leave this here! Moving this to xml_utils.py will cause a circular import problem with snippets.factory.py
def parse_to_snippets(node: ET.Element) -> ET.Element:
  """
  Builds an XML tree that looks exactly like node but with RavenSnippet objects where defined.
  @ In, node, ET.Element, the node to parse
  @ Out, parsed: ET.Element, the parsed XML node
  """
  # Base case: The node matches a registered RavenSnippet class. RavenSnippets know how to represent
  # their entire contiguous block of XML, so no further recursion is necessary once a valid RavenSnippet
  # is found.
  if snippet_factory.has_registered_class(node):
    snippet = snippet_factory.from_xml(node)
    return snippet

  # If the node doesn't match a registered RavenSnippet class, copy over the node to the
  parsed = ET.Element(node.tag, node.attrib)
  parsed.text = node.text
  parsed.tail = node.tail

  # Recurse over node children (if any)
  for child in node:
    parsed_child = parse_to_snippets(child)
    parsed.append(parsed_child)

  return parsed

class RavenTemplate(Template):
  """ Template class for RAVEN workflows """
  # Default stats abbreviations. Different run modes have different defaults
  DEFAULT_STATS_NAMES = {
    "opt": ["expectedValue", "sigma", "median"],
    "sweep": ["maximum", "minimum", "percentile", "samples", "variance"]
  }

  # Prefixes for financial metrics only
  FINANCIAL_PREFIXES = ["sharpe", "sortino", "es", "VaR", "glr"]
  FINANCIAL_STATS_NAMES = ["sharpeRatio", "sortinoRatio", "expectedShortfall", "valueAtRisk", "gainLossRatio"]

  def __init__(self) -> None:
    """
    Constructor
    @ In, None
    @ Out, None
    """
    super().__init__()
    # Naming templates
    self.addNamingTemplates({"jobname"        : "{case}_{io}",
                             "stepname"       : "{action}_{subject}",
                             "variable"       : "{unit}_{feature}",
                             "dispatch"       : "Dispatch__{component}__{tracker}__{resource}",
                             "tot_activity"   : "TotalActivity__{component}__{tracker}__{resource}",
                             "data object"    : "{source}_{contents}",
                             "distribution"   : "{variable}_dist",
                             "ARMA sampler"   : "{rom}_sampler",
                             "lib file"       : "heron.lib", # TODO use case name?
                             "cashfname"      : "_{component}{cashname}",
                             "re_cash"        : "_rec_{period}_{driverType}{driverName}",
                             "cluster_index"  : "_ROM_Cluster",
                             "metric_name"    : "{stats}_{econ}",
                             "statistic"      : "{prefix}_{name}"
                             })
    self._template = None

  ########################
  # PUBLIC API FUNCTIONS #
  ########################

  def loadTemplate(self, filename: str, path: str) -> None:
    """
      Loads template file statefully.
      @ In, filename, str, name of file to load (xml)
      @ In, path, str, path to file relative to HERON/templates/
      @ Out, None
    """
    super().loadTemplate(filename, path)
    self._template = parse_to_snippets(self._template)

  def createWorkflow(self, **kwargs) -> None:
    """
    Create a workflow for the specified Case and its components and sources
    @ In, kwargs, dict, keyword arguments
    @ Out, None
    """
    # Universal workflow settings
    self._set_verbosity(kwargs["case"].get_verbosity())

  def writeWorkflow(self, template: ET.Element, destination: str, run: bool = False) -> None:
    """
      Writes a template to file.
      @ In, template, xml.etree.ElementTree.Element, file to write
      @ In, destination, str, path and filename to write to
      @ In, run, bool, optional, if True then run the workflow after writing? good idea?
      @ Out, errors, int, 0 if successfully wrote [and run] and nonzero if there was a problem
    """
    # Ensure all node attribute values and text are expressed as strings. Errors are thrown if any of these aren't
    # strings. Enforcing this here allows flexibility with how node values are stored and manipulated before write
    # time, such as storing values as lists or numeric types. For example, text fields which are a comma-separated
    # list of values can be stored in the RavenSnippet object as a list, and new items can be inserted into that
    # list as needed, then the list can be converted to a string only now at write time.
    stringify_node_values(template)

    # Remove any unused top-level nodes (Models, Samplers, etc.) to keep things looking clean
    for node in template:
      if len(node) == 0:
        template.remove(node)

    super().writeWorkflow(template, destination, run)
    print(f"Wrote '{self.write_name}' to '{destination}'")

  @property
  def template_xml(self) -> ET.Element:
    """
      Getter property for the template XML ET.Element tree
      @ In, None
      @ Out, _template, ET.Element, the XML tree
    """
    return self._template

  def get_write_path(self, dest_dir: str) -> str:
    """
      Get the path of to write the template to
      @ In, dest_dir, str, the directory to write the file to
      @ Out, path, str, the path (directory + file name) to write to
    """
    write_name = getattr(self, "write_name", None)
    if not write_name:
      raise ValueError(f"Template class {self.__class__.__name__} object has no 'write_name' attribute.")
    path = os.path.join(dest_dir, write_name)
    return path

  #####################
  # SNIPPET UTILITIES #
  #####################
  def _add_snippet(self, snippet: RavenSnippet, parent: str | ET.Element | None = None) -> None:
    """
    Add an XML snippet to the template XML
    @ In, snippet, RavenSnippet, the XML snippet to add
    @ In, parent, str | ET.Element | None, the parent node to add the snippet
    @ Out, None
    """
    if isinstance(snippet, ET.Element) and not isinstance(snippet, RavenSnippet):
      raise TypeError(f"The XML block to be added is not a RavenSnippet object. Received type: {type(snippet)}. "
                      "Perhaps something went wrong when parsing the template XML, and the correct RavenSnippet "
                      "subclass wasn't found?")
    if snippet is None:
      raise ValueError("Received None instead of a RavenSnippet object. Perhaps something went wrong when finding "
                       "an XML node?")

    # If a parent node was provided, just append the snippet to its parent node.
    if isinstance(parent, ET.Element):
      parent.append(snippet)
      return

    # Otherwise, figure out where to put the XML snippet. Either a string for a parent node (maybe doesn't exist yet)
    # was provided, or the desired location is inferred from the snippet class (e.g. Models, DataObjects, etc.).
    if parent and isinstance(parent, str):
      parent_path = parent
    else:
      # Find parent node based on snippet "class" attribute
      parent_path = snippet.snippet_class

    if parent_path is None:
      raise ValueError(f"The path to a parent node for node {snippet} could not be determined!")

    # Make the parent node if it doesn't exist. This is helpful if it's unknown if top-level nodes (Models, Optimizers,
    # Steps, etc.) exist without having to add a check everywhere a snippet needs to get added.
    add_node_to_tree(snippet, parent_path, self._template)

  ##############################
  # FEATURE BUILDING UTILITIES #
  ##############################
  # These functions help set options and build workflow features. They are roughly organized by which portion of the
  # RAVEN template they modify.

  # Global attributes
  def _set_verbosity(self, verbosity: str) -> None:
    """
    Sets the verbosity attribute of the root Simulation node
    @ In, verbosity, str, the verbosity level
    @ Out, None
    """
    self._template.set("verbosity", verbosity)

  def _set_case_name(self, name: str) -> None:
    """
    Sets the JobName and WorkingDir values in the RunInfo block to the given name
    @ In, name, str, case name to use
    @ Out, None
    """
    run_info = self._template.find("RunInfo")  # type: RunInfo
    run_info.job_name = name
    run_info.working_dir = name

  def _add_step_to_sequence(self, step: Step, index: int | None = None) -> None:
    """
    Add a step to the Sequence node
    @ In, step, Step, the step to add
    @ In, index, int, optional, the index to add the step at
    @ Out, None
    """
    run_info = self._template.find("RunInfo")  # type: RunInfo
    idx = index if index is not None else len(run_info.sequence)
    run_info.sequence.insert(idx, step)

  # TODO refactor to fit snippet style
  # from PR #397
  def _get_parallel_xml(self, hostname):
    """
      Finds the xml file to go with the given hostname.
      @ In, hostname, string with the hostname to search for
      @ Out, xml, xml.eTree.ElementTree or None, if an xml file is found then use it, otherwise return None
    """
    # Should this allow loading from another directory (such as one
    #  next to the input file?)
    path = os.path.join(os.path.dirname(__file__),"parallel","*.xml")
    filenames = glob.glob(path)
    for filename in filenames:
      cur_xml = ET.parse(filename).getroot()
      regexp = cur_xml.attrib['hostregexp']
      if re.match(regexp, hostname):
        return cur_xml
    return None

  # Steps
  def _load_file_to_object(self, source: Source, target: RavenSnippet) -> IOStep:
    """
    Load a source file to a target object
    @ In, source, Source, the source to load
    @ In, target, RavenSnippet, the object to load to
    @ Out, step, IOStep, the step used to do the loading
    """
    # Get the file to load. Might already exist in the template XML
    file = self._template.find("Files/Input[@name='{source.name}']")  # type: File
    if file is None:
      file = File(source.name)
      file.path = source._target_file

    # Create an IOStep to load the file to the target
    step_name = self.namingTemplates["stepname"].format(action="read", subject=source.name)
    step = IOStep(step_name)
    step.append(file.to_assembler_node("Input"))
    step.append(target.to_assembler_node("Output"))

    self._add_snippet(file)
    self._add_snippet(step)

    return step

  @staticmethod
  def _load_pickled_rom(source: Source) -> tuple[File, PickledROM, IOStep]:
    """
    Loads a pickled ROM
    @ In, source, Source, the ROM source
    @ Out, file, File, a Files/Input snippet pointing to the ROM file
    @ Out, rom, PickledROM, the model snippet
    @ Out, step, IOStep, a step to load the model
    """
    # Create the Files/Input node for the ROM source file
    file = File(source.name)
    file.path = source._target_file

    # Create ROM snippet
    rom = PickledROM(source.name)
    if source.needs_multiyear is not None:
      rom.add_subelements({"Multicycle" : {"cycles": source.needs_multiyear}})
    if source.limit_interp is not None:
      rom.add_subelements(maxCycles=source.limit_interp)
    if source.eval_mode == 'clustered':
      ET.SubElement(rom, "clusterEvalMode").text = "clustered"

    # Create an IOStep to load the ROM from the file
    step = IOStep(f"read_{source.name}")
    step.append(file.to_assembler_node("Input"))
    step.append(rom.to_assembler_node("Output"))

    return file, rom, step

  @staticmethod
  def _print_rom_meta(rom: RavenSnippet) -> tuple[DataSet, PrintOutStream, IOStep]:
    """
    Print the metadata for a ROM, making the DataSet, Print OutStream, and IOStep to accomplish this.
    @ In, rom, RavenSnippet, the ROM to print
    @ Out, dataset, DataSet, the ROM metadata data object
    @ Out, outstream, PrintOutStream, the outstream to print the data object to file
    @ Out, step, IOStep, the step to print the ROM meta
    """
    if rom.snippet_class != "Models":
      raise ValueError("The RavenSnippet class provided is not a Model!")

    # Create the output data object
    dataset = DataSet(f"{rom.name}_meta")

    # Create the outstream for the dataset
    outstream = PrintOutStream(dataset.name)
    outstream.source = dataset

    # create step
    step = IOStep(f"print_{dataset.name}")
    step.append(rom.to_assembler_node("Input"))
    step.append(dataset.to_assembler_node("Output"))
    step.append(outstream.to_assembler_node("Output"))

    return dataset, outstream, step

  # VariableGroups
  def _create_case_labels_vargroup(self, labels: dict[str, str], name: str = "GRO_case_labels") -> VariableGroup:
    """
    Create a variable group for case labels
    @ In, labels, dict[str, str], the case labels
    @ In, name, str, optional, the name of the group
    @ Out, group, VariableGroup, the case labels variable group
    """
    group = VariableGroup(name)
    group.variables.extend(map(lambda label: f"{label}_label", labels.keys()))
    return group

  def _get_statistical_results_vars(self, case: HeronCase, components: list[Component]) -> list[str]:
    """
    Collects result metric names for statistical metrics. Should only be used with templates which have multiple
    time series samples.
    @ In, case, Case, HERON case
    @ In, components, list[Component], HERON components
    @ Out, var_names, list[str], list of variable names
    """
    # Add statistics for economic metrics to variable group. Use all statistics.
    default_names = self.DEFAULT_STATS_NAMES.get(case.get_mode(), [])
    # This gets the unique values from default_names and the case result statistics dict keys. Set operations
    # look cleaner but result in a randomly ordered list. Having a consistent ordering of statistics is beneficial
    # from a UX standpoint.
    stats_names = list(dict.fromkeys(default_names + list(case.get_result_statistics())))
    econ_metrics = case.get_econ_metrics(nametype="output")
    stats_var_names = get_result_stats(econ_metrics, stats_names, case)

    # Add total activity statistics for variable group. Use only non-financial statistics.
    non_fin_stat_names = [name for name in stats_names if name not in self.FINANCIAL_STATS_NAMES]
    tot_activity_metrics = get_component_activity_vars(components, self.namingTemplates["tot_activity"])
    activity_var_names = get_result_stats(tot_activity_metrics, non_fin_stat_names, case)

    var_names = stats_var_names + activity_var_names

    # The optimization objective might not have made it into the list. Make sure it's there.
    if case.get_mode() == "opt" and (objective := get_opt_objective(case)) not in var_names:
      var_names.insert(0, objective)

    return var_names

  def _get_deterministic_results_vars(self, case: HeronCase, components: list[Component]) -> list[str]:
    """
    Collects result metric names for deterministic cases
    @ In, case, Case, HERON case
    @ In, components, list[Component], HERON components
    @ Out, var_names, list[str], list of variable names
    """
    econ_metrics = case.get_econ_metrics(nametype="output")
    tot_activity_metrics = get_component_activity_vars(components, self.namingTemplates["tot_activity"])
    var_names = econ_metrics + tot_activity_metrics
    return var_names

  def _get_activity_metrics(self, components: list[Component]) -> list[str]:
    """
    Gets the names of component activity metrics
    @ In, components, list[Component], HERON components
    @ Out, act_metrics, list[str], component activity metric names
    """
    act_metrics = []
    for component in components:
      for tracker in component.get_tracking_vars():
        resource_list = sorted(list(component.get_resources()))
        for resource in resource_list:
          # NOTE: Assumes the only activity metric we care about is total activity
          default_stats_tot_activity = self.namingTemplates["tot_activity"].format(component=component.name,
                                                                                   tracker=tracker,
                                                                                   resource=resource)
          act_metrics.append(default_stats_tot_activity)
    return act_metrics

  # Models
  def _add_time_series_roms(self, ensemble_model: EnsembleModel, case: HeronCase, sources: list[Source]) -> None:
    """
    Create and modify snippets based on sources
    @ In, case, Case, HERON case
    @ In, sources, list[Source], case sources
    @ Out, None
    """
    dispatch_eval = self._template.find("DataObjects/DataSet[@name='dispatch_eval']")  # type: DataSet

    # Gather any ARMA sources from the list of sources
    arma_sources = [s for s in sources if s.is_type("ARMA")]

    # Add cluster index info to dispatch variable groups and data objects
    if any(source.eval_mode == "clustered" for source in arma_sources):
      vg_dispatch = self._template.find("VariableGroups/Group[@name='GRO_dispatch']")  # type: VariableGroup
      vg_dispatch.variables.append(self.namingTemplates["cluster_index"])
      dispatch_eval.add_index(self.namingTemplates["cluster_index"], "GRO_dispatch_in_Time")

    # Add models, steps, and their requisite data objects and outstreams for each case source
    for source in arma_sources:
      # An ARMA source is a pickled ROM that needs to be loaded.
      # Load the ROM from file
      source_file, pickled_rom, load_iostep = self._load_pickled_rom(source)
      self._add_snippet(source_file)
      self._add_snippet(pickled_rom)
      self._add_snippet(load_iostep)
      self._add_step_to_sequence(load_iostep, index=0)

      # Print the pickled ROM metadata
      meta_dataset, meta_outstream, meta_iostep = self._print_rom_meta(pickled_rom)
      self._add_snippet(meta_dataset)
      self._add_snippet(meta_outstream)
      self._add_snippet(meta_iostep)
      self._add_step_to_sequence(meta_iostep, index=1)

      # Add loaded ROM to the EnsembleModel
      inp_name = self.namingTemplates["data object"].format(source=source.name, contents="placeholder")
      inp_do = PointSet(inp_name)
      inp_do.inputs.append("scaling")
      self._add_snippet(inp_do)

      eval_name = self.namingTemplates["data object"].format(source=source.name, contents="samples")
      eval_do = DataSet(eval_name)
      eval_do.inputs.append("scaling")
      out_vars = source.get_variable()
      eval_do.outputs.extend(out_vars)
      eval_do.add_index(case.get_time_name(), out_vars)
      eval_do.add_index(case.get_year_name(), out_vars)
      if source.eval_mode == "clustered":
        eval_do.add_index(self.namingTemplates["cluster_index"], out_vars)
      self._add_snippet(eval_do)

      rom_assemb = pickled_rom.to_assembler_node("Model")
      rom_assemb.append(inp_do.to_assembler_node("Input"))
      rom_assemb.append(eval_do.to_assembler_node("TargetEvaluation"))
      ensemble_model.append(rom_assemb)

      # update variable group with ROM output variable names
      self._template.find("VariableGroups/Group[@name='GRO_dispatch_in_Time']").variables.extend(out_vars)

  def _get_stats_for_econ_postprocessor(self,
                                        case: HeronCase,
                                        econ_vars: list[str],
                                        activity_vars: list[str]) -> list[tuple[Statistic, str]]:
    """
    Get pairs of Statistic objects and metric/variable names to which to apply that statistic
    @ In, case, HeronCase, the HERON case
    @ In, econ_vars, list[str], economic metric names
    @ In, activity_vars, list[str], activity variable names
    @ Out, stats_to_add, list[tuple[Statistic, str]], statistics and the variables they act on
    """
    # Econ metrics with all statistics names
    # NOTE: This logic is borrowed from ravenTemplate._get_statistical_results_vars, but it's more useful to have the
    # names, prefixes, and variable name separate here, not as one big string. Otherwise, we have to try to break that
    # string back up, which would be sensitive to metric and variable naming conventions. We duplicate a little of the
    # logic but get something more robust in return.
    default_names = self.DEFAULT_STATS_NAMES.get(case.get_mode(), [])
    stats_names = list(dict.fromkeys(default_names + list(case.get_result_statistics())))
    econ_stats = get_statistics(stats_names, case.stats_metrics_meta)
    # Activity metrics with non-financial statistics
    non_fin_stat_names = [name for name in stats_names if name not in self.FINANCIAL_STATS_NAMES]
    activity_stats = get_statistics(non_fin_stat_names, case.stats_metrics_meta)

    # Collect the statistics to add to the postprocessor
    stats_to_add = list(it.chain(
      it.product(econ_stats, econ_vars),
      it.product(activity_stats, activity_vars)
    ))

    # The metric needed for the objective function might not have been added yet.
    if case.get_mode() == "opt":
      opt_settings = case.get_optimization_settings()
      try:
        statistic = opt_settings["stats_metric"]["name"]
      except (KeyError, TypeError):
        statistic = "expectedValue"  # default to expectedValue
      opt_stat = get_statistics([statistic], case.stats_metrics_meta)[0]
      target_var, _ = case.get_opt_metric()
      target_var_output_name = case.economic_metrics_meta[target_var]["output_name"]
      if (opt_stat, target_var_output_name) not in stats_to_add:
        stats_to_add.append((opt_stat, target_var_output_name))

    return stats_to_add

  # Distributions and SampledVariables
  def _create_new_sampled_capacity(self, var_name: str, capacities: list[float]) -> SampledVariable:
    """
    Creates a uniform distribution and SampledVariable object for a given list of capacities
    @ In, var_name, str, name of the variable
    @ In, capacities, list[float], list of capacity values
    @ Out, sampled_var, SampledVariable, variable to be sampled
    """
    dist_name = self.namingTemplates["distribution"].format(variable=var_name)
    dist = Uniform(dist_name)
    min_cap = min(capacities)
    max_cap = max(capacities)
    dist.lower_bound = min_cap
    dist.upper_bound = max_cap
    self._add_snippet(dist)

    sampled_var = SampledVariable(var_name)
    sampled_var.distribution = dist

    return sampled_var

  def _get_uncertain_cashflow_params(self,
                                     components: list[Component]) -> tuple[list[SampledVariable], list[Distribution]]:
    """
    Create SampledVariable and Distribution snippets for all uncertain cashflow parameters.
    @ In, components, list[Component]
    @ Out, sampled_vars, list[SampledVariable], objects to link sampler variables to distributions
    @ Out, distributions, list[Distribution], distribution snippets
    """
    sampled_vars = []
    distributions = []

    # For each component, cashflow, and cashflow equation parameter, find any which are uncertain, and create
    # distribution and sampled variable objects.
    for component in components:
      for cashflow in component.get_cashflows():
        for param_name, vp in cashflow.get_uncertain_params().items():
          unit_name = f"{component.name}_{cashflow.name}"
          feat_name = self.namingTemplates["variable"].format(unit=unit_name, feature=param_name)
          dist_name = self.namingTemplates["distribution"].format(variable=feat_name)

          # Reconstruct distribution XML node from valuedParam definition
          dist_node = vp._vp.get_distribution()  # type: ET.Element
          dist_node.set("name", dist_name)
          dist_snippet = snippet_factory.from_xml(dist_node)
          distributions.append(dist_snippet)

          # Create sampled variable snippet
          sampler_var = SampledVariable(feat_name)
          sampler_var.distribution = dist_snippet
          sampled_vars.append(sampler_var)

    return sampled_vars, distributions

  # Samplers
  def _create_sampler_variables(self,
                                case: HeronCase,
                                components: list[Component]) -> tuple[dict[SampledVariable, list[float]],
                                                                      dict[str, ValuedParam]]:
    """
    Create the Distribution and SampledVariable objects and the list of constant capacities that need to
    be added to samplers and optimizers.
    @ In, case, Case, HERON case
    @ In, components, list[Component], HERON components
    @ Out, sampled_variables, dict[SampledVariable, list[float]], variable objects for the sampler/optimizer
    @ Out, constants, dict[str, float], constant variables
    """
    sampled_variables = {}
    constants = {}

    # Make Distribution and SampledVariable objects for sampling dispatch variables
    for key, value in case.dispatch_vars.items():
      var_name = self.namingTemplates["variable"].format(unit=key, feature="dispatch")
      vals = value.get_value(debug=case.debug["enabled"])
      if isinstance(vals, list):
        sampled_var = self._create_new_sampled_capacity(var_name, vals)
        sampled_variables[sampled_var] = vals

    # Make Distribution and SampledVariable objects for capacity variables. Capacities with non-parametric
    # ValuedParams are fixed values and are added instead as constants.
    for component in components:
      interaction = component.get_interaction()
      name = component.name
      var_name = self.namingTemplates["variable"].format(unit=name, feature="capacity")
      cap = interaction.get_capacity(None, raw=True)  # type: ValuedParam

      if not cap.is_parametric():  # we already know the value
        continue

      vals = cap.get_value(debug=case.debug["enabled"])
      if isinstance(vals, list):  # multiple values meaning either opt bounds or sweep values
        sampled_var = self._create_new_sampled_capacity(var_name, vals)
        sampled_variables[sampled_var] = vals
      else:  # just one value meaning it's a constant
        constants[var_name] = vals

    return sampled_variables, constants

  def _add_labels_to_sampler(self, sampler: Sampler, labels: dict[str, str]) -> None:
    """
    Add case labels as constants for a sampler or optimizer
    @ In, sampler, Sampler, sampler to add labels to
    @ In, case, Case, HERON case
    @ Out, None
    """
    for key, value in labels.items():
      var_name = self.namingTemplates["variable"].format(unit=key, feature="label")
      sampler.add_constant(var_name, value)

  def _configure_static_history_sampler(self,
                                        custom_sampler: CustomSampler,
                                        case: HeronCase,
                                        sources: list[Source],
                                        scaling: int | None = 1.0) -> None:
    """
    Configures a sampler and relevant data objects and variable groups for using static histories in the workflow
    @ In, custom_sampler, CustomSampler, the sampler to use to sample the static histories
    @ In, case, HeronCase, the HERON case,
    @ In, sources, list[Source], the case sources
    @ In, scaling, int, optional, the scaling constant for the custom_sampler
    @ Out, None
    """
    indices = [case.get_year_name(), case.get_time_name()]
    cluster_index = self.namingTemplates["cluster_index"]
    if case.debug["enabled"]:
      indices.append(cluster_index)

    time_series_vargroup = self._template.find("VariableGroups/Group[@name='GRO_timeseries']")  # type: VariableGroup

    for source in filter(lambda x: x.is_type("CSV"), sources):
      # Add the source variables to the GRO_timeseries_in variable group
      source_vars = source.get_variable()
      self._template.find("VariableGroups/Group[@name='GRO_timeseries']").variables.extend(source_vars)

      # Create a new <DataObject> that will store the csv data
      csv_dataset = DataSet(source.name)
      csv_dataset.inputs.extend([case.get_time_name(), case.get_year_name()])
      csv_dataset.outputs.extend(source_vars)
      for index in indices:
        csv_dataset.add_index(index, source_vars)
      self._add_snippet(csv_dataset)

      # Use an IOStep to load the CSV data into the DataSet
      read_static = self._load_file_to_object(source, csv_dataset)
      self._add_step_to_sequence(read_static, index=0)

      # Add variables to the custom sampler for the
      custom_sampler.append(csv_dataset.to_assembler_node("Source"))
      for var in it.filterfalse(custom_sampler.has_variable, it.chain(indices, source_vars)):
        # NOTE: Being careful not to add duplicate time index variables to the custom sampler in case somebody tries
        # to include multiple CSV sources.
        custom_sampler.add_variable(SampledVariable(var))

      # Add the static history variables to the dispatch model
      new_vars = it.chain(
        source_vars,
        filter(lambda x: x not in time_series_vargroup.variables, indices)
      )
      time_series_vargroup.variables.extend(new_vars)

    if custom_sampler.find("constant[@name='scaling']") is None and scaling is not None:
      custom_sampler.add_constant("scaling", scaling)

  def _create_grid_sampler(self,
                           case: HeronCase,
                           components: list[Component],
                           capacity_vars: VariableGroup | str | list[str],
                           results_vars:VariableGroup | str | list[str]) -> tuple[Grid, PointSet]:
    """
    Creates a grid sampler for sweep cases
    @ In, case, the HERON case
    @ In, components, list[Component], the case components
    @ In, capacity_vars, VariableGroup | str | list[str], the capacity variable names
    @ In, results_vars, VariableGroup | str | list[str], the result stat/metric names
    @ Out, sampler, Grid, the grid sampler
    @ Out, results_data, PointSet, a data object to hold the results data at each grid point
    """
    # Define a PointSet for the results variables at each grid point
    results_data = PointSet("grid")

    if isinstance(capacity_vars, list):
      results_data.inputs.extend(capacity_vars)
    else:
      results_data.inputs.append(capacity_vars)

    if isinstance(results_vars, list):
      results_data.outputs.extend(results_vars)
    else:
      results_data.outputs.append(results_vars)

    self._add_snippet(results_data)

    # Define grid sampler and build the variables and their distributions that it'll sample
    sampler = Grid("grid")
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
    labels = case.get_labels()
    if labels:
      vargroup = self._create_case_labels_vargroup(labels)
      self._add_snippet(vargroup)
      results_data.outputs.append(vargroup.name)
    self._add_labels_to_sampler(sampler, labels)

    return sampler, results_data

  def _create_ensemble_forward_sampler(self, samplers: list[Sampler], name: str | None = None) -> EnsembleForward:
    """
    Wraps a list of samplers in an EnsembleForward sampler
    @ In, samplers, list[Sampler], the samplers to include in the ensemble
    @ In, name, str, optional, the name of the ensemble sampler
    @ Out, ensemble_sampler, EnsembleForward, the ensemble sampler; default: "ensemble_sampler"
    """
    ensemble_sampler = EnsembleForward(name or "ensemble_sampler")
    for sampler in samplers:
      ensemble_sampler.append(sampler)
    return ensemble_sampler

  # Optimizers
  def _create_bayesian_opt(self, case: HeronCase, components: list[Component]) -> BayesianOptimizer:
    """
    Set up the Bayesian optimization optimizer, LHS sampler, GPR model, and necessary distributions and data objects
    for using Bayesian optimization.
    @ In, case, HeronCase, the HERON case
    @ In, components, list[Component], the case components
    @ Out, optimizer, BayesianOptimizer, the Bayesian optimization node
    """
    # Create major XML blocks
    optimizer = BayesianOptimizer("cap_opt")
    sampler = Stratified("LHS_samp")
    gpr = GaussianProcessRegressor("gpROM")

    # Add blocks to XML template
    self._add_snippet(optimizer)
    self._add_snippet(sampler)
    self._add_snippet(gpr)

    # Connect optimizer to sampler and ROM components
    optimizer.set_sampler(sampler)
    optimizer.set_rom(gpr)

    # Apply any specified optimization settings
    opt_settings = case.get_optimization_settings() or {}  # default to empty dict if None
    optimizer.set_opt_settings(opt_settings)
    # Set GPR kernel if provided
    if opt_settings and (custom_kernel := opt_settings["algorithm"]["BayesianOpt"].get("kernel", None)):
      gpr.custom_kernel = custom_kernel

    # Create sampler variables and their respective distributions
    variables, consts = self._create_sampler_variables(case, components)
    for sampled_var in variables:
      sampled_var.use_grid(construction="equal", kind="CDF", steps=4, values=[0, 1])
      optimizer.add_variable(sampled_var)
      sampler.add_variable(sampled_var)
    for var_name, val in consts.items():
      optimizer.add_constant(var_name, val)

    # Set number of denoises
    optimizer.denoises = case.get_num_samples()

    # Set GPR features list and target
    for component in components:
      name = component.name
      interaction = component.get_interaction()
      cap = interaction.get_capacity(None, raw=True)
      if cap.is_parametric() and isinstance(cap.get_value(debug=case.debug["enabled"]) , list):
        gpr.features.append(self.namingTemplates["variable"].format(unit=name, feature="capacity"))
    gpr.target.append(get_opt_objective(case))

    return optimizer

  def _create_gradient_descent(self, case: HeronCase, components: list[Component]) -> GradientDescent:
    """
    Set up the gradient descent optimizer
    @ In, case, HeronCase, the HERON case
    @ In, components, list[Component], the case components
    @ Out, optimizer, GradientDescent, the gradient descent optimizer node
    """
    # Create necessary XML blocks
    optimizer = GradientDescent("cap_opt")
    self._add_snippet(optimizer)

    # Apply any specified optimization settings
    opt_settings = case.get_optimization_settings()
    optimizer.set_opt_settings(opt_settings)
    optimizer.objective = get_opt_objective(case)

    # Set number of denoises
    optimizer.denoises = case.get_num_samples()

    # Create sampler variables and their respective distributions
    variables, consts = self._create_sampler_variables(case, components)

    for sampled_var, vals in variables.items():
      # initial value
      min_val = min(vals)
      max_val = max(vals)
      delta = max_val - min_val
      # start 5% away from zero
      initial = min_val + 0.05 * delta if max_val > 0 else max_val - 0.05 * delta
      sampled_var.initial = initial
      optimizer.add_variable(sampled_var)

    for var_name, val in consts.items():
      optimizer.add_constant(var_name, val)

    return optimizer

  # Files
  def _get_function_files(self, sources: list[Source]) -> list[File]:
    """
    Get the File object for each source that is a Function
    @ In, sources, list[Source], the sources
    @ Out, files, list[File], the files
    """
    # Add Function sources as Files
    files = []
    for function in [s for s in sources if s.is_type("Function")]:
      file = self._template.find(f"Files/Input[@name='{function.name}']")
      if file is None:  # Add function to <Files> if not found there
        file = File(function.name)
        path = Path(function._source)
        # magic variable name that will get resolved later are like %VARNAME%/some/path
        if not str(path).startswith("%"):
          path = ".." / path
        file.path = path
        # file.path = Path(function._source)
        self._add_snippet(file)
      files.append(file)
    return files

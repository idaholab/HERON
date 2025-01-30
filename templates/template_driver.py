# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Driver for selecting the appropriate template, then creating and writing the workflow(s)

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-12-23
"""
from pathlib import Path
import dill as pk

from .imports import Base
from .heron_types import HeronCase, Component, Source
from .raven_template import RavenTemplate
from .bilevel_templates import BilevelTemplate
from .flat_templates import FlatMultiConfigTemplate
from .debug_template import DebugTemplate


class TemplateDriver(Base):
  """ Selects the best template to use for the given case, populate the workflow, and write it to file """
  def __init__(self, **kwargs) -> None:
    """
    Constructor
    @ In, kwargs, dict, keyword arguments to pass to Base.__init__
    @ Out, None
    """
    super().__init__(**kwargs)
    self.template: RavenTemplate = None

  ##############
  # Public API #
  ##############
  def create_workflow(self, case: HeronCase, components: list[Component], sources: list[Source]) -> None:
    """
    Create a workflow for the specified Case and its components and sources
    @ In, case, HeronCase, the HERON case
    @ In, components, list[Component], components in HERON case
    @ In, sources, list[Source], external models, data, and functions
    @ Out, None
    """
    has_static_history = any(s.is_type("CSV") for s in sources)
    has_synthetic_history = any(s.is_type("ARMA") for s in sources)
    if has_static_history and has_synthetic_history:
      self.raiseAnError(ValueError, "Mixing ARMA and CSV sources is not yet supported! "
                        f"ARMA sources: {[s.name for s in sources if s.is_type('ARMA')]}; "
                        f"CSV sources: {[s.name for s in sources if s.is_type('CSV')]}")

    mode = case.get_mode()

    if case.debug["enabled"]:
      # Debug mode runs a workflow with fixed capacities with lots of outputs to help determine if a simulation has
      # been correctly specified.
      self.template = DebugTemplate()
    elif has_static_history \
         and not has_synthetic_history \
         and mode == "sweep" \
         and not self._has_uncertain_econ_params(components) \
         and self._number_of_static_history_samples(case, sources) == 1:
      # Fixed history workflow: only run the dispatch optimization once per capacity configuration.
      # This is quite a restrictive case due to some limitations in how sampling and post-processing is done in
      # RAVEN.
      self.template = FlatMultiConfigTemplate()
    else:
      # Use the bilevel problem formulation. Outer workflow samples component capacities and uncertain economic
      # parameters. Inner workflow performs dispatch optimization over multiple synthetic histories.
      # This catches a large number of cases that are not easily done in a flat workflow in RAVEN:
      #   - Multiple histories and capacity configurations (PostProcessor won't calculate statistics for each capacity)
      #   - Any opt mode case (can't specify both a sampler and an optimizer in a MultiRun step)
      #   - Any case with uncertain economic parameters
      self.template = BilevelTemplate(mode, has_static_history, has_synthetic_history)

    # TODO: A case where all capacities are fixed could also be make to be a flat workflow. There is no demand for this
    #       type of workflow right now (to my knowledge, as of 2024-12-20), so this hasn't yet been implemented.

    self.template.loadTemplate(self.template.template_name, "xml")
    self.template.createWorkflow(case=case, components=components, sources=sources)

  def write_workflow(self, dest_dir: str, case: HeronCase, components: list[Component], sources: list[Source]) -> None:
    """
    Write the workflow to file
    @ In, dest_dir, str, directory to write to
    @ In, case, HeronCase, the HERON case object
    @ In, components, list[Component], case components
    @ In, sources, list[Source], case sources
    @ Out, None
    """
    print("========================")
    print("HERON: writing files ...")
    print("========================")
    dest = self.template.get_write_path(dest_dir)
    self.template.writeWorkflow(self.template.template_xml, dest)

    # Write library of info so it can be read in dispatch during inner run. Doing this here ensures that the lib file
    # is written just once, no matter the number of workflow files written by the template.
    lib_file = Path(dest_dir) / self.template.namingTemplates["lib file"]
    with lib_file.open("wb") as lib:
      pk.dump((case, components, sources), lib)
    print(f"Wrote '{lib_file.name}' to '{str(lib_file.resolve())}'")

  ###################
  # Utility methods #
  ###################
  @staticmethod
  def _has_all_capacities_fixed(components: list[Component]) -> bool:
    """
    Checks if all components have fixed capacities
    @ In, components, list[Component], list of HERON case components
    @ Out, has_all_capacities_fixed, bool, if all components have fixed capacities
    """
    return not any(comp.get_capacity(None, raw=True).is_parametric() for comp in components)

  @staticmethod
  def _has_uncertain_econ_params(components: list[Component]) -> bool:
    """
    Checks if any components have uncertain cashflow parameters which affect the dispatch. For this to be the case,
    a component must be dispatchable and must have an hourly recurring cashflow with an uncertain parameter.
    @ In, components, list[Component], HERON case components
    @ Out, has_uncertain_variable_costs, bool, if there are any uncertain variable costs for dispatchable componentss
    """
    for component in components:
      if len(component.get_uncertain_cashflow_params()) > 0:
        return True
    return False

  @staticmethod
  def _number_of_static_history_samples(case: HeronCase, sources: list[Source]) -> int:
    """
    Gets the maximum number of static history samples among CSV sources
    @ In, case, HeronCase, the case object
    @ In, source, list[Source], file sources
    @ Out, num_samples, int, number of static history samples
    """
    return max(
      map(
        lambda x: x.get_structure(case)["num_samples"],
        filter(lambda x: x.type == "CSV", sources)
      ),
    )

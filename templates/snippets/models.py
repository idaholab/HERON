# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Model snippets

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
import xml.etree.ElementTree as ET

from ..xml_utils import find_node
from ..decorators import listproperty
from .base import RavenSnippet


class RavenCode(RavenSnippet):
  """ Sets up a <Code> model to run inner RAVEN workflow. """
  tag = "Code"
  snippet_class = "Models"
  subtype = "RAVEN"

  def __init__(self, name: str | None = None) -> None:
    """
    Constructor
    @ In, name, str, optional, the model name
    @ Out, None
    """
    super().__init__(name)
    self._py_cmd = None

  def add_alias(self, name: str, suffix: str | None = None, loc: str | None = None) -> None:
    """
    Add an alias node to the model.
    @ In, name, str, the variable name to alias
    @ In, suffix, str, optional, a suffix to append to 'name'
    @ In, loc, str, optional, the location of the variable in the workflow
                              default: Samplers|MonteCarlo@name:mc_arma_dispatch
    @ Out, None
    """
    varname = name if not suffix else f"{name}_{suffix}"
    if not loc:
      loc = "Samplers|MonteCarlo@name:mc_arma_dispatch"  # where this is pointing 9/10 times
    alias_text = f"{loc}|constant@name:{varname}"
    alias = ET.SubElement(self, "alias", {"variable": varname, "type": "input"})
    alias.text = alias_text

  def set_inner_data_handling(self, dest: str, dest_type: str) -> None:
    """
    Set the inner-to-outer data handling source object.
    @ In, dest, str, the name of the data object
    @ In, dest_type, str, the type of object used to pass the data ("csv" or "netcdf")
    @ Out, None
    """
    if dest_type == "csv":
      remove_tag = "outputDatabase"
      keep_tag = "outputExportOutStreams"
    elif dest_type == "netcdf":
      remove_tag = "outputExportOutStreams"
      keep_tag = "outputDatabase"
    else:
      raise ValueError("Model output export destination must be a CSV <Print> OutStream or "
                       f"a NetCDF Database. Received: {type(dest)}")

    if (remove_node := self.find(remove_tag)) is not None:
      self.remove(remove_node)

    keep_node = self.find(keep_tag)
    if keep_node is None:
      keep_node = ET.SubElement(self, keep_tag)
    keep_node.text = dest

  @property
  def executable(self) -> str | None:
    """
    RAVEN executable path getter
    @ In, None
    @ Out, executable, str | None, the executable path
    """
    node = self.find("executable")
    return None if node is None else node.text

  @executable.setter
  def executable(self, value: str) -> None:
    """
    RAVEN executable path setter
    @ In, value, str, the executable path
    @ Out, None
    """
    find_node(self, "executable").text = value

  @property
  def python_command(self) -> str | None:
    """
    Python command getter
    @ In, None
    @ Out, py_cmd, str | None, the Python command
    """
    return self._py_cmd

  @python_command.setter
  def python_command(self, cmd: str) -> None:
    """
    Python command setter
    @ In, cmd, str, the python command
    @ Out, None
    """
    if self._py_cmd is None:
      ET.SubElement(self, "clargs", {"type": "prepend", "arg": cmd})
    else:
      node = self.find(f"clargs[@type='prepend' and @arg='{cmd}']")
      node.set("arg", cmd)
    self._py_cmd = cmd


class GaussianProcessRegressor(RavenSnippet):
  """ A Gaussian Process Regressor model snippet """
  tag = "ROM"
  snippet_class = "Models"
  subtype = "GaussianProcessRegressor"

  def __init__(self, name: str | None = None) -> None:
    """
    Constructor
    @ In, name, str, optional, snippet name
    @ Out, None
    """
    # FIXME: Only custom_kernel setting exposed to HERON input
    default_settings = {
      "alpha": 1e-8,
      "n_restarts_optimizer": 5,
      "normalize_y": True,
      "kernel": "Custom",
      "custom_kernel": "(Constant*Matern)",
      "anisotropic": True,
      "multioutput": False
    }
    super().__init__(name, default_settings)

  @listproperty
  def features(self) -> list[str]:
    """
    Features list getter
    @ In, None
    @ Out, features, list[str], features list
    """
    node = self.find("Features")
    return getattr(node, "text", [])

  @features.setter
  def features(self, value: list[str]) -> None:
    """
    Features list setter
    @ In, value, list[str], features list
    @ Out, None
    """
    find_node(self, "Features").text = value

  @listproperty
  def target(self) -> str:
    """
    Target getter
    @ In, None
    @ Out, target, str, target variables
    """
    node = self.find("Target")
    return getattr(node, "text", [])

  @target.setter
  def target(self, value: list[str]) -> None:
    """
    Target setter
    @ In, value, list[str], target list
    @ Out, None
    """
    find_node(self, "Target").text = value

  @property
  def custom_kernel(self) -> str:
    """
    Custom kernel getter
    @ In, None
    @ Out, custom_kernel, str, the custom kernel
    """
    node = self.find("custom_kernel")
    return node.text

  @custom_kernel.setter
  def custom_kernel(self, value: str) -> None:
    """
    Custom kernel setter
    @ Ine, value, str, the custom kernel
    @ Out, None
    """
    self.find("custom_kernel").text = value


class EnsembleModel(RavenSnippet):
  """ EnsembleModel snippet class """
  tag = "EnsembleModel"
  snippet_class = "Models"
  subtype = ""


class EconomicRatioPostProcessor(RavenSnippet):
  """ PostProcessor snippet for EconomicRatio postprocessors """
  tag = "PostProcessor"
  snippet_class = "Models"
  subtype = "EconomicRatio"

  def add_statistic(self, tag: str, prefix: str, variable: str, **kwargs) -> None:
    """
    Add a statistic to the postprocessor
    @ In, tag, str, the node tag (also the statistic name)
    @ In, prefix, str, the statistic prefix
    @ In, variable, str, the variable name
    @ In, kwargs, dict, keyword arguments for additional attributes
    @ Out, None
    """
    # NOTE: This allows duplicate nodes to be added. It's good to avoid that but won't cause anything to crash.
    ET.SubElement(self, tag, prefix=prefix, **kwargs).text = variable


class ExternalModel(RavenSnippet):
  """ ExternalModel snippet class """
  tag = "ExternalModel"
  snippet_class = "Models"
  subtype = ""

  @classmethod
  def from_xml(cls, node: ET.Element) -> "ExternalModel":
    """
    Create a snippet class from a XML node
    @ In, node, ET.Element, the XML node
    @ Out, model, ExternalModel, an external model snippet object
    """
    model = cls()
    model.attrib |= node.attrib
    for sub in node:
      if sub.tag == "variables":
        model.variables = [v.strip() for v in sub.text.split(",")]
      else:
        model.append(sub)
    return model

  @listproperty
  def variables(self) -> list[str]:
    """
    Variables list getter
    @ In, None
    @ Out, variables, list[str], the model variables
    """
    node = self.find("variables")
    return getattr(node, "text", [])

  @variables.setter
  def variables(self, value: list[str]) -> None:
    """
    Variables list getter
    @ In, value, list[str], the model variables
    @ Out, None
    """
    find_node(self, "variables").text = value


class HeronDispatchModel(ExternalModel):
  """ ExternalModel snippet for HERON dispatch manager models """
  snippet_class = "Models"
  subtype = "HERON.DispatchManager"


class PickledROM(RavenSnippet):
  """ Pickled ROM snippet class """
  tag = "ROM"
  snippet_class = "Models"
  subtype = "pickledROM"

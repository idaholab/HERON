# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Sampler snippets

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
from typing import Any
import xml.etree.ElementTree as ET

from ..xml_utils import find_node
from .base import RavenSnippet
from .distributions import Distribution


class SampledVariable(RavenSnippet):
  """ A snippet class for sampled variables """
  tag = "variable"

  def use_grid(self, kind: str, construction: str, values: list[float], steps: int | None = None) -> None:
    """
    Use a grid of values to sample the variable
    @ In, kind, str, the type of sampling to do
    @ In, construction, str, how to construct the values to sample
    @ In, values, list[float], values used by the constructor
    @ In, steps, int, optional, the number of steps to make along the interval defined by 'values'
    @ Out, None
    """
    attrib = {"type": kind, "construction": construction}
    if steps:
      attrib["steps"] = steps
    ET.SubElement(self, "grid", attrib).text = " ".join([str(v) for v in values])

  @property
  def initial(self) -> float | None:
    """
    Getter for initial value
    @ In, None
    @ Out, initial, float | None, the initial value
    """
    node = self.find("initial")
    return None if node is None else node.text

  @initial.setter
  def initial(self, value: float) -> None:
    """
    Setter for initial value
    @ In, value, float, the initial value
    @ Out, None
    """
    find_node(self, "initial").text = value

  @property
  def distribution(self) -> str | None:
    """
    Getter for distribution name
    @ In, None
    @ Out, distribution, str | None, the distribution name
    """
    node = self.find("distribution")
    return None if node is None else node.text

  @distribution.setter
  def distribution(self, value: Distribution | str) -> None:
    """
    Setter for distribution name
    @ In, value, Distribution | str, the distribution object or its name
    @ Out, None
    """
    find_node(self, "distribution").text = str(value)


class Sampler(RavenSnippet):
  """ Sampler snippet base class """
  snippet_class = "Samplers"

  @property
  def num_sampled_vars(self) -> int:
    """
    Getter for the number of variables sampled by the sampler
    @ In, None
    @ Out, num_sampled_vars, int, the number of sampled variables
    """
    return len(self.findall("variable"))

  @property
  def denoises(self) -> int | None:
    """
    Getter for denoises
    @ In, None
    @ Out, denoises, int | None, the number of denoises
    """
    node = self.find("constant[@name='denoises']")
    return None if node is None else node.text

  @denoises.setter
  def denoises(self, value: int) -> None:
    """
    Setter for denoises
    @ In, value, int, the number of denoises
    @ Out, None
    """
    find_node(self, "constant[@name='denoises']").text = value

  @property
  def init_seed(self) -> int | None:
    """
    Getter for sampler seed
    @ In, None
    @ Out, init_seed, int | None, the seed value
    """
    node = self.find("samplerInit/initialSeed")
    return None if node is None else node.text

  @init_seed.setter
  def init_seed(self, value: int) -> None:
    """
    Setter for sampler seed
    @ In, value, int, the seed value
    @ Out, None
    """
    find_node(self, "samplerInit/initialSeed").text = value

  @property
  def init_limit(self) -> int | None:
    """
    Getter for sampling limit
    @ In, None
    @ Out, init_limit, int | None, the sampling limit
    """
    node = self.find("samplerInit/limit")
    return None if node is None else node.text

  @init_limit.setter
  def init_limit(self, value: int) -> None:
    """
    Setter for sampling limit
    @ In, value, int, the sampling limit
    @ Out, None
    """
    find_node(self, "samplerInit/limit").text = value

  def add_variable(self, variable: SampledVariable) -> None:
    """
    Add a variable to sample to the sampler
    @ In, variable, SampledVariable, the variable to sample
    @ Out, None
    """
    self.append(variable)

  def add_constant(self, name: str, value: Any) -> None:
    """
    Add a constant to the sampler
    @ In, name, str, the name of the constant
    @ In, value, Any, the value of the constant
    @ Out, None
    """
    ET.SubElement(self, "constant", attrib={"name": name}).text = value

  def has_variable(self, variable: str | SampledVariable) -> bool:
    """
    Does the sampler sample a given variable?
    @ In, variable, str | SampledVariable, the variable to check for
    @ Out, var_found, bool, if the variable is in the sampler
    """
    var_name = variable if isinstance(variable, str) else variable.name
    var_found = self.find(f"variable[@name='{var_name}']") is not None
    return var_found

class Grid(Sampler):
  """ Grid sampler snippet class """
  tag = "Grid"

class MonteCarlo(Sampler):
  """ Monte Carlo sampler snippet class """
  tag = "MonteCarlo"

class Stratified(Sampler):
  """ Stratified sampler snippet class """
  tag = "Stratified"

  def __init__(self, name: str | None = None) -> None:
    """
    Constructor
    @ In, name, str, optional, entity name
    @ Out, None
    """
    super().__init__(name)
    # Must have samplerInit node
    ET.SubElement(self, "samplerInit")

class CustomSampler(Sampler):
  """ Custom sampler snippet class """
  tag = "CustomSampler"

class EnsembleForward(Sampler):
  """ Ensemble sampler snippet class """
  tag = "EnsembleForward"
  sampler_types = {samp_type.tag: samp_type for samp_type in [Grid, MonteCarlo, Stratified, CustomSampler]}

  @classmethod
  def from_xml(cls, node: ET.Element) -> "EnsembleForward":
    """
    Alternative constructor for creating an EnsembleForward instance from an XML node
    @ In, node, ET.Element, the XML node
    @ Out, sampler, EnsembleForward, the ensemble sampler snippet object
    """
    # The EnsembleForward sampler takes other samplers as subelements, so here we take care to create the
    # appropriate Sampler objects for these subnodes if they're of an implemented type.
    sampler = cls()
    sampler.attrib |= node.attrib
    for sub in node:
      if sub.tag in cls.sampler_types:
        sampler.append(cls.sampler_types[sub.tag].from_xml(sub))
      else:
        sampler.append(sub)
    return sampler

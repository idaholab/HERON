# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Define OutStreams for RAVEN workflows

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
import xml.etree.ElementTree as ET

from ..xml_utils import find_node
from ..decorators import listproperty
from .base import RavenSnippet
from .dataobjects import DataObject


class OutStream(RavenSnippet):
  """ Base class for OutStream entities """
  snippet_class = "OutStreams"

  @property
  def source(self) -> str | None:
    """
    Source getter
    @ In, None
    @ Out, source, str | None, the data source
    """
    node = self.find("source")
    return None if node is None or not node.text else str(node.text)

  @source.setter
  def source(self, value: str | DataObject) -> None:
    """
    Source setter
    @ In, value, str | DataObject, the source
    @ Out, None
    """
    find_node(self, "source").text = value


class PrintOutStream(OutStream):
  """ OutStream snippet for Print OutStreams """
  tag = "Print"

  def __init__(self, name: str | None = None) -> None:
    """
    Constructor
    @ In, name, str, optional, entity name
    @ Out None
    """
    super().__init__(name)
    ET.SubElement(self, "type").text = "csv"  # The only supported output file type


class OptPathPlot(OutStream):
  """ OutStream snippet for OptPath plots """
  tag = "Plot"
  subtype = "OptPath"

  @listproperty
  def variables(self) -> list[str]:
    """
    Variables list getter
    @ In, None
    @ Out, variables, list[str], variables list
    """
    node = self.find("vars")
    return getattr(node, "text", [])

  @variables.setter
  def variables(self, value: list[str]) -> None:
    """
    Variables list setter
    @ In, value, list[str], variables list
    @ Out, None
    """
    find_node(self, "vars").text = value


class HeronDispatchPlot(OutStream):
  """ OutStream snippet for HERON dispatch plots """
  tag = "Plot"
  subtype = "HERON.DispatchPlot"

  @classmethod
  def from_xml(cls, node: ET.Element) -> "HeronDispatchPlot":
    """
    Alternate construction from XML node
    @ In, node, ET.Element, the XML node
    @ Out, plot, HeronDispatchPlot, the corresponding snippet object
    """
    plot = cls()
    plot.attrib |= node.attrib
    for sub in node:
      if sub.tag == "signals":
        plot.signals = [s.strip() for s in sub.text.split(",")]
      else:
        plot.append(sub)
    return plot

  @property
  def macro_variable(self) -> str | None:
    """
    Getter for macro variable node
    @ In, None
    @ Out, macro_variable, str | None, macro variable name
    """
    node = self.find("macro_variable")
    return None if node is None else node.text

  @macro_variable.setter
  def macro_variable(self, value: str) -> None:
    """
    Setter for macro variable node
    @ In, value, str, macro variable name
    @ Out, None
    """
    find_node(self, "macro_variable").text = value

  @property
  def micro_variable(self) -> str | None:
    """
    Getter for micro variable node
    @ In, None
    @ Out, micro_variable, str | None, micro variable name
    """
    node = self.find("micro_variable")
    return None if node is None else node.text

  @micro_variable.setter
  def micro_variable(self, value: str) -> None:
    """
    Setter for micro variable node
    @ In, value, str, micro variable name
    @ Out, None
    """
    find_node(self, "micro_variable").text = value

  @listproperty
  def signals(self) -> list[str]:
    """
    Getter for signals node
    @ In, None
    @ Out, signals, str | None, signals variable names
    """
    node = self.find("signals")
    return getattr(node, "text", [])

  @signals.setter
  def signals(self, value: list[str]) -> None:
    """
    Setter for signals node
    @ In, value, str, signals variable names
    @ Out, None
    """
    find_node(self, "signals").text = value


class TealCashFlowPlot(OutStream):
  """ OutStream snippet for TEAL cashflow plots """
  tag = "Plot"
  subtype = "TEAL.CashFlowPlot"

# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Snippet class for the RunInfo block

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
import re
import socket
from pathlib import Path
import xml.etree.ElementTree as ET

from .base import RavenSnippet
from .steps import Step
from ..xml_utils import find_node
from ..decorators import listproperty


class RunInfo(RavenSnippet):
  """ Snippet class for the RAVEN XML RunInfo block """
  tag = "RunInfo"

  def set_parallel_run_settings(self, parallel_run_info: dict[str, str]) -> None:
    """
    Set how to run in parallel
    @ In, parallel_run_info, dict[str, str], settings for parallel execution
    @ Out, None
    """
    # Get special pre-sets for known computing environments
    try:
      hostname = socket.gethostbyaddr(socket.gethostname())[0]
    except socket.gaierror:
      hostname = "unknown"
    parallel_xml = get_parallel_xml(hostname)
    self._apply_parallel_xml(parallel_xml)

    # Handle "memory" setting first since its parent node is the "mode" node
    if memory_val := parallel_run_info.pop("memory", None):
      find_node(self, "mode/memory").text = memory_val
    # All other settings get tacked onto the main RunInfo block
    for tag, value in parallel_run_info.items():
      find_node(self, tag).text = value

  def _apply_parallel_xml(self, parallel_xml: ET.Element) -> None:
    """
    Apply the parallel processing settings in parallel_xml to the snippet XML tree
    @ In, parallel_xml, ET.Element, the XML tree with parallel settings
    @ Out, None
    """
    for child in parallel_xml.find("useParallel"):
      self.append(child)

    # Add parallel method settings from parallel_xml, if present
    outer_node = parallel_xml.find("outer")
    if outer_node is None:
      self.use_internal_parallel = True
    else:
      for child in outer_node:
        self.append(child)

  @property
  def job_name(self) -> str | None:
    """
    Getter for job name
    @ In, None
    @ Out, job_name, str | None, the job name
    """
    node = self.find("JobName")
    return getattr(node, "text", None)

  @job_name.setter
  def job_name(self, value: str) -> None:
    """
    Setter for job name
    @ In, value, str, the job name
    @ Out, None
    """
    find_node(self, "JobName").text = str(value)

  @property
  def working_dir(self) -> str | None:
    """
    Getter for working directory
    @ In, None
    @ Out, working_dir, str | None, the working directory
    """
    node = self.find("WorkingDir")
    return None if node is None or node.text is None else str(node.text)

  @working_dir.setter
  def working_dir(self, value: str) -> None:
    """
    Setter for working directory
    @ In, value, str, the working directory
    @ Out, None
    """
    find_node(self, "WorkingDir").text = str(value)

  @property
  def batch_size(self) -> int:
    """
    Getter for batch size
    @ In, None
    @ Out, batch_size, int | None, the batch size
    """
    node = self.find("batchSize")
    return None if node is None else int(getattr(node, "text", 1))

  @batch_size.setter
  def batch_size(self, value: int) -> None:
    """
    Setter for batch size
    @ In, value, int, the batch size
    @ Out, None
    """
    find_node(self, "batchSize").text = int(value)

  @property
  def use_internal_parallel(self) -> bool:
    """
    Getter for internal parallel flag
    @ In, None
    @ Out, internal_parallel, bool, the internal parallel flag
    """
    node = self.find("internalParallel")
    return False if node is None else bool(getattr(node, "text", False))

  @use_internal_parallel.setter
  def use_internal_parallel(self, value: bool) -> None:
    """
    Setter for internal parallel flag
    @ In, value, bool, the internal parallel flag
    @ Out, None
    """
    # Set node text if True, remove node if False
    node = find_node(self, "internalParallel")
    if value:
      node.text = value
    else:
      self.remove(node)

  @property
  def num_mpi(self) -> int | None:
    """
    Getter for number of MPI processes
    @ In, None
    @ Out, num_mpi, int | None, the number of processes
    """
    node = self.find("NumMPI")
    return None if node is None else int(getattr(node, "text", 1))

  @num_mpi.setter
  def num_mpi(self, value: int) -> None:
    """
    Setter for number of MPI processes
    @ In, value, int, the number of MPI processes
    @ Out, None
    """
    find_node(self, "NumMPI").text = int(value)

  @listproperty
  def sequence(self) -> list[str]:
    """
    Getter for the step sequence
    @ In, None
    @ Out, sequence, list[str], list of steps
    """
    node = self.find("Sequence")
    return getattr(node, "text", [])

  @sequence.setter
  def sequence(self, value: list[str | Step]) -> None:
    """
    Setter for the step sequence
    @ In, value, str, the step sequence
    @ Out, None
    """
    find_node(self, "Sequence").text = [str(v).strip() for v in value]

#####################
# UTILITY FUNCTIONS #
#####################

def get_default_parallel_settings() -> ET.Element:
  """
  The default parallelization settings. Used when the hostname doesn't match any parallel settings
  XMLs found in HERON/templates/parallel.
  @ In, None
  @ Out, parallel, ET.Element, the default parallel settings
  """
  parallel = ET.Element("parallel")
  use_parallel = ET.SubElement(parallel, "useParallel")
  mode = ET.SubElement(use_parallel, "mode")
  mode.text = "mpi"
  mode.append(ET.Element("runQSUB"))
  return parallel


def get_parallel_xml(hostname: str) -> ET.Element:
  """
  Finds the xml file to go with the given hostname.
  @ In, hostname, string with the hostname to search for
  @ Out, xml, xml.eTree.ElementTree, if an xml file is found then use it, otherwise return the default settings
  """
  # Should this allow loading from another directory (such as one next to the input file?)
  path = Path(__file__).parent.parent / "parallel"
  for filename in path.glob("*.xml"):
    cur_xml = ET.parse(filename).getroot()
    regexp = cur_xml.attrib["hostregexp"]
    print(f"Checking {filename} regexp {regexp} for hostname {hostname}")
    if re.match(regexp, hostname):
      print("Success!")
      return cur_xml
  return get_default_parallel_settings()

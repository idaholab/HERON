#!/usr/bin/env python
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Runs HERON.
"""
import os
from queue import Empty
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import HERON.src._utils as hutils
try:
  import ravenframework
except ModuleNotFoundError:
  sys.path.append(hutils.get_raven_loc())

from HERON.src import input_loader
from HERON.src.base import Base
from HERON.src.Moped import MOPED
from HERON.src.Herd import HERD
from HERON.src.NetworkPlot import NetworkPlot

from ravenframework.MessageHandler import MessageHandler


class HERON(Base):
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    Base.__init__(self)
    self._components = []   # units involved in this study
    self._sources = []      # sources involved in this study (arma, functions, static histories, etc)
    self._case = None       # information about the case we're running
    self._input_dir = None  # location of the input XML file
    self._input_name = None # name of the input XML file

    messageHandler = MessageHandler()
    messageHandler.initialize({'verbosity': 'all',
                               'callerLength': 18,
                               'tagLength': 7,
                               'suppressErrs': False,})
    self.messageHandler = messageHandler

  def read_input(self, name: str) -> None:
    """
      Loads data from input
      @ In, name, str, name of file to read from
      @ Out, None
    """
    location, fname = os.path.split(name)
    self._input_dir = location
    self._input_name = fname
    inp = input_loader.load(name)
    objects = input_loader.parse(inp, location, self.messageHandler)
    self._components = objects['components']
    self._sources = objects['sources']
    self._case = objects['case']

  def __repr__(self):
    """
      String representation of object.
      @ In, None
      @ Out, repr, str, string rep
    """
    return '<HERON Simulation>'

  def print_me(self, tabs=0, tab='  ') -> None:
    """
      Prints info about self.
      @ In, tabs, int, number of tabs to insert
      @ In, tab, str, how tab should be written
      @ Out, None
    """
    pre = tab*tabs
    self.raiseADebug("==========================")
    self.raiseADebug("Printing simulation state:")
    self.raiseADebug("==========================")
    assert self._case is not None
    self._case.print_me(tabs=tabs+1, tab=tab)
    for comp in self._components:
      comp.print_me(tabs=tabs+1, tab=tab)
    for source in self._sources:
      source.print_me(tabs=tabs+1, tab=tab)

  def plot_resource_graph(self) -> None:
    """
      Plots the resource graph of the HERON simulation using components
      from the input file.

      @ In, None
      @ Out, None
    """
    if self._case.debug['enabled']:  # TODO do this every time?
      graph = NetworkPlot(self._components)
      img_path = os.path.join(self._input_dir, 'network.png')
      graph.save(img_path)

  def create_raven_workflow(self, case=None):
    """
      Loads, modifies, and writes a RAVEN template workflow based on the Case.
      @ In, case, Cases.Case, optional, case to run (defaults to self._case)
      @ Out, None
    """
    if case is None:
      case = self._case
    # let the case do the work
    assert case is not None
    case.write_workflows(self._components, self._sources, self._input_dir)

  def run_moped_workflow(self, case=None, components=None, sources=None):
    """
      Runs MOPED workflow for generating pyomo problem and solves it
      @ In, case, HERON case object with necessary run settings
      @ Out, None
    """
    if case is None:
      case = self._case
    if components is None:
      components = self._components
    if sources is None:
      sources = self._sources
    assert case is not None and components is not None and sources is not None
    moped = MOPED()
    self.raiseAMessage("***** You are running Monolithic Optimizer for Probabilistic Economic Dispatch (MOPED) *****")
    moped.setInitialParams(case, components, sources)
    moped.run()

  def run_dispatches_workflow(self):
    """
      Runs DISPATCHES workflow for creating framework and running with IDAES
      @ In, None
      @ Out, None
    """
    # checking to see if DISPATCHES is properly installed
    try:
      import dispatches.case_studies as tmp_lib
      del tmp_lib
    except ModuleNotFoundError as mnferr:
      raise IOError('DISPATCHES has not been found in current conda environment.' +
                    'Please re-install the conda environment from RAVEN using the ' +
                    '--optional flag.') from mnferr
    case = self._case
    components = self._components
    sources = self._sources
    assert case is not None and components is not None and sources is not None
    herd = HERD()
    print("*******************************************************************************")
    print("HERON is Running DISPATCHES")
    print("*******************************************************************************")
    herd.setInitialParams(case, components, sources)
    herd.run()

def main():
  """
    Runs HERON input from command line arguments
    @ In, None
    @ Out, None
  """
  parser = argparse.ArgumentParser(description='Holistic Energy Resource Optimization Network (HERON)')
  parser.add_argument('xml_input_file', nargs='?', default="", help='HERON XML input file')
  parser.add_argument('--definition', action="store_true", dest="definition", help='HERON input file definition compatible with the NEAMS Workbench')
  args = parser.parse_args()

  sim = HERON()

  # User requested the input definition be printed
  if args.definition:
    from HERON.src import input_definition
    input_definition.print_input_definition()
    sys.exit(0)
  if args.xml_input_file == "":
    parser.error("the following arguments are required: xml_input_file")

  sim.read_input(args.xml_input_file) # TODO expand to use arguments?
  # print details
  sim.print_me()
  sim.plot_resource_graph()

  if sim._case._workflow == 'standard':
    sim.create_raven_workflow()
  elif sim._case._workflow == 'MOPED':
    sim.run_moped_workflow()
  elif sim._case._workflow == 'DISPATCHES':
    sim.run_dispatches_workflow()

if __name__ == '__main__':
  main()


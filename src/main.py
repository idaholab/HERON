#!/usr/bin/env python
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Runs HERON.
"""
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import HERON.src._utils as hutils
sys.path.append(hutils.get_raven_loc())

from HERON.src import input_loader
from HERON.src.base import Base
from HERON.src import Moped

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

  def read_input(self, name):
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

  def print_me(self, tabs=0, tab='  '):
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
    moped = Moped.MOPED()
    self.raiseAMessage("***** You are running Monolithic Optimizer for Probabilistic Economic Dispatch (MOPED) *****")
    moped.setInitialParams(case, components, sources)
    moped.run()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Holistic Energy Resource Optimization Network (HERON)')
  parser.add_argument('xml_input_file', help='HERON XML input file')
  args = parser.parse_args()
  sim = HERON()
  sim.read_input(args.xml_input_file) # TODO expand to use arguments?
  # print details
  sim.print_me()
  if sim._case._workflow == 'standard':
    sim.create_raven_workflow()
  elif sim._case._workflow == 'MOPED':
    sim.run_moped_workflow()
  # TODO someday? sim.run()


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

from HERON.src import input_loader
from HERON.src import install_skills
from HERON.src import inspect_data
from HERON.src.base import Base
from HERON.src.Moped import MOPED
from HERON.src.Herd import HERD
from HERON.src.NetworkPlot import NetworkPlot
from HERON.src.messages import HeronMessageHandler


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

    self.messageHandler = HeronMessageHandler(verbosity='all', caller_length=18, tag_length=7)

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

  def create_raven_workflow(self, case=None, python_cmd_raven=None):
    """
      Loads, modifies, and writes a RAVEN template workflow based on the Case.
      @ In, case, Cases.Case, optional, case to run (defaults to self._case)
      @ In, python_cmd_raven, string, optional, custom python command to use for running RAVEN
          Driving use case is specifying that RAVEN be run with code coverage
      @ Out, None
    """
    if case is None:
      case = self._case
    # let the case do the work
    assert case is not None
    if python_cmd_raven:
      case.set_py_cmd_for_raven(python_cmd_raven)
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

# subcommands; any other first argument is treated as "run" so "heron input.xml" keeps working
SUBCOMMANDS = ('run', 'inspect', 'install-skills')

def run_case(args):
  """
    Runs a HERON input file (the "run" subcommand, and the default when no subcommand is given)
    @ In, args, argparse.Namespace, parsed command line arguments
    @ Out, None
  """
  # User requested the input definition be printed
  if args.definition:
    from HERON.src import input_definition
    input_definition.print_input_definition()
    sys.exit(0)
  if args.xml_input_file == "":
    args.parser.error("the following arguments are required: xml_input_file")

  sim = HERON()
  sim.read_input(args.xml_input_file) # TODO expand to use arguments?
  # print details
  sim.print_me()
  sim.plot_resource_graph()

  if sim._case._workflow == 'standard':
    if args.python_cmd_raven is not None:
      print("creating workflow")
      sim.create_raven_workflow(python_cmd_raven=args.python_cmd_raven)
    else:
      sim.create_raven_workflow()
  elif sim._case._workflow == 'MOPED':
    sim.run_moped_workflow()
  elif sim._case._workflow == 'DISPATCHES':
    sim.run_dispatches_workflow()

def run_inspect(args):
  """
    Reports the time structure of a time series data source (the "inspect" subcommand)
    @ In, args, argparse.Namespace, parsed command line arguments
    @ Out, None
  """
  sim = HERON()
  try:
    info = inspect_data.inspect_data(args.data_file, time_var=args.time_var, year_var=args.year_var)
  except (FileNotFoundError, KeyError, ValueError) as err:
    sim.raiseAnError(type(err), str(err))
  sim.raiseAMessage('\n' + inspect_data.format_report(info))

def run_install_skills(args):
  """
    Installs the HERON agent skills into a directory (the "install-skills" subcommand)
    @ In, args, argparse.Namespace, parsed command line arguments
    @ Out, None
  """
  sim = HERON()
  try:
    installed = install_skills.install_skills(args.skills_dir)
  except FileNotFoundError as err:
    sim.raiseAnError(FileNotFoundError, str(err))
  for path in installed:
    sim.raiseAMessage(f'Installed skill: {path}')

def build_parser():
  """
    Builds the command line parser, with one subparser per subcommand
    @ In, None
    @ Out, parser, argparse.ArgumentParser, command line parser
  """
  parser = argparse.ArgumentParser(prog='heron',
                                   description='Holistic Energy Resource Optimization Network (HERON)',
                                   epilog='"heron <input.xml>" is shorthand for "heron run <input.xml>".')
  subparsers = parser.add_subparsers(dest='command', metavar='command')

  run = subparsers.add_parser('run', help='Run a HERON input file (default)',
                              description='Run a HERON XML input file, writing the RAVEN workflows it defines')
  run.add_argument('xml_input_file', nargs='?', default="", help='HERON XML input file')
  run.add_argument('--definition', action="store_true", dest="definition", help='HERON input file definition compatible with the NEAMS Workbench')
  run.add_argument('--python_command_for_raven', dest='python_cmd_raven', help='Custom python command for running RAVEN inner')
  run.set_defaults(func=run_case, parser=run)

  inspect = subparsers.add_parser('inspect', help='Report the time structure of a time series ROM or CSV',
                                  description='Report the time structure of a synthetic history ROM (.pk) or CSV, and the matching <time_discretization>')
  inspect.add_argument('data_file', metavar='FILE', help='Serialized synthetic history ROM (.pk) or CSV file')
  inspect.add_argument('--time-variable', dest='time_var', metavar='NAME', help='CSV only: name of the time column')
  inspect.add_argument('--year-variable', dest='year_var', metavar='NAME', help='CSV only: name of the year column')
  inspect.set_defaults(func=run_inspect)

  skills = subparsers.add_parser('install-skills', help='Install the HERON agent skills into a directory',
                                 description='Install the HERON agent skills into a directory')
  skills.add_argument('skills_dir', metavar='DIR', help='Directory to install the skills into (e.g. ~/.claude/skills)')
  skills.set_defaults(func=run_install_skills)
  return parser

def main(argv=None):
  """
    Runs HERON from command line arguments
    @ In, argv, list(str), optional, command line arguments (defaults to sys.argv[1:])
    @ Out, None
  """
  if argv is None:
    argv = sys.argv[1:]
  # no subcommand given (e.g. "heron input.xml", "heron --definition"): run, as HERON always has
  if not argv or (argv[0] not in SUBCOMMANDS and argv[0] not in ('-h', '--help')):
    argv = ['run'] + list(argv)
  args = build_parser().parse_args(argv)
  args.func(args)

if __name__ == '__main__':
  main()

# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the abce entity.
"""
from __future__ import unicode_literals, print_function
import os
import sys
import importlib

import numpy as np

from HERON.src.base import Base

from HERON.src.dispatch.Factory import known as known_dispatchers
from HERON.src.dispatch.Factory import get_class as get_dispatcher

from HERON.src.ValuedParams import factory as vp_factory
from HERON.src.ValuedParamHandler import ValuedParamHandler

from HERON.src.validators.Factory import known as known_validators
from HERON.src.validators.Factory import get_class as get_validator

import HERON.src._utils as hutils
try:
  import ravenframework
except ModuleNotFoundError:
  framework_path = hutils.get_raven_loc()
  sys.path.append(framework_path)
from ravenframework.utils import InputData, InputTypes


class ABCE(Base):
  """
    Produces something, often as the cost of something else
    TODO this case is for "sweep-opt", need to make a superclass for generic
  """

  def __init__(self, run_dir, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Base.__init__(self, **kwargs)
    self.name = None                   # case name
    self._mode = None                  # extrema to find: opt, sweep
    self._metric = 'NPV'               # TODO: future work - economic metric to focus on: lcoe, profit, cost
    self.run_dir = run_dir             # location of HERON input file
    self._verbosity = 'all'            # default verbosity for RAVEN inner/outer

    self.dispatch_name = None          # name of dispatcher to use
    self.dispatcher = None             # type of dispatcher to use
    self.validator_name = None         # name of dispatch validation to use
    self.validator = None              # type of dispatch validation to use
    self.dispatch_vars = {}            # non-component optimization ValuedParams

    self.useParallel = False           # parallel tag specified?
    self.outerParallel = 0             # number of outer parallel runs to use
    self.innerParallel = 0             # number of inner parallel runs to use

    self._diff_study = None            # is this only a differential study?
    self._num_samples = 1              # number of ARMA stochastic samples to use ("denoises")
    self._hist_interval = None         # time step interval, time between production points
    self._hist_len = None              # total history length, in same units as _hist_interval
    self._num_hist = None              # number of history steps, hist_len / hist_interval
    self._global_econ = {}             # global economics settings, as a pass-through
    self._increments = {}              # stepwise increments for resource balancing
    self._time_varname = 'time'        # name of the time-variable throughout simulation
    self._year_varname = 'Year'        # name of the year-variable throughout simulation
    self._labels = {}                  # extra information pertaining to current case
    self.debug = {                     # debug options, as enabled by the user (defaults included)
        'enabled': False,              # whether to enable debug mode
        'inner_samples': 1,            # how many inner realizations to sample
        'macro_steps': 1,              # how many "years" for inner realizations
        'dispatch_plot': True          # whether to output a plot in debug mode
    }

    self.data_handling = {             # data handling options
      'inner_to_outer': 'netcdf',      # how to pass inner data to outer (csv, netcdf)
    }

    self._time_discretization = None   # (start, end, number) for constructing time discretization, same as argument to np.linspace
    self._Resample_T = None            # user-set increments for resources
    self._optimization_settings = None # optimization settings dictionary for outer optimization loop
    self._workflow = 'standard' # setting for how to run HERON, default is through raven workflow
    self._result_statistics = {        # desired result statistics (keys) dictionary with attributes (values)
        'sigma': None,                 # user can specify additional result statistics
        'expectedValue': None,
        'median': None}

    # clean up location
    self.run_dir = os.path.abspath(os.path.expanduser(self.run_dir))

  #### API ####
  def write_workflows(self, case, components, sources):
    """
      Writes workflows for this case to XMLs on disk.
      @ In, components, HERON components, components for the simulation
      @ In, sources, HERON sources, sources for the simulation
      @ In, loc, str, location in which to write files
      @ Out, None
    """
    # load templates
    template_class = self._load_template()
    inner, outer = template_class.createWorkflow(case, components, sources)

    template_class.writeWorkflow((inner, outer), loc)

  #### UTILITIES ####
  def _load_template(self):
    """
      Loads template files for modification
      @ In, None
      @ Out, template_class, RAVEN Template, instantiated Template class
    """
    src_dir = os.path.dirname(os.path.realpath(__file__))
    heron_dir = os.path.abspath(os.path.join(src_dir, '..'))
    template_dir = os.path.abspath(os.path.join(heron_dir, 'templates'))
    template_name = 'abce_driver'
    # import template module
    sys.path.append(heron_dir)
    module = importlib.import_module(f'templates.{template_name}', package="HERON")
    # load template, perform actions
    template_class = module.TemplateAbce(messageHandler=self.messageHandler)
    template_class.loadTemplate(template_dir)
    return template_class

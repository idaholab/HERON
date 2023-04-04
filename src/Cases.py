
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the Cases entity.
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

class Case(Base):
  """
    Produces something, often as the cost of something else
    TODO this case is for "sweep-opt", need to make a superclass for generic
  """

  # metrics that can be used for objective in optimization or returned with results
  # each metric contains a dictionary with the following keys:
  # 'prefix' - printed result name
  # 'optimization_default' - 'min' or 'max' for optimization
  # 'percent' (only for percentile) - list of percentiles to return
  # 'threshold' (only for sortinoRatio, gainLossRatio, expectedShortfall, valueAtRisk) - threshold value for calculation
  metrics_mapping = {'expectedValue': {'prefix': 'mean', 'optimization_default': 'max'},
                     'minimum': {'prefix': 'min', 'optimization_default': 'max'},
                     'maximum': {'prefix': 'max', 'optimization_default': 'max'},
                     'median': {'prefix': 'med', 'optimization_default': 'max'},
                     'variance': {'prefix': 'var', 'optimization_default': 'min'},
                     'sigma': {'prefix': 'std', 'optimization_default': 'min'},
                     'percentile': {'prefix': 'perc', 'optimization_default': 'max', 'percent': ['5', '95']},
                     'variationCoefficient': {'prefix': 'varCoeff', 'optimization_default': 'min'},
                     'skewness': {'prefix': 'skew', 'optimization_default': 'min'},
                     'kurtosis': {'prefix': 'kurt', 'optimization_default': 'min'},
                     'samples': {'prefix': 'samp'},
                     'sharpeRatio': {'prefix': 'sharpe', 'optimization_default': 'max'},
                     'sortinoRatio': {'prefix': 'sortino', 'optimization_default': 'max', 'threshold': 'median'},
                     'gainLossRatio': {'prefix': 'glr', 'optimization_default': 'max', 'threshold': 'median'},
                     'expectedShortfall': {'prefix': 'es', 'optimization_default': 'min', 'threshold': ['0.05']},
                     'valueAtRisk': {'prefix': 'VaR', 'optimization_default': 'min', 'threshold': ['0.05']}}

  #### INITIALIZATION ####
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    input_specs = InputData.parameterInputFactory('Case', ordered=False, baseNode=None,
                                                  descr=r"""The \xmlNode{Case} node contains the general physics and
                                                  economics information required for a HERON workflow to be created
                                                  and solved.""")
    input_specs.addParam('name', param_type=InputTypes.StringType, required=True,
                         descr=r"""the name by which this analysis should be referred within HERON.""")

    # Optional Identifier Nodes
    label_specs = InputData.parameterInputFactory(name='label', ordered=False,
                                                  descr=r"""provides static label information to the model;
                                                  unused in computation. These data will be passed along through
                                                  the meta class and output in the simulation result files.
                                                  These data can also be accessed within user-defined transfer
                                                  functions by using \texttt{meta['HERON']['Case'].get_labels()}.""")
    label_specs.addParam(name='name',param_type=InputTypes.StringType,
                         descr=r"""the generalized name of the identifier.
                         Example: ``$<$label name="state"$>$Idaho$<$/label$>$''""")
    input_specs.addSub(label_specs)

    mode_options = InputTypes.makeEnumType('ModeOptions', 'ModeOptionsType', ['opt', 'sweep'])
    desc_mode_options = r"""determines the mode of operation for the outer/inner RAVEN.
                        If ``sweep'' then parametrically sweep over distributed values.
                        If ``opt'' then search distributed values for economic metric optima.
                        """
    input_specs.addSub(InputData.parameterInputFactory('mode', contentType=mode_options, strictMode=True,
                                                       descr=desc_mode_options))

    verbosity_options = InputTypes.makeEnumType('VerbosityOptions', 'VerbosityOptionsType',
                                                ['silent', 'quiet', 'all', 'debug'])
    desc_verbosity_options = r"""determines the level of verbosity for the outer/inner RAVEN runs. \default{all}.
                             If ``silent'' only errors are displayed.
                             If ``quiet'' errors and warnings are displayed.
                             If ``all'' errors, warnings, and messages are displayed.
                             If ``debug'' errors, warnings, messages, and debug messages are displayed."""
    input_specs.addSub(InputData.parameterInputFactory('verbosity', contentType=verbosity_options,
                                                       strictMode=True, descr=desc_verbosity_options))

    workflow_options = InputTypes.makeEnumType('WorkflowOptions', 'WorkflowOptionsType',
                                               ['standard', 'MOPED', 'combined', 'DISPATCHES'])

    desc_workflow_options = r"""determines the desired workflow(s) for the HERON analysis. \default{standard}.
                            If ``standard'' runs HERON as usual (writes outer/inner for RAVEN workflow).
                            If ``MOPED'' runs monolithic solver MOPED using the information in xml input.
                            If ``combined'' runs both workflows, setting up RAVEN workflow and solving with MOPED.
                            See Workflow Options section in user guide for more details"""
    input_specs.addSub(InputData.parameterInputFactory('workflow', contentType=workflow_options,
                                                       strictMode=True, descr=desc_workflow_options))

    # not yet implemented TODO
    #econ_metrics = InputTypes.makeEnumType('EconMetrics', 'EconMetricsTypes', ['NPV', 'lcoe'])
    #desc_econ_metrics = r"""indicates the economic metric that should be used for the HERON analysis. For most cases, this
    #                    should be NPV."""
    # input_specs.addSub(InputData.parameterInputFactory('metric', contentType=econ_metrics, descr=desc_econ_metrics))
    # input_specs.addSub(InputData.parameterInputFactory('differential', contentType=InputTypes.BoolType, strictMode=True,
    # descr=r"""(not implemented) allows differentiation between two HERON runs as a desired
    # economic metric."""

    # debug mode, for checking dispatch and etc
    debug = InputData.parameterInputFactory('debug', descr=r"""Including this node enables a reduced-size
        run with increased outputs for checking how the sampling, dispatching, and cashflow mechanics
        are working for a particular input. Various options for modifying how the debug mode operates
        are included for convenience; however, just including this node will result in a minimal run.""")
    debug.addSub(InputData.parameterInputFactory('inner_samples', contentType=InputTypes.IntegerType,
        descr=r"""sets the number of inner realizations of the stochastic synthetic histories and dispatch
              optimization to run per outer sample. Overrides the \xmlNode{num_arma_steps} option while
              \xmlNode{debug} mode is enabled. \default{1}"""))
    debug.addSub(InputData.parameterInputFactory('macro_steps', contentType=InputTypes.IntegerType,
        descr=r"""sets the number of macro steps (e.g. years) the stochastic synthetic histories and dispatch
              optimization should include. \default{1}"""))
    debug.addSub(InputData.parameterInputFactory('dispatch_plot', contentType=InputTypes.BoolType,
        descr=r"""provides a dispatch plot after running through \xmlNode{inner_samples} and
              \xmlNode{macro_steps} provided. To prevent plotting output during debug mode set to "False".
              \default{True}"""))
    debug.addSub(InputData.parameterInputFactory('cashflow_plot', contentType=InputTypes.BoolType,
        descr=r"""provides a cashflow plot after running through \xmlNode{inner_samples} and
              \xmlNode{macro_steps} provided. To prevent plotting output during debug mode set to "False".
              \default{True}"""))
    input_specs.addSub(debug)

    parallel = InputData.parameterInputFactory('parallel', descr=r"""Describes how to parallelize this run. If not present defaults to no parallelization (1 outer, 1 inner)""")
    parallel.addSub(InputData.parameterInputFactory('outer', contentType=InputTypes.IntegerType,
        descr=r"""the number of parallel runs to use for the outer optimization run. The product of this
              number and \xmlNode{inner} should be at most the number of parallel process available on
              your computing device. This should also be at most the number of samples needed per outer iteration;
              for example, with 3 opt bound variables and using finite differencing, at most 4 parallel outer runs
              can be used. \default{number of variable sweeps + 1}"""))
    parallel.addSub(InputData.parameterInputFactory('inner', contentType=InputTypes.IntegerType,
        descr=r"""the number of parallel runs to use per inner sampling run. This should be at most the number
              of denoising samples, and at most the number of parallel processes available on your computing
              device. \default{number of denoising samples}"""))
    #XXX RAVEN should be providing this InputData
    runinfo = InputData.parameterInputFactory('runinfo',
                descr=r"""this is copied into the RAVEN runinfo block, and defaults are specified in RAVEN""")
    runinfo.addSub(InputData.parameterInputFactory('expectedTime', contentType=InputTypes.StringType,
                  descr=r"""the expected time for the run to take in hours, minutes, seconds (example 24:00:00 for 1 day) """))
    runinfo.addSub(InputData.parameterInputFactory('clusterParameters', contentType=InputTypes.StringType,
                  descr=r"""Extra parameters needed by the cluster qsub command"""))
    runinfo.addSub(InputData.parameterInputFactory('RemoteRunCommand', contentType=InputTypes.StringType,
                   descr=r"""The shell command used to run remote commands"""))
    runinfo.addSub(InputData.parameterInputFactory('memory', contentType=InputTypes.StringType,descr=r"""The amount of memory needed per core (example 4gb)"""))
    parallel.addSub(runinfo)
    # TODO HPC?
    input_specs.addSub(parallel)

    data_handling = InputData.parameterInputFactory('data_handling', descr=r"""Provides options for data handling within HERON operations.""")
    inner_outer_data = InputTypes.makeEnumType('InnerOuterData', 'InnerOuterDataType', ['csv', 'netcdf'])
    data_handling.addSub(InputData.parameterInputFactory('inner_to_outer', contentType=inner_outer_data,
        descr=r"""which type of data format to transfer results from inner (stochastic dispatch optimization) runs to
                  the outer (capacity and meta-variable optimization) run. CSV is generally slower and not recommended,
                  but may be useful for debugging. NetCDF is more generally more efficient. \default{netcdf}"""))
    input_specs.addSub(data_handling)

    input_specs.addSub(InputData.parameterInputFactory('num_arma_samples', contentType=InputTypes.IntegerType,
                                                       descr=r"""provides the number of synthetic histories that should
                                                       be considered per system configuration in order to obtain a
                                                       reasonable representation of the economic metric. Sometimes
                                                       referred to as ``inner samples'' or ``denoisings''."""))

    # time discretization
    time_discr = InputData.parameterInputFactory('time_discretization',
                                                 descr=r"""node that defines how within-cycle time discretization should
                                                 be handled for solving the dispatch.""")
    time_discr.addSub(InputData.parameterInputFactory('time_variable', contentType=InputTypes.StringType,
                                                      descr=r"""name for the \texttt{time} variable used in this
                                                      simulation. \default{time}"""))
    time_discr.addSub(InputData.parameterInputFactory('year_variable', contentType=InputTypes.StringType,
                                                      descr=r"""name for the \texttt{year} or \texttt{macro} variable
                                                      used in this simulation. \default{Year}"""))
    time_discr.addSub(InputData.parameterInputFactory('start_time', contentType=InputTypes.FloatType,
                                                      descr=r"""value for \texttt{time} variable at which the inner
                                                      dispatch should begin. \default{0}"""))
    time_discr.addSub(InputData.parameterInputFactory('end_time', contentType=InputTypes.FloatType,
                                                      descr=r"""value for \texttt{time} variable at which the inner
                                                      dispatch should end. If not specified, both \xmlNode{time_interval}
                                                      and \xmlNode{num_timesteps} must be defined."""))
    time_discr.addSub(InputData.parameterInputFactory('num_steps', contentType=InputTypes.IntegerType,
                                                      descr=r"""number of discrete time steps for the inner dispatch.
                                                      Either this node or \xmlNode{time_interval} must be defined."""))
    time_discr.addSub(InputData.parameterInputFactory('time_interval',contentType=InputTypes.FloatType,
                                                      descr=r"""length of a time step for the inner dispatch, in units of
                                                      the time variable (not indices). Either this node or
                                                      \xmlNode{num_timesteps} must be defined. Note that if an integer
                                                      number of intervals do not fit between \xmlNode{start_time} and
                                                      \xmlNode{end_time}, an error will be raised."""))
    input_specs.addSub(time_discr)

    # economics global settings
    econ = InputData.parameterInputFactory('economics', ordered=False,
                                           descr= r"""node containing general economic setting in which to perform
                                           HERON analysis.""")
    econ.addSub(InputData.parameterInputFactory('ProjectTime', contentType=InputTypes.FloatType,
                                                descr=r"""the number of cycles (usually years) for the HERON analysis
                                                to cover."""))
    econ.addSub(InputData.parameterInputFactory('DiscountRate', contentType=InputTypes.FloatType,
                                                descr=r"""rate representing the time value of money to the firm used
                                                to discount cash flows in the multicycle economic analysis. Passed to
                                                the CashFlow module."""))
    econ.addSub(InputData.parameterInputFactory('tax', contentType=InputTypes.FloatType,
                                                descr=r"""the taxation rate, a metric which represents the
                                                rate at which the firm is taxed. Passed to the CashFlow module."""))
    econ.addSub(InputData.parameterInputFactory('inflation', contentType=InputTypes.FloatType,
                                                descr=r"""a metric which represents the rate at which the average
                                                price of goods and services in an economy increases over a cycle,
                                                usually a year. Passed to the CashFlow module."""))
    econ.addSub(InputData.parameterInputFactory('verbosity', contentType=InputTypes.IntegerType,
                                                descr=r"""the level of output to print from the CashFlow calculations.
                                                Passed to the CashFlow module."""))
    # is this actually CashFlow verbosity or is it really HERON verbosity?
    input_specs.addSub(econ)

    # dispatcher
    dispatch = InputData.parameterInputFactory('dispatcher', ordered=False,
                                               descr=r"""This node defines the dispatch strategy and options to use in
                                               the ``inner'' run.""")
    for d in known_dispatchers:
      vld_spec = get_dispatcher(d).get_input_specs()
      dispatch.addSub(vld_spec)
    input_specs.addSub(dispatch)

    # validator
    validator = InputData.parameterInputFactory('validator', ordered=False,
                                                descr=r"""This node defines the dispatch validation strategy and options
                                                to use in the ``inner'' run.""")
    for d in known_validators:
      vld_spec = get_validator(d).get_input_specs()
      validator.addSub(vld_spec)
    input_specs.addSub(validator)

    # optimization settings
    optimizer = InputData.parameterInputFactory('optimization_settings',
                                                descr=r"""This node defines the settings to be used for the optimizer in
                                                the ``outer'' run.""")
    metric_options = InputTypes.makeEnumType('MetricOptions', 'MetricOptionsType', list(cls.metrics_mapping.keys()))
    desc_metric_options = r"""determines the statistical metric (calculated by RAVEN BasicStatistics
                          or EconomicRatio PostProcessors) from the ``inner'' run to be used as the
                          objective in the ``outer'' optimization.
                          \begin{itemize}
                            \item For ``percentile'' the additional parameter \textit{percent=`X'}
                            is required where \textit{X} is the requested percentile (a floating
                            point value between 0.0 and 100.0).
                            \item For ``sortinoRatio'' and ``gainLossRatio'' the additional
                            parameter \textit{threshold=`X'} is required where \textit{X} is the
                            requested threshold (`median' or `zero').
                            \item For ``expectedShortfall'' and ``valueAtRisk'' the additional
                            parameter \textit{threshold=`X'} is required where \textit{X} is the
                            requested $\alpha$ value (a floating point value between 0.0 and 1.0).
                          \end{itemize}
                          """
    metric = InputData.parameterInputFactory('metric', contentType=metric_options, strictMode=True,
                                             descr=desc_metric_options)
    metric.addParam(name='percent',
                    param_type=InputTypes.FloatType,
                    descr=r"""requested percentile (a floating point value between 0.0 and 100.0).
                              Required when \xmlNode{metric} is ``percentile.''
                              \default{5}""")
    metric.addParam(name='threshold',
                    param_type=InputTypes.StringType,
                    descr=r"""\begin{itemize}
                                \item requested threshold (`median' or `zero'). Required when
                                \xmlNode{metric} is ``sortinoRatio'' or ``gainLossRatio.''
                                \default{`zero'}
                                \item requested $ \alpha $ value (a floating point value between 0.0
                                and 1.0). Required when \xmlNode{metric} is ``expectedShortfall'' or
                                ``valueAtRisk.'' \default{0.05}
                              \end{itemize}""")
    optimizer.addSub(metric)
    type_options = InputTypes.makeEnumType('TypeOptions', 'TypeOptionsType',
                                           ['min', 'max'])
    desc_type_options = r"""determines whether the objective should be minimized or maximized.
                            \begin{itemize}
                              \item when metric is ``expectedValue,'' ``minimum,'' ``maximum,''
                              ``median,'' ``percentile,'' ``sharpeRatio,'' ``sortinoRatio,''
                              ``gainLossRatio'' \default{max}
                              \item when metric is ``variance,'' ``sigma,'' ``variationCoefficient,''
                              ``skewness,'' ``kurtosis,'' ``expectedShortfall,'' ``valueAtRisk''
                              \default{min}
                            \end{itemize}"""
    type_sub = InputData.parameterInputFactory('type', contentType=type_options, strictMode=True,
                                               descr=desc_type_options)
    optimizer.addSub(type_sub)
    persistenceSub = InputData.parameterInputFactory('persistence',contentType=InputTypes.IntegerType,
                                                      descr=r"""provides the number of consecutive times convergence should be reached before a trajectory
                                                      is considered fully converged. This helps in preventing early false convergence.""" )
    optimizer.addSub(persistenceSub)
    input_specs.addSub(optimizer)

    convergence = InputData.parameterInputFactory('convergence',
                                                  descr=r"""defines the optimization convergence criteria.""")
    gradient_sub = InputData.parameterInputFactory('gradient',
                                                    descr=r"""termination criterion for the gradient
                                                              \default{1e-4}""")
    convergence.addSub(gradient_sub)
    objective_sub = InputData.parameterInputFactory('objective',
                                                    descr=r"""termination criterion for the objective function
                                                              \default{1e-8}""")
    convergence.addSub(objective_sub)
    stepsize_sub = InputData.parameterInputFactory('stepSize',
                                                    descr=r"""termination criterion for the design space step size""")
    convergence.addSub(stepsize_sub)

    optimizer.addSub(convergence)

    # Add magic variables that will be passed to the outer and inner.
    dispatch_vars = InputData.parameterInputFactory(
        'dispatch_vars',
        descr=r"This node defines a set containing additional variables"
        "to sample that are not associated with a specific component."
    )
    value_param = vp_factory.make_input_specs(
        'variable',
        descr=r"This node defines the single additional dispatch variable used in the case."
    )
    value_param.addParam(
        'name',
        param_type=InputTypes.StringType,
        descr=r"The unique name of the dispatch variable."
    )
    dispatch_vars.addSub(value_param)
    input_specs.addSub(dispatch_vars)

    # result statistics
    result_stats = InputData.parameterInputFactory('result_statistics',
                                                   descr=r"""This node defines the additional statistics
                                                   to be returned with the results. The statistics
                                                   \texttt{expectedValue} (prefix ``mean''),
                                                   \texttt{sigma} (prefix ``std''), and \texttt{median}
                                                   (prefix ``med'') are always returned with the results.
                                                   Each subnode is the RAVEN-style name of the desired
                                                   return statistic.""")
    for stat in cls.metrics_mapping:
      if stat not in ['expectedValue', 'sigma', 'median']:
        statistic = InputData.parameterInputFactory(stat, strictMode=True,
                                                    descr=r"""{} uses the prefix ``{}'' in the result output.""".format(stat, cls.metrics_mapping[stat]['prefix']))
        if stat == 'percentile':
          statistic.addParam('percent', param_type=InputTypes.StringType,
                            descr=r"""requested percentile (a floating point value between 0.0 and 100.0).
                            When no percent is given, returns both 5th and 95th percentiles.""")
        elif stat in ['sortinoRatio', 'gainLossRatio']:
          statistic.addParam('threshold', param_type=InputTypes.StringType,
                            descr=r"""requested threshold (``median" or ``zero").""", default='``median"')
        elif stat in ['expectedShortfall', 'valueAtRisk']:
          statistic.addParam('threshold', param_type=InputTypes.StringType,
                            descr=r"""requested threshold (a floating point value between 0.0 and 1.0).""",
                            default='``0.05"')
        result_stats.addSub(statistic)
    input_specs.addSub(result_stats)

    return input_specs

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
        'dispatch_plot': True,         # whether to output a dispatch plot in debug mode
        'cashflow_plot': True          # whether to output a cashflow plot in debug mode
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

  def read_input(self, xml):
    """
      Sets settings from input file
      @ In, xml, xml.etree.ElementTree.Element, input from user
      @ Out, None
    """
    # get specs for allowable inputs
    specs = self.get_input_specs()()
    specs.parseNode(xml)
    self.name = specs.parameterValues['name']
    for item in specs.subparts:
      # TODO move from iterative list to seeking list, at least for required nodes?
      if item.getName() == 'debug':
        self.debug['enabled'] = True
        for node in item.subparts:
          self.debug[node.getName()] = node.value
      elif item.getName() == 'label':
        self._labels[item.parameterValues['name']] = item.value
      if item.getName() == 'verbosity':
        self._verbosity = item.value
      if item.getName() == 'mode':
        self._mode = item.value
      elif item.getName() == 'parallel':
        self.useParallel = True
        self.parallelRunInfo = {}
        for sub in item.subparts:
          if sub.getName() == 'outer':
            self.outerParallel = sub.value
          elif sub.getName() == 'inner':
            self.innerParallel = sub.value
          elif sub.getName() == 'runinfo':
            for subsub in sub.subparts:
              self.parallelRunInfo[subsub.getName()] = str(subsub.value)
      elif item.getName() == 'metric':
        self._metric = item.value
      elif item.getName() == 'differential':
        self._diff_study = item.value
      elif item.getName() == 'num_arma_samples':
        self._num_samples = item.value
      elif item.getName() == 'time_discretization':
        self._time_discretization = self._read_time_discr(item)
      elif item.getName() == 'economics':
        for sub in item.subparts:
          self._global_econ[sub.getName()] = sub.value
          if self.debug['enabled'] and sub.getName() == "ProjectTime":
            self._global_econ[sub.getName()] = self.debug['macro_steps']
      elif item.getName() == 'dispatcher':
        # instantiate a dispatcher object.
        inp = item.subparts[0]
        name = inp.getName()
        typ = get_dispatcher(name)
        self.dispatcher = typ()
        self.dispatcher.read_input(inp)
      elif item.getName() == 'validator':
        vld = item.subparts[0]
        name = vld.getName()
        typ = get_validator(name)
        self.validator = typ()
        self.validator.read_input(vld)
      elif item.getName() == 'optimization_settings':
        self._optimization_settings = self._read_optimization_settings(item)
      elif item.getName() == 'dispatch_vars':
        for node in item.subparts:
          var_name = node.parameterValues['name']
          vp = ValuedParamHandler(var_name)
          vp.read(var_name, node, self.get_mode())
          self.dispatch_vars[var_name] = vp
      elif item.getName() == 'data_handling':
        self.data_handling = self._read_data_handling(item)
      elif item.getName() == 'workflow':
        self._workflow = item.value
      elif item.getName() == 'result_statistics':
        new_result_statistics = self._read_result_statistics(item)
        self._result_statistics.update(new_result_statistics)

    # checks
    if self._mode is None:
      self.raiseAnError('No <mode> node was provided in the <Case> node!')
    if self.dispatcher is None:
      self.raiseAnError('No <dispatch> node was provided in the <Case> node!')
    if self._time_discretization is None:
      self.raiseAnError('<time_discretization> node was not provided in the <Case> node!')
    if self.innerParallel == 0 and self.useParallel:
      #set default inner parallel to number of samples (denoises)
      self.innerParallel = self._num_samples
    #Note that if self.outerParallel == 0 and self.useParallel
    # then outerParallel will be set in template_driver _modify_outer_samplers
    cores_requested = self.innerParallel * self.outerParallel
    if cores_requested > 1:
      # check to see if the number of processes available can meet the request
      detected = os.cpu_count() - 1 # -1 to prevent machine OS locking up
      if detected < cores_requested:
        self.raiseAWarning('System may be overloaded and greatly increase run time! ' +
                           f'Number of available cores detected: {detected}; ' +
                           f'Number requested: {cores_requested} (inner: {self.innerParallel} * outer: {self.outerParallel}) ')

    # TODO what if time discretization not provided yet?
    self.dispatcher.set_time_discr(self._time_discretization)
    self.dispatcher.set_validator(self.validator)

    self.raiseADebug(f'Successfully initialized Case {self.name}.')

  def _read_data_handling(self, node):
    """
      Reads the data handling node.
      @ In, node, InputParams.ParameterInput, data handling head node
      @ Out, settings, dict, options for data handling
    """
    settings = {}
    # read settings
    for sub in node.subparts:
      name = sub.getName()
      if name == 'inner_to_outer':
        settings['inner_to_outer'] = sub.value
    # set defaults
    if 'inner_to_outer' not in settings:
      settings['inner_to_outer'] = 'netcdf'
    return settings

  def _read_time_discr(self, node):
    """
      Reads the time discretization node.
      @ In, node, InputParams.ParameterInput, time discretization head node
      @ Out, discr, tuple, (start, end, num_steps) for creating numpy linspace
    """
    # name of time variable
    var_name = node.findFirst('time_variable')
    if var_name is not None:
      self._time_varname = var_name.value
    # name of year variable
    year_name = node.findFirst('year_variable')
    if year_name is not None:
      self._year_varname = year_name.value
    # start
    start_node = node.findFirst('start_time')
    if start_node is None:
      start = 0.0
    else:
      start = start_node.value
    # options:
    end_node = node.findFirst('end_time')
    dt_node = node.findFirst('time_interval')
    num_node = node.findFirst('num_steps')
    # - specify end and num steps
    if (end_node and num_node):
      end = end_node.value
      num = num_node.value
    # - specify end and dt
    elif (end_node and dt_node):
      end = end_node.value
      dt = dt_node.value
      num = int(np.floor((end - start) / dt))
    # - specify dt and num steps
    elif (dt_node and num_node):
      dt = dt_node.value
      num = num_node.value
      end = dt * num + start
    else:
      self.raiseAnError(IOError, 'Invalid time discretization choices! Must specify any of the following pairs: ' +
                                 '(<end_time> and <num_steps>) or ' +
                                 '(<end_time> and <time_interval>) or ' +
                                 '(<num_steps> and <time_interval>.)')
    # TODO can we take it automatically from an ARMA later, either by default or if told to?
    return (start, end, num)

  def _read_optimization_settings(self, node):
    """
      Reads optimization settings node
      @ In, node, InputParams.ParameterInput, optimization settings head node
      @ Out, opt_settings, dict, optimization settings as dictionary
    """
    opt_settings = {}
    for sub in node.subparts:
      sub_name = sub.getName()
      # add metric information to opt_settings dictionary
      if sub_name == 'metric':
        opt_settings[sub_name] = {}
        metric_name = sub.value
        opt_settings[sub_name]['name'] = metric_name
        # some metrics have an associated parameter
        if metric_name == 'percentile':
          try:
            opt_settings[sub_name]['percent'] = sub.parameterValues['percent']
          except KeyError:
            opt_settings[sub_name]['percent'] = 5
        elif metric_name in ['sortinoRatio', 'gainLossRatio']:
          try:
            opt_settings[sub_name]['threshold'] = sub.parameterValues['threshold']
          except KeyError:
            opt_settings[sub_name]['threshold'] = 'zero'
        elif metric_name in ['expectedShortfall', 'valueAtRisk']:
          try:
            opt_settings[sub_name]['threshold'] = sub.parameterValues['threshold']
          except KeyError:
            opt_settings[sub_name]['threshold'] = 0.05
      elif sub_name == 'convergence':
        opt_settings[sub_name] = {}
        for ssub in sub.subparts:
          opt_settings[sub_name][ssub.getName()] = ssub.value
      else:
        # add other information to opt_settings dictionary (type is only information implemented)
        opt_settings[sub_name] = sub.value

    return opt_settings

  def _read_result_statistics(self, node):
    """
      Reads result statistics node
      @ In, node, InputParams.ParameterInput, result statistics head node
      @ Out, result_statistics, dict, result statistics settings as dictionary
    """
    # result_statistics keys are statistic name value is percent, threshold value, or None
    result_statistics = {}
    for sub in node.subparts:
      sub_name = sub.getName()
      if sub_name == 'percentile':
        try:
          percent = sub.parameterValues['percent']
          # if multiple percents are given, set as a list
          if sub_name in result_statistics:
            if isinstance(result_statistics[sub_name], list):
              if percent not in result_statistics[sub_name]:
                result_statistics[sub_name].append(percent)
            else:
              result_statistics[sub_name] = [result_statistics[sub_name], percent]
          else:
            result_statistics[sub_name] = percent
        except KeyError:
          result_statistics[sub_name] = self.metrics_mapping[sub_name]['percent']
      elif sub_name in ['sortinoRatio', 'gainLossRatio']:
        try:
          result_statistics[sub_name] = sub.parameterValues['threshold']
        except KeyError:
          result_statistics[sub_name] = self.metrics_mapping[sub_name]['threshold']
      elif sub_name in ['expectedShortfall', 'valueAtRisk']:
        try:
          threshold = sub.parameterValues['threshold']
          # if multiple thresholds are given, set as a list
          if sub_name in result_statistics:
            if isinstance(result_statistics[sub_name], list):
              if threshold not in result_statistics[sub_name]:
                result_statistics[sub_name].append(threshold)
            else:
              result_statistics[sub_name] = [result_statistics[sub_name], threshold]
          else:
            result_statistics[sub_name] = sub.parameterValues['threshold']
        except KeyError:
          result_statistics[sub_name] = self.metrics_mapping[sub_name]['threshold']
      else:
        result_statistics[sub_name] = None

    return result_statistics

  def initialize(self, components, sources):
    """
      Called after all objects are created, allows post-input initialization
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources (placeholders)
      @ Out, None
    """
    # check sources
    for src in sources:
      src.checkValid(self, components, sources)
    # dispatcher
    self.dispatcher.initialize(self, components, sources)

  def __repr__(self):
    """
      Determines how this class appears when printed.
      @ In, None
      @ Out, repr, str, string representation
    """
    return '<HERON Case>'

  def print_me(self, tabs=0, tab='  ', **kwargs):
    """
      Prints info about self
      @ In, tabs, int, number of tabs to insert before print
      @ In, tab, str, tab prefix
      @ Out, None
    """
    pre = tab*tabs
    self.raiseADebug(pre+'Case:')
    self.raiseADebug(pre+'  name:', self.name)
    self.raiseADebug(pre+'  mode:', self._mode)
    self.raiseADebug(pre+'  meric:', self._metric)
    self.raiseADebug(pre+'  diff_study:', self._diff_study)

  #### ACCESSORS ####
  def get_increments(self):
    """
      Accessor.
      @ In, None
      @ Out, self._increments, dict, increments for resource evaluation
    """
    return self._increments

  def get_working_dir(self, which):
    """
      Accessor.
      @ In, which, str, o or i, whether outer or inner working dir is needed
      @ Out, working_dir, str, relevant working dir
    """
    if which == 'outer':
      io = 'o'
    elif which == 'inner':
      io = 'i'
    else:
      raise NotImplementedError(f'Unrecognized working dir request: "{which}"')
    return f'{self.name}_{io}'

  def load_econ(self, components):
    """
      Loads active component cashflows
      @ In, components, list, list of HERON components
      @ Out, None
    """
    if 'active' not in self._global_econ:
      # NOTE self._metric can only be NPV right now!
      ## so no need for the "target" in the indicator
      indic = {'name': [self._metric]}
      indic['active'] = []
      for comp in components:
        comp_name = comp.name
        for cf in comp.get_cashflows():
          cf_name = cf.name
          indic['active'].append(f'{comp_name}|{cf_name}')
      self._global_econ['Indicator'] = indic

  def get_econ(self, components):
    """
      Accessor for economic settings for this case
      @ In, components, list, list of HERON components
      @ Out, get_econ, dict, dictionary of global economic settings
    """
    self.load_econ(components)
    return self._global_econ

  def get_labels(self):
    """
      Accessor
      @ In, None
      @ Out, _labels, dict, labels for this case
    """
    return self._labels

  def get_metric(self):
    """
      Accessor
      @ In, None
      @ Out, metric, str, target metric for this case
    """
    return self._metric

  def get_mode(self):
    """
      Accessor
      @ In, None
      @ Out, mode, str, mode of analysis for this case (sweep, opt, etc)
    """
    return self._mode

  def get_verbosity(self):
    """
      Accessor
      @ In, None
      @ Out, verbosity, str, level of vebosity for RAVEN inner/outer runs.
    """
    return self._verbosity

  def get_num_samples(self):
    """
      Accessor
      @ In, None
      @ Out, num_samples, int, number of dispatch realizations to consider
    """
    if self.debug['enabled']:
      return self.debug['inner_samples']
    else:
      return self._num_samples

  def get_num_timesteps(self):
    """
      Accessor
      @ In, None
      @ Out, num_timesteps, int, number of time steps for inner dispatch
    """
    return self._num_hist

  def get_time_name(self):
    """
      Provides the name of the time variable.
      @ In, None
      @ Out, time name, string, name of time variable
    """
    return self._time_varname

  def get_year_name(self):
    """
      Provides the name of the time variable.
      @ In, None
      @ Out, time name, string, name of time variable
    """
    return self._year_varname

  def get_Resample_T(self):
    """
      Accessor
      @ In, None
      @ Out, Resample_T, float, user-requested time deltas
    """
    return self._Resample_T

  def get_hist_interval(self):
    """
      Accessor
      @ In, None
      @ Out, hist_interval, float, user-requested time deltas
    """
    return self._hist_interval

  def get_hist_length(self):
    """
      Accessor
      @ In, None
      @ Out, hist_len, int, length of inner histories
    """
    return self._hist_len

  def get_dispatch_var(self, name):
    """
      Accessor
      @ In, name, str, the name of the dispatch_var
      @ Out, dispatch_var, ValuedParamHandler, a ValuedParam object.
    """
    return self.dispatch_vars[name]

  #### API ####
  def write_workflows(self, components, sources, loc):
    """
      Writes workflows for this case to XMLs on disk.
      @ In, components, HERON components, components for the simulation
      @ In, sources, HERON sources, sources for the simulation
      @ In, loc, str, location in which to write files
      @ Out, None
    """
    # load templates
    template_class = self._load_template()
    inner, outer = template_class.createWorkflow(self, components, sources)

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
    template_name = 'template_driver'
    # import template module
    sys.path.append(heron_dir)
    module = importlib.import_module(f'templates.{template_name}', package="HERON")
    # load template, perform actions
    template_class = module.Template(messageHandler=self.messageHandler)
    template_class.loadTemplate(template_dir)
    return template_class

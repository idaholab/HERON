
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Defines the Cases entity.
"""
from __future__ import unicode_literals, print_function
import os
import sys
import copy
import importlib

import numpy as np

from base import Base
import Components
import Placeholders

from dispatch.Factory import known as known_dispatchers
from dispatch.Factory import get_class as get_dispatcher

import _utils as hutils
framework_path = hutils.get_raven_loc()
sys.path.append(framework_path)
from utils import InputData, InputTypes, xmlUtils

class Case(Base):
  """
    Produces something, often as the cost of something else
    TODO this case is for "sweep-opt", need to make a superclass for generic
  """
  #### INITIALIZATION ####
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    input_specs = InputData.parameterInputFactory('Case', ordered=False, baseNode=None,
        descr=r"""The \xmlNode{Case} node contains the general physics and economics information
                required for a HERON workflow to be created and solved.""")
    input_specs.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""the name by which this analysis should be referred within HERON.""")

    mode_options = InputTypes.makeEnumType('ModeOptions', 'ModeOptionsType', ['opt', 'sweep'])
    desc_mode_options = r"""determines whether the outer RAVEN should perform optimization,
                         or a parametric (``sweep'') study. \default{sweep}"""
    input_specs.addSub(InputData.parameterInputFactory('mode', contentType=mode_options,
                         strictMode=True, descr=desc_mode_options))

    # not yet implemented TODO
    #econ_metrics = InputTypes.makeEnumType('EconMetrics', 'EconMetricsTypes', ['NPV', 'lcoe'])
    #desc_econ_metrics = r"""indicates the economic metric that should be used for the HERON analysis. For most cases, this
    #                    should be NPV."""
    # input_specs.addSub(InputData.parameterInputFactory('metric', contentType=econ_metrics, descr=desc_econ_metrics))
    # input_specs.addSub(InputData.parameterInputFactory('differential', contentType=InputTypes.BoolType, strictMode=True,
    # descr=r"""(not implemented) allows differentiation between two HERON runs as a desired
    # economic metric."""

    input_specs.addSub(InputData.parameterInputFactory('num_arma_samples', contentType=InputTypes.IntegerType,
        descr=r"""provides the number of synthetic histories that should be considered per system configuration
              in order to obtain a reasonable representation of the economic metric. Sometimes referred to as
              ``inner samples'' or ``denoisings''."""))

    # time discretization
    time_discr = InputData.parameterInputFactory('time_discretization',
        descr=r"""node that defines how within-cycle time discretization should be handled for
        solving the dispatch.""")
    time_discr.addSub(InputData.parameterInputFactory('time_variable', contentType=InputTypes.StringType,
        descr=r"""name for the \texttt{time} variable used in this simulation. \default{time}"""))
    time_discr.addSub(InputData.parameterInputFactory('start_time', contentType=InputTypes.FloatType,
        descr=r"""value for \texttt{time} variable at which the inner dispatch should begin. \default{0}"""))
    time_discr.addSub(InputData.parameterInputFactory('end_time', contentType=InputTypes.FloatType,
        descr=r"""value for \texttt{time} variable at which the inner dispatch should end. If not specified,
              both \xmlNode{time_interval} and \xmlNode{num_timesteps} must be defined."""))
    time_discr.addSub(InputData.parameterInputFactory('num_steps', contentType=InputTypes.IntegerType,
        descr=r"""number of discrete time steps for the inner dispatch.
              Either this node or \xmlNode{time_interval} must be defined."""))
    time_discr.addSub(InputData.parameterInputFactory('time_interval', contentType=InputTypes.FloatType,
        descr=r"""length of a time step for the inner dispatch, in units of the time variable (not indices).
              Either this node or \xmlNode{num_timesteps} must be defined. Note that if an integer number of
              intervals do not fit between \xmlNode{start_time} and \xmlNode{end_time}, an error will be raised."""))
    input_specs.addSub(time_discr)

    # economics global settings
    econ = InputData.parameterInputFactory('economics', ordered=False,
        descr= r"""node containing general economic setting in which to perform HERON analysis.""")
    econ.addSub(InputData.parameterInputFactory('ProjectTime', contentType=InputTypes.FloatType,
        descr=r"""the number of cycles (usually years) for the HERON analysis to cover."""))
    econ.addSub(InputData.parameterInputFactory('DiscountRate', contentType=InputTypes.FloatType,
        descr=r"""rate representing the time value of money to the firm used to discount cash flows
              in the multicycle economic analysis. Passed to the CashFlow module."""))
    econ.addSub(InputData.parameterInputFactory('tax', contentType=InputTypes.FloatType,
        descr=r"""the taxation rate, a metric which represents the
               rate at which the firm is taxed. Passed to the CashFlow module."""))
    econ.addSub(InputData.parameterInputFactory('inflation', contentType=InputTypes.FloatType,
        descr=r"""a metric which represents the rate at which the average price of goods and
              services in an economy increases over a cycle, usually a year.
              Passed to the CashFlow module."""))
    econ.addSub(InputData.parameterInputFactory('verbosity', contentType=InputTypes.IntegerType,
        descr=r"""the level of output to print from the CashFlow calculations. Passed to the CashFlow
              module.""")) # is this actually CashFlow verbosity or is it really HERON verbosity?
    input_specs.addSub(econ)

    # increments for resources
    dispatch = InputData.parameterInputFactory('dispatcher', ordered=False,
        descr=r"""This node defines the dispatch strategy and options to use in the ``inner'' run.""")
    # TODO get types directly from Factory!
    dispatch_options = InputTypes.makeEnumType('DispatchOptions', 'DispatchOptionsType', [d for d in known_dispatchers])
    dispatch.addSub(InputData.parameterInputFactory('type', contentType=dispatch_options,
        descr=r"""the name of the ``inner'' dispatch strategy to use."""))
    incr = InputData.parameterInputFactory('increment', contentType=InputTypes.FloatType,
        descr=r"""When performing an incremental resource balance as part of a dispatch solve, this
              determines the size of incremental adjustments to make for the given resource. If this
              value is large, then the solve is accelerated, but may miss critical inflection points
              in economical tradeoff. If this value is small, the solve may take much longer.""")
    incr.addParam('resource', param_type=InputTypes.StringType, required=True,
        descr=r"""indicates the resource for which this increment is being defined.""")
    dispatch.addSub(incr)
    input_specs.addSub(dispatch)

    return input_specs

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Base.__init__(self, **kwargs)
    self.name = None           # case name
    self._mode = None          # extrema to find: min, max, sweep
    self._metric = 'NPV'       # UNUSED (future work); economic metric to focus on: lcoe, profit, cost

    self.dispatch_name = None  # type of dispatcher to use
    self.dispatcher = None     # type of dispatcher to use

    self._diff_study = None    # is this only a differential study?
    self._num_samples = 1      # number of ARMA stochastic samples to use ("denoises")
    self._hist_interval = None # time step interval, time between production points
    self._hist_len = None      # total history length, in same units as _hist_interval
    self._num_hist = None      # number of history steps, hist_len / hist_interval
    self._global_econ = {}     # global economics settings, as a pass-through
    self._increments = {}      # stepwise increments for resource balancing
    self._time_varname = 'time' # name of the variable throughout simulation

    self._time_discretization = None # (start, end, number) for constructing time discretization, same as argument to np.linspace
    self._Resample_T = None    # user-set increments for resources

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
      # TODO move from iterative list to seeking list, at least for required nodes
      if item.getName() == 'mode':
        self._mode = item.value
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
      elif item.getName() == 'dispatcher':
        # instantiate a dispatcher object.
        dispatch_name = item.findFirst('type').value
        dispatcher_type = get_dispatcher(dispatch_name)
        self.dispatcher = dispatcher_type()
        self.dispatcher.read_input(item)
        # XXX Remove -> send to dispatcher instead
        for sub in item.subparts:
          if item.getName() == 'increment':
            self._increments[item.parameterValues['resource']] = item.value

    # checks
    if self._mode is None:
      self.raiseAnError('No <mode> node was provided in the <Case> node!')
    if self.dispatcher is None:
      self.raiseAnError('No <dispatch> node was provided in the <Case> node!')
    if self._time_discretization is None:
      self.raiseAnError('<time_discretization> node was not provided in the <Case> node!')

    # TODO what if time discretization not provided yet?
    self.dispatcher.set_time_discr(self._time_discretization)

    # derivative calculations
    # OLD self._num_hist = self._hist_len // self._hist_interval # TODO what if it isn't even?

    self.raiseADebug('Successfully initialized Case {}.'.format(self.name))

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

  def initialize(self, components, sources):
    """
      Called after all objects are created, allows post-input initialization
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources (placeholders)
      @ Out, None
    """
    self.dispatcher.initialize(self, components, sources)

  def __repr__(self):
    """
      Determines how this class appears when printed.
      @ In, None
      @ Out, repr, str, string representation
    """
    return '<HERON Case>'

  def print_me(self, tabs=0, tab='  '):
    """
      Prints info about self
      @ In, tabs, int, number of tabs to insert before print
      @ In, tab, str, tab prefix
      @ Out, None
    """
    pre = tab*tabs
    print(pre+'Case:')
    print(pre+'  name:', self.name)
    print(pre+'  mode:', self._mode)
    print(pre+'  meric:', self._metric)
    print(pre+'  diff_study:', self._diff_study)

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
      raise NotImplementedError('Unrecognized working dir request: "{}"'.format(which))
    return '{case}_{io}'.format(case=self.name, io=io)

  def get_econ(self, components):
    """
      Accessor for economic settings for this case
      @ In, components, list, list of HERON components
      @ Out, get_econ, dict, dictionary of global economic settings
    """
    # only add additional params the first time this is called
    if 'active' not in self._global_econ:
      # NOTE self._metric can only be NPV right now! XXX TODO FIXME
      ## so no need for the "target" in the indicator
      indic = {'name': [self._metric]}
      indic['active'] = []
      for comp in components:
        comp_name = comp.name
        for cf in comp.get_cashflows():
          cf_name = cf.name
          indic['active'].append('{}|{}'.format(comp_name, cf_name))
      self._global_econ['Indicator'] = indic
    return self._global_econ

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

  def get_num_samples(self):
    """
      Accessor
      @ In, None
      @ Out, num_samples, int, number of dispatch realizations to consider
    """
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
    inner, outer, cash = template_class.createWorkflow(self, components, sources)

    template_class.writeWorkflow((inner, outer, cash), loc)

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
    module = importlib.import_module('templates.{}'.format(template_name))
    # load template, perform actions
    template_class = module.Template()
    template_class.loadTemplate(template_dir)
    return template_class

  def _modify(self, templates, components, sources):
    """
      Modifies template files to prepare case.
      @ In, templates, dict, map of file templates
      @ In, components, list, HERON Components
      @ In, sources, list, HERON Placeholders
      @ Out, _modify, dict, modified files
    """
    outer = self._modify_outer(templates['outer'], components, sources)
    inner = self._modify_inner(templates['inner'], components, sources)
    return {'outer':outer, 'inner':inner}

  def _modify_outer(self, template, components, sources):
    """
      Modifies the "outer" template file
      @ In, template, xml, file template
      @ In, components, list, HERON Components
      @ In, sources, list, HERON Placeholders
      @ Out, template, xml, modified template
    """
    ###################
    # RUN INFO        #
    ###################
    run_info = template.find('RunInfo')
    case_name = self.get_working_dir('outer') #self.string_templates['jobname'].format(case=self.name, io='o')
    ooooooooo # I don't think this gets run!
    # job name
    run_info.find('JobName').text = case_name
    # working dir
    run_info.find('WorkingDir').text = case_name
    # TODO sequence, maybe should be modified after STEPS (or part of it)

    ###################
    # STEPS           #
    # FIlES           #
    ###################
    # no ARMA steps, that's all on the inner

    ###################
    # VARIABLE GROUPS #
    ###################
    var_groups = template.find('VariableGroups')
    # capacities
    caps = var_groups[0]
    caps.text = ', '.join('{}_capacity'.format(x.name) for x in components)
    # results
    ## these don't need to be changed, they're fine.

    ###################
    # DATA OBJECTS    #
    ###################
    # thanks to variable groups, we don't really need to adjust these.

    ###################
    # MODELS          #
    ###################
    # aliases
    raven = template.find('Models').find('Code')
    text = 'Samplers|MonteCarlo@name:mc_arma|constant@name:{}_capacity'
    for component in components:
      name = component.name
      attribs = {'variable':'{}_capacity'.format(name), 'type':'input'}
      new = xmlUtils.newNode('alias', text=text.format(name), attrib=attribs)
      raven.append(new)
    # TODO location of RAVEN? We should know globally somehow.

    ###################
    # DISTRIBUTIONS   #
    # SAMPLERS        #
    ###################
    dists_node = template.find('Distributions')
    samps_node = template.find('Samplers').find('Grid')
    # number of denoisings
    ## assumption: first node is the denoises node
    samps_node.find('constant').text = str(self._num_samples)
    # add sweep variables to input
    dist_template = xmlUtils.newNode('Uniform')
    dist_template.append(xmlUtils.newNode('lowerBound'))
    dist_template.append(xmlUtils.newNode('upperBound'))
    var_template = xmlUtils.newNode('variable')
    var_template.append(xmlUtils.newNode('distribution'))
    var_template.append(xmlUtils.newNode('grid', attrib={'type':'value', 'construction':'custom'}))
    for component in components:
      interaction = component.get_interaction()
      # if produces, then its capacity might be flexible
      # TODO this algorithm does not check for everthing to be swept! Future work could expand it.
      ## Currently checked: Component.Interaction.Capacity
      # TODO doesn't test for Components.Demand; maybe the user wants to perturb the demand level?
      if isinstance(interaction, (Components.Producer, Components.Storage)):
        # is the capacity variable (being swept over)?
        name = component.name
        var_name = self.string_templates['variable'].format(unit=name, feature='capacity')
        if isinstance(interaction._capacity, list):
          dist_name = self.string_templates['distribution'].format(unit=name, feature='capacity')
          new = copy.deepcopy(dist_template)
          new.attrib['name'] = dist_name
          new.find('lowerBound').text = str(min(interaction._capacity))
          new.find('upperBound').text = str(max(interaction._capacity))
          dists_node.append(new)
          # also mess with the Sampler block
          new = copy.deepcopy(var_template)
          new.attrib['name'] = var_name
          new.find('distribution').text = dist_name
          new.find('grid').text = ', '.join(str(x) for x in sorted(interaction._capacity))
          samps_node.append(new)
          # TODO assumption (input checked): only one interaction per component
        # elif the capacity is fixed, it becomes a constant
        else:
          samps_node.append(xmlUtils.newNode('constant', text=interaction._capacity, attrib={'name': var_name}))

    ###################
    # OUTSTREAMS      #
    ###################
    # no changes needed here!

    # TODO copy needed model/ARMA/etc files to Outer Working Dir so they're known
    return template

  def _modify_inner(self, template, components, sources):
    """
      Modifies the "inner" template file
      @ In, template, xml, file template
      @ In, components, list, HERON Components
      @ In, sources, list, HERON Placeholders
      @ Out, template, xml, modified template
    """
    ###################
    # RUN INFO        #
    # STEPS           #
    ###################
    run_info = template.find('RunInfo')
    steps = template.find('Steps')
    models = template.find('Models')
    # case, working dir
    case_name = self.get_working_dir('inner') #self.string_templates['jobname'].format(case=self.name, io='i')
    run_info.find('JobName').text = case_name
    run_info.find('WorkingDir').text = case_name
    # steps
    ## for now, just load all sources
    ## TODO someday, only load what's needed
    for source in sources:
      name = source.name
      if isinstance(source, Placeholders.ARMA):
        # add a model block
        models.append(xmlUtils.newNode('ROM', attrib={'name':name, 'subType':'pickledROM'}))
        # add a read step
        read_step_name = self.string_templates['stepname'].format(action='read', subject=name)
        new_step = xmlUtils.newNode('IOStep', attrib={'name': read_step_name})
        new_step.append(xmlUtils.newNode('Input', attrib={'class':'Files', 'type':''}, text=source._source))
        new_step.append(xmlUtils.newNode('Output', attrib={'class':'Models', 'type':'ROM'}, text=name))
        # add a sample step (sample #denoises)
        # add a custom sampler to read the samples
        ## sequence # NOTE order matters!
        seq = run_info.find('Sequence').text
        if seq is None:
          run_info.find('Sequence').text = read_step_name
        else:
          run_info.find('Sequence').text = '{}, {}'.format(seq, read_step_name)

        # TODO ensemble model? data object?
        steps.append(new_step)
      elif isinstance(source, Placeholders.CSV):
        # add a File
        # add a step to read
        # add a data object to read into
        ## TODO no one needs this right now, so forget it.
        pass # TODO
      elif isinstance(source, Placeholders.Function):
        pass # TODO
    return template

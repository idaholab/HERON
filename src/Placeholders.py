
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Evaluated signal values for use in HERON
"""
from __future__ import unicode_literals, print_function
import os
import sys
import abc
import copy

from base import Base
from scipy import interpolate
import _utils as hutils

framework_path = hutils.get_raven_loc()
sys.path.append(framework_path)
from utils import InputData, InputTypes, utils, xmlUtils

class Placeholder(Base):
  """
    Objects that hold a place in the HERON workflow
    but don't hold value until converted into the RAVEN workflow.
  """
  def __init__(self, **kwargs):
    """
      Constructor.
      @ In, kwargs, dict, passthrough args
      @ Out, None
    """
    Base.__init__(self, **kwargs)
    self.name = None         # identifier
    self._source = None      # name of file? the signal should come from
    self._var_names = None   # LIST of names of output variable from CSV or ARMA
    self._type = None        # source type, such as CSV, pickle, function ...
    self._target_file = None # source file to take data from
    self._workingDir = kwargs['loc'] # where is the HERON input file?

  @classmethod
  @abc.abstractmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, specs, InputData, specs
    """
    pass

  def read_input(self, xml):
    """
      Sets settings from input file
      @ In, xml, xml.etree.ElementTree.Element, input from user
      @ Out, None
    """
    specs = self.get_input_specs()()
    specs.parseNode(xml)
    self.name = specs.parameterValues['name']
    self._source = specs.value
    # check source exists
    if self._source.startswith('%HERON%'):
      # magic word for "relative to HERON root"
      heron_path = hutils.get_heron_loc()
      self._target_file = os.path.abspath(self._source.replace('%HERON%', heron_path))
    else:
      # check absolute path
      rel_interp = os.path.abspath(os.path.join(self._workingDir, self._source))
      if os.path.isfile(rel_interp):
        self._target_file = rel_interp
      else:
        # check absolute path
        abs_interp = os.path.abspath(self._source)
        if os.path.isfile(abs_interp):
          self._target_file = abs_interp
    # check source
    if not os.path.isfile(self._target_file):
      self.raiseAnError(IOError, f'File not found for <DataGenerator><{self._type}> named "{self.name}".' +
                        f'\nLooked in: "{self._target_file}"' +
                        f'\nGiven location: "{self._source}"')
    return specs

  def checkValid(self, case, components, sources):
    """
      Check validity of placeholder given rest of system
      @ In, case, HERON.Case, case
      @ In, case, list(HERON.Component), components
      @ In, sources, list(HERON.Placeholder), sources
      @ Out, None
    """
    pass # overwrite to check

  def print_me(self, tabs=0, tab='  '):
    """
      Prints info about self
      @ In, tabs, int, number of tabs to prepend
      @ In, tab, str, format for tabs
      @ Out, None
    """
    pre = tab*tabs
    print(pre+'DataGenerator:')
    print(pre+'  name:', self.name)
    print(pre+'  source:', self._source)
    print(pre+'  variables:', self._var_names)

  def is_type(self, typ):
    """
      Checks for matching type
      @ In, typ, str, type to check against
      @ Out, is_type, bool, True if matching request
    """
    # maybe it's not anything we know about
    if typ not in ['ARMA', 'Function']:
      return False
    return eval('isinstance(self, {})'.format(typ))

  def get_variable(self):
    """
      Returns the variable(s) in use for this placeholder.
      @ In, None
      @ Out, var_names, list, variable names
    """
    return self._var_names





class ARMA(Placeholder):
  """
    Placeholder for signals coming from the ARMA
  """
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('ARMA', contentType=InputTypes.StringType, ordered=False, baseNode=None,
        descr=r"""This data source is a source of synthetically-generated histories trained by RAVEN.
              The RAVEN ARMA ROM should be trained and serialized before using it in HERON. The text
              of this node indicates the location of the serialized ROM. This location is usually relative
              with respect to the HERON XML input file; however, a full absolute path can be used,
              or the path can be prepended with ``\%HERON\%'' to be relative to the installation
              directory of HERON.""")
    specs.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""identifier for this data source in HERON and in the HERON input file. """)
    specs.addParam('variable', param_type=InputTypes.StringListType, required=True,
        descr=r"""provides the names of the variables from the synthetic history generators that will
              be used in this analysis.""")
    # TODO someday read this directly off the model instead of asking the user!
    specs.addParam('evalMode', param_type=InputTypes.StringType, required=False,
        descr=r"""desired sampling mode for the ARMA. See the RAVEN manual for options. \default{clustered}""")
    return specs

  def __init__(self, **kwargs):
    """
      Constructor.
      @ In, kwargs, dict, passthrough args
      @ Out, None
    """
    Placeholder.__init__(self, **kwargs)
    self._type = 'ARMA'
    self._var_names = None # variables from the ARMA to use
    self.eval_mode = None # ARMA evaluation style (clustered, full, truncated)
    self.needs_multiyear = None # if not None, then this is a 1-year ARMA that needs multiyearing
    self.limit_interp = None # if not None, gives the years to limit this interpolated ROM to

  def read_input(self, xml):
    """
      Sets settings from input file
      @ In, xml, xml.etree.ElementTree.Element, input from user
      @ Out, None
    """
    specs = Placeholder.read_input(self, xml)
    self._var_names = specs.parameterValues['variable']
    self.eval_mode = specs.parameterValues.get('evalMode', 'clustered')
    # check that the source ARMA exists

  def checkValid(self, case, components, sources):
    """
      Check validity of placeholder given rest of system
      @ In, case, HERON.Case, case
      @ In, case, list(HERON.Component), components
      @ In, sources, list(HERON.Placeholder), sources
      @ Out, None
    """
    print(f'Checking ROM at "{self._target_file}" ...')
    structure = hutils.get_synthhist_structure(self._target_file)
    interpolated = 'macro' in structure
    clustered = bool(structure['clusters'])
    # segmented = bool(structure['segments']) # TODO
    print(f'For DataGenerator <{self._type}> "{self.name}", detected: ' +
          f'{"" if interpolated else "NOT"} interpolated, ' +
          f'{"" if clustered else "NOT"} clustered.')
    # expect that project life == num macro years
    project_life = hutils.get_project_lifetime(case, components) - 1 # one less for construction year
    if interpolated:
      # if interpolated, needs more checking
      interp_years = structure['macro']['num']
      if interp_years >= project_life:
        print(f' -> "{self.name}" interpolates {interp_years} macro steps, and project life is {project_life}, so histories will be trunctated.')
        self.limit_interp = project_life
      else:
        raise RuntimeError(f' -> "{self.name}" interpolates {interp_years} macro steps, but project life is {project_life}!')
    else:
      # if single year, we can use multiyear so np
      print(f' -> "{self.name}" will be extended to project life ({project_life}) macro steps using <Multicycle>.')
      self.needs_multiyear = project_life




class Function(Placeholder):
  """
    Placeholder for values that are evaluated on the fly
  """
  # TODO combine with RAVEN's external Function class?
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('Function', contentType=InputTypes.StringType,
        ordered=False, baseNode=None,
        descr=r"""This data source is a custom Python function to provide derived values.
              Python functions have access to the variables within the dispatcher. The text
              of this node indicates the location of the python file. This location is usually relative
              with respect to the HERON XML input file; however, a full absolute path can be used,
              or the path can be prepended with ``\%HERON\%'' to be relative to the installation
              directory of HERON.""")
    specs.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""identifier for this data source in HERON and in the HERON input file. """)
    return specs

  def __init__(self, **kwargs):
    """
      Constructor.
      @ In, kwargs, dict, passthrough args
      @ Out, None
    """
    Placeholder.__init__(self, **kwargs)
    self._type = 'Function'
    self._module = None
    self._module_methods = {}

  def __getstate__(self):
    """
      Serialization.
      @ In, None
      @ Out, d, dict, object contents
    """
    # d = super(self, __getstate__) TODO only if super has one ...
    d = copy.deepcopy(dict((k, v) for k, v in self.__dict__.items() if k not in ['_module']))
    return d

  def __setstate__(self, d):
    """
      Deserialization.
      @ In, d, dict, object contents
      @ Out, None
    """
    self.__dict__ = d
    load_string, _ = utils.identifyIfExternalModelExists(self, self._target_file, '')
    module = utils.importFromPath(load_string, True)
    if not module:
      raise IOError(f'Module "{self._source}" for function "{self.name}" was not found!')
    self._module = module
    self._set_callables(module)

  def read_input(self, xml):
    """
      Sets settings from input file
      @ In, xml, xml.etree.ElementTree.Element, input from user
      @ Out, None
    """
    Placeholder.read_input(self, xml)
    # load module
    load_string, _ = utils.identifyIfExternalModelExists(self, self._target_file, '')
    module = utils.importFromPath(load_string, True)
    if not module:
      raise IOError(f'Module "{self._source}" for function "{self.name}" was not found!')
    self._module = module
    # TODO do we need to set the var_names? self._var_names = _var_names
    self._set_callables(module)

  def _set_callables(self, module):
    """
      Build a dict of callable methods with the right format
      @ In, module, python Module, module to load methods from
      @ Out, None
    """
    for name, member in module.__dict__.items():
      # check all conditions for not acceptable formats; if none of those true, then it's a good method
      ## check callable as a function
      if not callable(member):
        continue
      self._module_methods[name] = member

  def evaluate(self, method, request, data_dict):
    """
      Evaluates requested method in stored module.
      @ In, method, str, method name
      @ In, request, dict, requested action
      @ In, data_dict, dict, dictonary of evaluation parameters (metadata)
      @ Out, result, dict, results of evaluation
    """
    result = self._module_methods[method](request, data_dict)
    if not (hasattr(result, '__len__') and len(result) == 2 and all(isinstance(r, dict) for r in result)):
      raise RuntimeError('From Function "{f}" method "{m}" expected {s}.{m} '.format(f=self.name, m=method, s=self._source) +\
                         'to return with form (results_dict, meta_dict) with both as dictionaries, but received:\n' +\
                         '    {}'.format(result))
    return result





class Resampling_time(Placeholder):
  """
    Placeholder for signals coming from the ARMA
    FIXME Probably note used any more, and should be removed.
  """
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('Resampling_time', contentType=InputTypes.StringType, ordered=False, baseNode=None)
    return specs

  def __init__(self, **kwargs):
    """
      Constructor.
      @ In, kwargs, dict, passthrough arguments
      @ Out, None
    """
    Placeholder.__init__(self, **kwargs)
    self._type = 'Resampling_time'

  def read_input(self, xml):
    """
      Sets settings from input file
      @ In, xml, xml.etree.ElementTree.Element, input from user
      @ Out, None
    """
    specs = Placeholder.read_input(self, xml)
    self._var_names = specs.parameterValues['variable']






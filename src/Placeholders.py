
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Evaluated signal values for use in HERON
"""
import os
import sys
import abc
import copy
import functools

import HERON.src._utils as hutils
from HERON.src.base import Base

try:
  import ravenframework
except ModuleNotFoundError:
  FRAMEWORK_PATH = hutils.get_raven_loc()
  sys.path.append(FRAMEWORK_PATH)
from ravenframework.utils import InputData, InputTypes, utils, xmlUtils

from ravenframework.ROMExternal import ROMLoader


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
    elif self._source.startswith('%FARM%'):
      # magic word for "relative to FARM root"
      farm_path = hutils.get_farm_loc()
      self._target_file = os.path.abspath(self._source.replace('%FARM%', farm_path))
    elif self._source.startswith('%HERON_DATA%'):
      #magic word to try and find HERON data files
      #note that this first checks for a HERON_DATA environment variable
      heron_data = os.environ.get("HERON_DATA")
      if heron_data is None:
        heron_path = hutils.get_heron_loc()
        # Look in a few places for the data directory. The first location that exists is used.
        # The first option is within the HERON root directory. This should exist for a source
        # code installation of HERON. The second location is where the data directory is
        # expected to be in a standalone installation of FORCE.
        possible_paths = ["data", "../../examples/data"]
        for path in possible_paths:
          path = os.path.join(heron_path, path)
          if os.path.exists(path):
            heron_data = path
            break
        else:
          checked_paths = [os.path.join(heron_path, path) for path in possible_paths]
          self.raiseAnError(IOError,"ERROR trying to find %HERON_DATA% and not found at any of [", ", ".join(checked_paths), "] and no HERON_DATA environmental variable found")
      elif not os.path.exists(heron_data):
        self.raiseAnError(IOError, "ERROR path", heron_data, "found in HERON_DATA environment variable does not seem to exist")
      self.raiseADebug("HERON_DATA found at", os.path.abspath(heron_data))
      self._target_file = os.path.abspath(self._source.replace('%HERON_DATA%', heron_data))
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
        else:
          # let relative path trigger the error
          self._target_file = rel_interp
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
    self.raiseADebug(pre+'DataGenerator:')
    self.raiseADebug(pre+'  name:', self.name)
    self.raiseADebug(pre+'  source:', self._source)
    self.raiseADebug(pre+'  variables:', self._var_names)

  def is_type(self, typ):
    """
      Checks for matching type
      @ In, typ, str, type to check against
      @ Out, is_type, bool, True if matching request
    """
    return self.type == typ

  @property
  def type(self) -> str:
    return self._type

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
              or the path can be prepended with ``\%HERON_DATA\%'' to be relative to the HERON data directory (``HERON/data'' in the git repository, but the location of ``\%HERON_DATA\%'' can be overridden by exporting a ``HERON_DATA'' environmental variable),
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
    self.raiseAMessage(f'Checking ROM at "{self._target_file}"')
    structure = hutils.get_synthhist_structure(self._target_file)
    interpolated = 'macro' in structure
    clustered = bool(structure['clusters'])
    # segmented = bool(structure['segments']) # TODO
    self.raiseAMessage(
        f'For DataGenerator <{self._type}> "{self.name}", detected: {"" if interpolated else "NOT"} interpolated, ' +
        f'{"" if clustered else "NOT"} clustered.'
    )
    # expect that project life == num macro years
    project_life = hutils.get_project_lifetime(case, components) - 1 # one less for construction year
    if interpolated:
      # if interpolated, needs more checking
      interp_years = structure['macro']['num']
      if interp_years >= project_life:
        self.raiseADebug(
            f'"{self.name}" interpolates {interp_years} macro steps,' +
            f'and project life is {project_life}, so histories will be trunctated.'
        )
        self.limit_interp = project_life
      else:
        self.raiseAnError(
            RuntimeError, f'"{self.name}" interpolates {interp_years} macro steps, but project life is {project_life}!'
        )
    else:
      # if single year, we can use multiyear so np
      self.raiseADebug(
          f'"{self.name}" will be extended to project life ({project_life}) macro steps using <Multicycle>.'
      )
      self.needs_multiyear = project_life




class Function(Placeholder):
  """
    Placeholder for values that are evaluated on the fly
  """
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
               or the path can be prepended with ``\%HERON_DATA\%'' to be relative to the HERON data directory (``HERON/data'' in the git repository, but the location of ``\%HERON_DATA\%'' can be overridden by exporting a ``HERON_DATA'' environmental variable),
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
    d = copy.deepcopy(dict((k, v) for k, v in self.__dict__.items() if k not in ['_module','_module_methods']))
    return d

  def __setstate__(self, d):
    """
      Deserialization.
      @ In, d, dict, object contents
      @ Out, None
    """
    self.__dict__ = d
    self._module = None
    self._module_methods = {}
    target_dir = os.path.dirname(os.path.abspath(self._target_file))
    if target_dir not in sys.path:
      sys.path.append(target_dir)
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
      raise RuntimeError(f'From Function "{self.name}" method "{method}" expected {self._source}.{method} ' +\
                         'to return with form (results_dict, meta_dict) with both as dictionaries, but received:\n' +\
                         f'    {result}')
    return result



class ROM(Placeholder):
  """
    Placeholder for values that are evaluated via a RAVEN ROM
  """
  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('ROM', contentType=InputTypes.StringType,
        ordered=False, baseNode=None,
        descr=r"""This data source is a trained RAVEN ROM to provide derived values.
              Variables within the dispatcher act as sources for the ROM inputs. The text
              of this node indicates the location of the serialized ROM. This location is usually
              relative with respect to the HERON XML input file; however, a full absolute path can
              be used, or the path can be prepended with ``\%HERON\%'' to be relative to the
              installation directory of HERON.""")
    specs.addParam('name', param_type=InputTypes.StringType, required=True,
        descr=r"""identifier for this data source in HERON and in the HERON input file. """)
    return specs

  def __init__(self, **kwargs):
    """
      Constructor.
      @ In, kwargs, dict, passthrough args
      @ Out, None
    """
    super().__init__(**kwargs)
    self._type = 'ROM'
    self._rom = None          # actual unpickled instance of ROM
    self._rom_location = None # string path to ROM

  def read_input(self, xml):
    """
      Sets settings from input file
      @ In, xml, xml.etree.ElementTree.Element, input from user
      @ Out, None
    """
    super().read_input(xml)

    self._runner = ROMLoader(self._target_file)
    # TODO is this serializable? or get/set state for this?

  def evaluate(self, rlz):
    """
      Evaluates requested method in stored module.
      @ In, rlz, dict, input realization as {input_name: value}
      @ Out, result, dict, results of evaluation
    """
    result = self._runner.evaluate(rlz)[0] # [0] because can batch evaluate, I think
    return result


class CSV(Placeholder):
  """
    Placeholder for values taken from a comma-separated file.
  """

  @classmethod
  def get_input_specs(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory(
      "CSV",
      contentType=InputTypes.StringType,
      ordered=False,
      baseNode=None,
      descr="""This data source is a static comma separated values (CSV) file.
               The text of this node indicates the location of the CSV file.
               This location is usually relative with respect to the HERON XML input file;
               however, a full absolute path can be used, or the path can be prepended
               with ``\%HERON\%'' to be relative to the installation directory of HERON.
               It is expected that variables contained in this file are defined as headers
               in the first row."""
    )

    specs.addParam(
      "name",
      param_type=InputTypes.StringType,
      required=True,
      descr="""identifier for this data source in HERON and in the HERON input file.""",
    )

    specs.addParam(
      'variable',
      param_type=InputTypes.StringListType,
      required=True,
      descr="""provides the names of the variables found in the CSV file
               and will be used in the workflow. Please note that all CSV files
               used in HERON for the purpose of input data must contain a variable
               titled ``RAVEN_sample_ID''. This variable can be a column of constant
               values (i.e., 0 or 1). This variable is unlikely to be used in the workflow
               but is required by RAVEN.""",
    )
    return specs

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, kwargs, dict, passthrough args
      @ Out, None
    """
    super().__init__(**kwargs)
    self._type = 'CSV'
    self._data = None
    self.eval_mode = "full"
    self.needs_multiyear = 1
    self.limit_interp = 1

  def read_input(self, xml):
    """
      Sets settings from input file.
      @ In, xml, xml.etree.ElementTree.Element, input from user.
      @ Out, None
    """
    specs = super().read_input(xml)
    self._var_names = specs.parameterValues['variable']
    with open(self._target_file, 'r', encoding='utf-8-sig') as f:
      headers = list(s.strip() for s in f.readline().split(','))
    for var in self._var_names:
      if var not in headers:
        self.raiseAnError(
          KeyError,
          f'Variable "{var}" requested for "{self.name}" but not found in "{self._target_file}"! Found: {headers}'
        )

  def checkValid(self, case, components, sources):
    """
      Check validity of placeholder given rest of system
      @ In, case, HERON.Case, case
      @ In, components, list(HERON.Component), components
      @ In, sources, list(HERON.Placeholder), sources
      @ Out, None
    """
    self.raiseAMessage(f'Checking CSV at "{self._target_file}"')
    structure = self.get_structure(case)
    interpolated = 'macro' in structure
    clustered = bool(structure['clusters'])
    # segmented = bool(structure['segments']) # TODO
    self.raiseAMessage(
        f'For DataGenerator <{self._type}> "{self.name}", detected: {"" if interpolated else "NOT"} interpolated, ' +
        f'{"" if clustered else "NOT"} clustered.'
    )
    # expect that project life == num macro years
    project_life = hutils.get_project_lifetime(case, components) - 1 # one less for construction year
    if interpolated:
      # if interpolated, needs more checking
      interp_years = structure['macro']['num']
      if interp_years >= project_life:
        self.raiseADebug(
            f'"{self.name}" interpolates {interp_years} macro steps, and project life is {project_life}, so histories will be trunctated.'
        )
        self.limit_interp = project_life
      else:
        self.raiseAnError(
            RuntimeError, f'"{self.name}" interpolates {interp_years} macro steps, but project life is {project_life}!'
        )
    else:
      # if single year, we can use multiyear so np
      self.raiseADebug(
          f'"{self.name}" will be extended to project life ({project_life}) macro steps using <Multicycle>.'
      )
      self.needs_multiyear = project_life

  @functools.cache
  def get_structure(self, case) -> dict:
    """
    Reads the CSV file to determine the structure of the time series samples it contains.
    @ In, None
    @ Out, structure, dict, the number of samples
    """
    # NOTE: The result is cached to avoid needing to read the CSV file repeatedly, and the caching is done here in the
    # Placeholder instead of the utility function so the caching is done on a per-file basis.
    return hutils.get_csv_structure(self._target_file, case.get_year_name(), case.get_time_name())

"""
  Class for managing interactions with the Dispatchers.
"""

import os
import sys
import pickle as pk

# set up path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _utils as hutils
import SerializationManager

raven_path = hutils.get_raven_loc()
sys.path.append(raven_path)
from utils import xmlUtils
sys.path.pop()

cashflow_path = hutils.get_cashflow_loc(raven_path=raven_path)

class DispatchRunner:
  """
    Manages the interface between RAVEN and running the dispatch
  """
  # TODO move naming templates to a common place for consistency!
  naming_template = {'comp capacity': '{comp}_capacity',
                     'dispatch var': 'Dispatch__{comp}__{res}',
                    }

  def __init__(self):
    """
      Constructor. Note instances of this are tied to single ExternalModel runs.
      @ In, None
      @ Out, None
    """
    self._case = None              # HERON case
    self._components = None        # HERON components list
    self._sources = None           # HERON sources (placeholders) list

  def load_heron_lib(self, path):
    """
      Loads HERON objects from library file.
      @ In, path, str, path (including filename) to HERON library
      @ Out, None
    """
    case, components, sources = SerializationManager.load_heron_lib(path, retry=6)
    # arguments
    self._case = case              # HERON case
    self._components = components  # HERON components list
    self._sources = sources        # HERON sources (placeholders) list
    # derivative
    self._dispatcher = self._case.dispatcher

  def extract_variables(self, raven, raven_dict):
    """
      Extract variables from RAVEN and apply them to HERON objects
      @ In, raven, object, RAVEN external model object
      @ In, raven_dict, dict, RAVEN input dictionary
      @ Out, pass_vars, dict, variables to pass to dispatcher
    """
    pass_vars = {}
    history_structure = {}
    # investigate sources for required ARMA information
    for source in self._sources:
      if source.is_type('ARMA'):
        # get structure of ARMA
        vars_needed = source.get_variable()
        for v in vars_needed:
          pass_vars[v] = getattr(raven, v)


    # get the key to mapping RAVEN multidimensional variables
    if hasattr(raven, '_indexMap'):
      pass_vars['_indexMap'] = raven._indexMap[0] # 0 is only because of how RAVEN EnsembleModel handles variables
      # collect all indices # TODO limit to those needed by sources?
      for target, required_indices in pass_vars['_indexMap'].items():
        for index in (i for i in required_indices if i not in pass_vars):
          pass_vars[index] = getattr(raven, index)
    else:
      # FIXME index isn't always "time" ...
      time = getattr(raven, 'time', None)
      if time is not None:
        pass_vars['time'] = time

    # TODO magic keywords (e.g. verbosity, MAX_TIMES, MAX_YEARS, ONLY_DISPATCH, etc)
    # TODO other arbitrary constants, such as sampled values from Outer needed in Inner?
    # component capacities
    for comp in self._components:
      name = self.naming_template['comp capacity'].format(comp=comp.name)
      update_capacity = raven_dict.get(name) # TODO is this ever not provided?
      comp.set_capacity(update_capacity)
    # TODO other case, component properties
    # load ARMA signals
    for source in self._sources:
      if source.is_type('ARMA'):
        vars_needed = source.get_variable()
        for v in vars_needed:
          pass_vars[v] = getattr(raven, v)
    return pass_vars

  def run(self, raven_vars):
    """
      Runs dispatcher.
      @ In, raven_vars, dict, dictionary of variables from raven to pass through
      @ Out, None?
    """
    # build meta variable
    ## this will be passed to external functions
    heron_meta = {}
    heron_meta['Case'] = self._case
    heron_meta['Components'] = self._components
    heron_meta['Sources'] = self._sources
    heron_meta['RAVEN_vars'] = raven_vars
    # build indexer for components
    ## indexer is as {component: {res: index}} where index is a standardized index for tracking activity
    heron_meta['resource_indexer'] = dict((comp, dict((res, r) for r, res in enumerate(comp.get_resources()))) for comp in self._components)
    # store meta
    meta = {'HERON': heron_meta}

    # check the "signal" history length
    signal_shapes = {}
    for source in self._sources:
      var_names = source.get_variable()
      if var_names is None:
        var_names = []
      for name in var_names:
        if name not in signal_shapes:
          s = raven_vars[name].shape
          signal_shapes[name] = s
    if not all((shape == s) for shape in signal_shapes.values()):
      print('History shapes:', signal_shapes)
      print('Index Map:', raven_vars.get('_indexMap', None))
      raise IOError('Synthetic histories are not of consistent shape! See "History Shapes" and "Index Map" above!')

    # determine analysis structure
    structure = self._get_structure(raven_vars)
    # TODO sanity check history length
    if structure['clustered']:
      seg_type = 'Cluster'
      segs = range(len(structure['clustered']))
    elif structure['segmented']:
      seg_type = 'Segment'
      segs = range(len(structure['segmented']))
    else:
      seg_type = 'All'
      segs = range(1)
    # for each year ... ?
    project_life = hutils.get_project_lifetime(self._case, self._components)
    num_segs = len(segs)
    active_index = {}
    for year in range(project_life):
      print(f'DEBUGG Dispatching year number {year}/{project_life}:')
      active_index['year'] = year
      # for each segment/cluster ..
      for s, seg in enumerate(segs):
        print(f'DEBUGG ... Dispatching {seg_type} ({s}/{num_segs}):')
        active_index['division'] = s
        active_index['division_obj'] = seg
        meta['HERON']['active_index'] = active_index
        meta['HERON']['division'] = seg
        ## TODO create "time" variable?
        ## -> chop up raven_vars for sources to corresponding segment/cluster, year, and time
        dispatch = self._dispatcher.dispatch(self._case,
                                            self._components,
                                            self._sources,
                                            meta)
        print(f'DEBUGG ... ... {seg_type} {s}/{num_segs} dispatched!')
    # TODO collect data per year/cluster/etc
    return dispatch

  def save_variables(self, raven, dispatch):
    """ generates RAVEN-acceptable variables TODO """
    # TODO clustering, multiyear
    # TODO should this be a Runner method or separate?
    template = self.naming_template['dispatch var']
    for comp_name, data in dispatch.items():
      for resource, usage in data.items():
        name = template.format(comp=comp_name, res=resource)
        setattr(raven, name, usage)
        # TODO indexMap?

  def _get_structure(self, raven_vars):
    structure = {}
    for source in self._sources:
      # only need ARMA information, not Functions
      if not source.is_type('ARMA'):
        continue
      structure[source] = {}
      print('DEBUGG target:', source._target_file)
      obj = pk.load(open(source._target_file, 'rb'))
      print('DEBUGG loading ARMA:')
      print(obj)
      meta = obj.writeXML().getRoot()
      print(xmlUtils.prettify(meta))
      print('DEBUGG type:', type(meta))
      # Interpolated? -> changes where clustered/segmented info is
      interpolated = meta.find('InterpolatedMultiyearROM')
      structure[source]['interpolated'] = interpolated
      # Clustered? Segmented?
      clustered = list(meta.iter('ClusterROM'))
      structure[source]['clustered'] = clustered
      # segmented?
      segmented = list(meta.iter('SegmentROM'))
      structure[source]['segmented'] = segmented
      # multiyear? -> depends on Case, not ARMA
    # check consistency across sources -> clustering etc needs to be consistent
    # XXX TODO
    # once they're consistent, we can just take the first entry to represent all the structures
    return next(iter(structure.values()))




def run(raven, raven_dict):
  """
    # TODO split into dispatch manager class and dispatch runner external model
    API for external models.
    This is run as part of the INNER ensemble model, run after the synthetic history generation
    @ In, raven, object, RAVEN variables object
    @ In, raven_dict, dict, additional RAVEN information
  """
  path = os.path.join(os.getcwd(), '..', 'heron.lib') # TODO custom name?
  # build runner
  runner = DispatchRunner()
  # load library file
  runner.load_heron_lib(path)
  # load data from RAVEN
  raven_vars = runner.extract_variables(raven, raven_dict)
  # TODO clustering, multiyear, etc?
  dispatch = runner.run(raven_vars)
  runner.save_variables(raven, dispatch)


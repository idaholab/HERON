
"""
  pyomo-based dispatch strategy
"""

import pyomo.environ as pyo

from Dispatcher import Dispatcher

class Pyomo(Dispatcher):
  """
    Dispatches using rolling windows in Pyomo
  """

  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('Dispatcher', ordered=False, baseNode=None)
    # TODO specific for pyomo dispatcher
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'PyomoDispatcher'
    self._abstract_model = None

  # def read_input(self, inputs):

  def initialize(self, case, components, sources, **kwargs):
    """
      Initialize dispatcher properties.
      @ In, case, Case, HERON case instance
      @ In, components, list, HERON components
      @ In, sources, list, HERON sources
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    # create abstract model
    self._abstract_model = pyo.AbstractModel()
    ###
    # NOTES
    # objective -> sum of cash flows     # Case should know how to do this maybe?
    # variables -> dispatch of comps     # components should add these
    # constraints:
    #  -> component capacity             # components should add these
    #  -> component ramp rates           # components should add these
    #  -> component transfer functions   # components should add these
    #  -> conservation of resources      # combination of components
    #  -> demand/load met if present     # sources? Or is this a component "sink"
    #  -> fixed dispatch (supply/demand) # components should add these
    #  -> policy requirements            # ?? idk where these come from, TODO
    ## components
    ## sources
    ## parameters








#### OLD ####
def run(raven, raven_dict):
  """
    API to RAVEN external model.
    @ In, raven, object, variable-containing object
    @ In, raven_dict, dict, additional raven information
    @ Out, None
  """
  raise NotImplementedError

def main(case, components, sources, other):
  """
    Runs dispatch.
    TODO
  """
  # load components
  # path = os.path.join(os.path.getcwd(), '..', 'heron.lib')
  # case, components, sources, other = SerializationManager.load_heron_lib(path, retry=6)

  pyo_abstract = other['pyo_model']

  # set up cluster windows? Should be parent API
  # rolling window (or just cluster window?)
  # set up pyomo model
  pyo_concrete = create_pyomo_concrete(pyo_abstract, case, components, sources)
  # optimize


def create_pyomo_model(abstract, case, components, sources):
  """
    Generates a Pyomo model to be solved
    @ In, case, TODO
  """
  # TODO could HERON initially create the AbstractModel, and it just gets filled in here?


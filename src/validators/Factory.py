# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

from .ExampleValidator import Example
from .FARMValidators import FARM_Beta, FARM_Gamma_LTI, FARM_Gamma_FMU

known = {
    'Example': Example,
    'FARM_Beta': FARM_Beta,
    'FARM_Gamma_LTI': FARM_Gamma_LTI,
    'FARM_Gamma_FMU': FARM_Gamma_FMU,
    # ModelicaGoverner: TODO,
}

def get_class(typ):
  """
    Returns the requested dispatcher type.
    @ In, typ, str, name of one of the dispatchers
    @ Out, class, object, class object
  """
  return known.get(typ, None)

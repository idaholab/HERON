# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

from .ExampleValidator import Example

import os
import sys

# set up path of raven
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _utils as hutils
raven_path = hutils.get_raven_loc()
sys.path.append(raven_path)
sys.path.pop()


# set up path of farm
farm_loc = hutils.get_farm_loc(raven_path=raven_path)
if farm_loc is not None:
  farm_path = os.path.abspath(os.path.join(farm_loc))
  sys.path.append(farm_path)
  import FARM
  from FARM.src.FARMValidatorsForHeron import FARM_Beta, FARM_Gamma_LTI, FARM_Gamma_FMU

known = {
    'Example': Example,
    # ModelicaGoverner: TODO,
}

if farm_loc is not None:
  known['FARM_Beta'] = FARM_Beta
  known['FARM_Gamma_LTI'] = FARM_Gamma_LTI
  known['FARM_Gamma_FMU'] = FARM_Gamma_FMU

def get_class(typ):
  """
    Returns the requested dispatcher type.
    @ In, typ, str, name of one of the dispatchers
    @ Out, class, object, class object
  """
  return known.get(typ, None)

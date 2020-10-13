
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
from .pyomo_dispatch import Pyomo
from .CustomDispatcher import Custom

known = {
    'pyomo': Pyomo,
    'custom': Custom,
}

def get_class(typ):
  """
    Returns the requested dispatcher type.
    @ In, typ, str, name of one of the dispatchers
    @ Out, class, object, class object
  """
  return known.get(typ, None)

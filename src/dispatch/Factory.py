
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
from .pyomo_dispatch import Pyomo
from .CustomDispatcher import Custom
from .AbceDispatcher import Abce
known = {
    'pyomo': Pyomo,
    'custom': Custom,
    'abce':Abce
}

def get_class(typ):
  """
    Returns the requested dispatcher type.
    @ In, typ, str, name of one of the dispatchers
    @ Out, class, object, class object
  """
  return known.get(typ, None)

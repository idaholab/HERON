
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
from .pyomo_dispatch import Pyomo
from .CustomDispatcher import Custom
from .blackbox_dispatch import BlackBoxDispatcher

known = {
    'pyomo': Pyomo,
    'custom': Custom,
    'blackbox': BlackBoxDispatcher
}

def get_class(typ):
  """
    Returns the requested dispatcher type.
    @ In, typ, str, name of one of the dispatchers
    @ Out, class, object, class object
  """
  return known.get(typ, None)

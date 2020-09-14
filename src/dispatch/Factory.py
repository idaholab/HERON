
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
# from generic import Generic
from .Marginal import MARGINAL
# from custom import Custom
from .pyomo_dispatch import Pyomo

known = {
    #'generic': Generic,
    'marginal': MARGINAL,
    #'custom': Custom,
    'pyomo': Pyomo,
}

def get_class(typ):
  """
    Returns the requested dispatcher type.
    @ In, typ, str, name of one of the dispatchers
    @ Out, class, object, class object
  """
  return known.get(typ, None)

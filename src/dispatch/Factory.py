
# from generic import Generic
# from marginal import marginal
# from custom import Custom
from .pyomo_dispatch import Pyomo

known = {
    #'generic': Generic,
    #'marginal': Marginal,
    #'custom': Custom,
    'pyomo': Pyomo,
}

def get_class(typ):
  return known.get(typ, None)

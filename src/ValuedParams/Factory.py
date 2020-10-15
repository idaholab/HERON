
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

from utils import InputData, InputTypes

from .SyntheticHistory import SyntheticHistory
from .Function import Function
from .Constant import Constant
from .Linear import Linear
from .Variable import Variable
from .Parametric import Parametric

known = {
    'ARMA': SyntheticHistory,
    'Function': Function,
    'Constant': Constant,
    'Linear': Linear,
    'Variable': Variable,
    'Parametric': Parametric,
}

def get_class(typ):
  """
    Returns the requested dispatcher type.
    @ In, typ, str, name of one of the dispatchers
    @ Out, class, object, class object
  """
  return known.get(typ, None)

def get_valued_param_specs(name, disallowed=None):
  """
    Compiles ValuedParam specs, applying a custom-chosen name for the spec
    @ In, name, str, name to give specifications
    @ In, disallowed, list, optional, particular options to disallow (default none)
    @ Out, raven.utils.InputData, specs
  """
  if disallowed is None:
    disallowed = []
  description = rf"""This value can be taken from any \emph{{one}} of the sources (described below):
                  {', '.join(known.keys())}."""
  specs = InputData.parameterInputFactory(name, descr=description)
  for name, option in known.items():
    if name not in disallowed:
      specs.addSub(option.get_input_specs())
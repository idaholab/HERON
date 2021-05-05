# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  ValuedParams are the flexible input sources used in HERON. In some way
  they represent placeholder values to be evaluated at run time from a variety of sources,
  ranging from constants to synthetic histories to AI and others.
"""
# only base types here, use factory for implementation types
from .ValuedParam import ValuedParam
from .Parametric import Parametric

# provide easy name access to module
from .Factory import factory


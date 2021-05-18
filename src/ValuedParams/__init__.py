# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  ValuedParams are the flexible input sources used in HERON. In some way
  they represent placeholder values to be evaluated at run time from a variety of sources,
  ranging from constants to synthetic histories to AI and others.
"""
# only type references here, as needed
from .ValuedParam import ValuedParam
from .Parametric import Parametric
from .ROM import ROM

# provide easy name access to module
from .Factory import factory


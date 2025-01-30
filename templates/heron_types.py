# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Alias types of major HERON classes for easier type hinting

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-12-23
"""
from typing import TypeAlias

# load utils
from .imports import Case, Placeholder, Component, ValuedParam

HeronCase: TypeAlias = Case
Source: TypeAlias = Placeholder

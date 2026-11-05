# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Imports from ravenframework and HERON.src

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-12-23
"""
import sys
from pathlib import Path


# import HERON
sys.path.append(str(Path(__file__).parent.parent.parent))
import HERON.src._utils as hutils
sys.path.pop()

# the templates write RAVEN workflows, so RAVEN is required from here on
hutils.ensure_raven_on_path()
RAVEN_LOC = Path(hutils.get_raven_loc())

from HERON.src.base import Base
from HERON.src.Cases import Case
from HERON.src.Components import Component
from HERON.src.Placeholders import Placeholder
from HERON.src.ValuedParams import ValuedParam

# import needed ravenframework modules
from ravenframework.utils import xmlUtils
from ravenframework.InputTemplates.TemplateBaseClass import Template
from ravenframework.Distributions import returnInputParameter

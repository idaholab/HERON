
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
import os
import sys
import dill as pk

def load_heron_lib(path, retry=0):
  """
    Loads serialized heron file
    @ In, path, str, path to file to load from
    @ In, retry, int, number of re-attempts for finding file
    @ Out, case, Case, HERON case object (None if not found)
    @ Out, components, list, list of HERON component objects (None if not found)
    @ Out, sources, list, list of HERON source objects (None if not found)
  """
  found = False
  while not found:
    counter = 0
    try:
      with open(path, 'rb') as lib:
        external_funcs = os.path.abspath(os.path.join(os.path.curdir, '..'))
        sys.path.append(external_funcs)
        case, components, sources = pk.load(lib)
        found = True
    except FileNotFoundError:
      print('WARNING: "{n}" not yet found; waiting and retrying {r} times ...'
            .format(n=path, r=retry-counter))
      counter += 1
      if counter > retry:
        case, components, source = None, None, None
        break
  return case, components, sources

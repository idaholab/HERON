# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Registry that maps type names to classes, for the HERON entities chosen by input node name
  (e.g. ValuedParams and TransferFuncs). Replaces HERON's uses of RAVEN's EntityFactory.
"""
from HERON.src.base import Base


class EntityFactory(Base):
  """
    Maps type names to the classes registered under them. Unlike RAVEN's EntityFactory, it does not
    load RAVEN plugins for dotted type names (e.g. "MyPlugin.MyType").
  """
  def __init__(self, name=None, **kwargs):
    """
      Constructor.
      @ In, name, str, optional, name of the kind of entity made (e.g. 'ValuedParam'), used in messages
      @ In, kwargs, dict, passthrough arguments (e.g. messageHandler)
      @ Out, None
    """
    Base.__init__(self, **kwargs)
    self.name = name
    self._registeredTypes = {} # type name -> class

  def __repr__(self):
    """
      String representation.
      @ In, None
      @ Out, repr, str, string rep
    """
    return f'<HERON {self.name} Factory>'

  def registerType(self, name, obj):
    """
      Registers a class under a type name, replacing any class already registered under that name.
      @ In, name, str, name by which the class should be known
      @ In, obj, type, class
      @ Out, None
    """
    self._registeredTypes[name] = obj

  def knownTypes(self):
    """
      Returns the registered type names.
      @ In, None
      @ Out, knownTypes, dict_keys, registered type names, in registration order
    """
    return self._registeredTypes.keys()

  def returnClass(self, Type):
    """
      Returns the class registered under a type name.
      @ In, Type, str, requested type name
      @ Out, returnClass, type, registered class
    """
    if Type not in self._registeredTypes:
      self.raiseAnError(NameError, f'"{self.name}" module does not recognize type "{Type}"; '
                        f'known types are: {", ".join(self.knownTypes())}')
    return self._registeredTypes[Type]

  def returnInstance(self, Type, **kwargs):
    """
      Returns a new instance of the class registered under a type name.
      @ In, Type, str, requested type name
      @ In, kwargs, dict, keyword arguments for the class constructor
      @ Out, returnInstance, object, new instance
    """
    return self.returnClass(Type)(**kwargs)

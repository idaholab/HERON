# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Decorators
  @author: Jacob Bryan (@j-bryan)
  @date: 2024-12-11
"""
from typing import Any


def _coerce_to_list(value: Any) -> list[Any]:
  """
  Try to make value a list that makes sense. Special handling for strings is taken to avoid splitting a
  string into characters. If commas are present in the string, it is assumed the string has comma-delimited
  values.
  @ In, value, Any, the value to coerce
  @ Out, coerced, list[Any], the coerced value object
  """
  if not value:
    # empty Collections (list, set, tuple, str, dict, ...) or None are all falsy
    coerced = []
  elif isinstance(value, str):
    # strings: either a comma-delimited list of values in the string or just a single item
    if "," in value:
      coerced = [s.strip() for s in value.split(",")]
    else:
      coerced = [value]
  else:
    # Just try coercing to a list otherwise and hope for the best. This should work without throwing
    # an exception for anything iterable.
    coerced = list(value)
  return coerced


class ListWrapper(list):
  """
  A wrapper class which emulates a list (and subclasses list for duck typing) which interfaces with a property
  """
  def __init__(self, property_instance, obj):
    """
    Constructor
    @ In, property_instance, listproperty, a listproperty object
    @ In, obj, Any, some object that can be made to be list-like (see _coerce_to_list function)
    @ Out, None
    """
    self.property_instance = property_instance
    self.obj = obj

  def _get_list(self):
    """
    Private getter for wrapped list
    @ In, None
    @ Out, obj, list, obj as a list
    """
    return _coerce_to_list(self.property_instance.fget(self.obj))

  def _set_list(self, value):
    """
    Private setter for the obj list
    @ In, value, coercible to list, object to set obj
    @ Out, None
    """
    self.property_instance.fset(self.obj, _coerce_to_list(value))

  def __getitem__(self, index):
    """
    Get item from list
    @ In, index, int, index of item to get
    @ Out, item, Any, the list item
    """
    return self._get_list()[index]

  def __setitem__(self, index, value):
    """
    Get item in list
    @ In, index, int, index of item to set
    @ In, value, Any, the value to store at the list index
    @ Out, None
    """
    lst = self._get_list()
    lst[index] = value
    self._set_list(lst)

  def __delitem__(self, index):
    """
    Delete item from list
    @ In, index, int, the index of the item to delete
    @ Out, None
    """
    lst = self._get_list()
    del lst[index]
    self._set_list(lst)

  def append(self, obj):
    """
    Append to list
    @ In, obj, Any, the object to append
    @ Out, None
    """
    lst = self._get_list()
    lst.append(obj)
    self._set_list(lst)

  def extend(self, iterable):
    """
    Extend list with iterable
    @ In, iterable, Iterable, the iterable with which to extend the lsit
    @ Out, None
    """
    lst = self._get_list()
    lst.extend(iterable)
    self._set_list(lst)

  def insert(self, index, obj):
    """
    Insert an object into the list
    @ In, index, int, the index to insert at
    @ In, obj, Any, the object to insert
    @ Out, None
    """
    lst = self._get_list()
    lst.insert(index, obj)
    self._set_list(lst)

  def remove(self, value):
    """
    Remove an object from the list
    @ In, value, Any, the value to remove
    @ Out, None
    """
    lst = self._get_list()
    lst.remove(value)
    self._set_list(lst)

  def pop(self, index=-1):
    """
    Remove an object from the list by index and return the object
    @ In, index, int, optional, the index to pull
    @ Out, val, Any, the popped item
    """
    lst = self._get_list()
    val = lst.pop(index)
    self._set_list(lst)
    return val

  def clear(self):
    """
    Clear the list
    @ In, None
    @ Out, None
    """
    self._set_list([])

  def index(self, value):
    """
    Get the index of a value in the list
    @ In, value, Any, the value to find in the list
    @ Out, index, int, the index of value
    """
    return self._get_list().index(value)

  def count(self, value):
    """
    Count the number of occurrences of a value
    @ In, value, Any, the value to count
    @ Out, count, int, the number of occurrences
    """
    return self._get_list().count(value)

  def sort(self, *, key=None, reverse=False):
    """
    Sort the list
    @ In, key, Callable[[Any], Any], optional, a function of one argument that is used to extract
                                               a comparison key from the values of the list
    @ In, reverse, bool, optional, if the sort should be done in descending order
    @ Out, None
    """
    lst = self._get_list()
    lst.sort(key=key, reverse=reverse)
    self._set_list(lst)

  def reverse(self):
    """
    Reverse the list
    @ In, None
    @ Out, None
    """
    lst = self._get_list()
    lst.reverse()
    self._set_list(lst)

  def copy(self):
    """
    Make a copy of the list
    @ In, None
    @ Out, copy, list, the copy of the list
    """
    return self._get_list().copy()

  def __len__(self):
    """
    The length of the list
    @ In, None
    @ Out, length, int, the length
    """
    return len(self._get_list())

  def __iter__(self):
    """
    Get an iterator over the list
    @ In, None
    @ Out, iter, Iterable, an iterator for the list
    """
    return iter(self._get_list())

  def __repr__(self):
    """
    A representation of the list
    @ In, None
    @ Out, repr, the list representation
    """
    return repr(self._get_list())

  def __eq__(self, other):
    """
    Equality comparison
    @ In, other, list, list to compare to
    @ Out, equal, bool, if the lists are equal
    """
    return self._get_list() == other

  def __contains__(self, value):
    """
    Does list contain value?
    @ In, value, Any, value to look for
    @ Out, contains, bool, if the value is in the list
    """
    return value in self._get_list()


class listproperty:
  """
  A approximation of the built-in "property" function/decorator, with additional logic for getting/setting values
  which are lists (or more precisely, ListWrapper objects) in a way that allows for list operations (e.g. append,
  extend, insert) on the property.
  """
  def __init__(self, fget=None, fset=None, fdel=None, doc=None):
    """
    Constructor
    @ In, fget, Callable, optional, getter function
    @ In, fset, Callable, optional, setter function
    @ In, fdel, Callable, optional, deleter function
    @ In, doc, str, optional, a docstring description of the property
    """
    self.fget = fget
    self.fset = fset
    self.fdel = fdel
    if doc is None and fget is not None:
      doc = fget.__doc__
    self.__doc__ = doc

  def __set_name__(self, owner, name):
    """
    Set the name of the property
    @ In, owner, type, the class owning the property (unused, included for consistency with 'property' builtin)
    @ In, name, str, the name of the property
    @ Out, None
    """
    self.__name__ = name

  def __get__(self, obj, objtype=None):
    """
    Get the property value
    @ In, obj, Any, the instance from which the property is accessed
    @ In, objtype, type, optional, the type of the instance (unused, included for consistency with 'property' builtin)
    @ Out, value, Any, the value of the property
    """
    if obj is None:
      return self
    if self.fget is None:
      raise AttributeError("unreadable attribute")
    return ListWrapper(self, obj)

  def __set__(self, obj, value):
    """
    Set the property value
    @ In, obj, Any, the instance on which the property is set
    @ In, value, Any, the value to be set
    @ Out, None
    """
    if self.fset is None:
      raise AttributeError("can't set attribute")
    self.fset(obj, value)

  def __delete__(self, obj):
    """
    Delete the property value
    @ In, obj, Any, the instance from which the property is deleted
    @ Out, None
    """
    if self.fdel is None:
      raise AttributeError("can't delete attribute")
    self.fdel(obj)

  def getter(self, fget):
    """
    Set getter function for property
    @ In, fget, Callable, getter function
    @ Out, obj, self with set getter
    """
    return type(self)(fget, self.fset, self.fdel, self.__doc__)

  def setter(self, fset):
    """
    Set setter function for property
    @ In, fget, Callable, setter function
    @ Out, obj, self with set setter
    """
    return type(self)(self.fget, fset, self.fdel, self.__doc__)

  def deleter(self, fdel):
    """
    Set deleter function for property
    @ In, fdel, Callable, deleter function
    @ Out, obj, self with set deleter
    """
    return type(self)(self.fget, self.fset, fdel, self.__doc__)

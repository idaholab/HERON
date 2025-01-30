# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  A RavenSnippet class factory

  @author: Jacob Bryan (@j-bryan)
  @date: 2024-11-08
"""
import xml.etree.ElementTree as ET

from .base import RavenSnippet
from . import distributions

from ..imports import returnInputParameter


def get_all_subclasses(cls: type[RavenSnippet]) -> list[type[RavenSnippet]]:
  """
    Recursively collect all of the classes that are a subclass of cls
    @ In, cls, type[RavenSnippet], the class to retrieve sub-classes.
    @ Out, getAllSubclasses, list[type[RavenSnippet]], list of classes which subclass cls
  """
  return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in get_all_subclasses(s)]


class SnippetFactory:
  """ A factory for RavenSnippet classes, useful for converting XML to RavenSnippet objects """
  def __init__(self) -> None:
    """
    Constructor
    @ In, None
    @ Out, None
    """
    self.registered_classes = {}  # dict[str, RavenSnippet]
    # NOTE: Keys for registered classes are formatted like XPaths. This functionality isn't currently used,
    # but could be useful, e.g. for finding any XML nodes which match a certain snippet class.

  def register_snippet_class(self, *cls: type[RavenSnippet]) -> None:
    """
    Register a RavenSnippet subclass with the factory
    @ In, *cls, type[RavenSnippet], classes to register
    @ Out, None
    """
    for snip_cls in cls:
      # Can't put a snippet into XML if it doesn't have a defined tag.
      # This prevents the factory from registering base classes.
      if not snip_cls.tag:
        continue

      key = self._get_snippet_class_key(snip_cls)

      # What if the key is already in the dict (duplicate name)?
      if key in self.registered_classes and (existing_cls := self.registered_classes[key]) != snip_cls:
        raise ValueError("A key collision has occurred when registering a RavenSnippet class! "
                         f"Key: {key}, Class to add: {snip_cls}, Existing class: {existing_cls}")

      self.registered_classes[key] = snip_cls

  def register_all_subclasses(self, cls: type[RavenSnippet]) -> None:
    """
    Register all suclasses of a class with the factory
    @ In, cls, type[RavenSnippet], base class to register
    @ Out, None
    """
    self.register_snippet_class(*get_all_subclasses(cls))

  def from_xml(self, node: ET.Element) -> ET.Element:
    """
    Produce a RavenSnippet object of the correct class with identical XML to an existing node
    @ In, node, ET.Element, the existing XML node
    @ Out, snippet, ET.Element, the matching RavenSnippet object, if one is registered
    """
    # Find the registered class which matches the tag and required attributes
    key = self._get_node_key(node)
    try:
      cls = self.registered_classes[key]
    except KeyError:
      # Not a registered type, so just toss the node back to the caller?
      return node

    snippet = cls.from_xml(node)
    return snippet

  def has_registered_class(self, node: ET.Element) -> bool:
    """
    Does the node have a registered class associated with it?
    @ In, node, ET.Element, the node to check
    @ Out, is_registered, bool, has a matching registered class
    """
    return self._get_node_key(node) in self.registered_classes

  # NOTE: I tried to combine the below methods since they're so similar. However, it gave me more trouble than expected
  # because of sometimes having the class type instead of a class object. This makes it so many attributes are not
  # accessible since the class hasn't yet been instantiated. Keeping them separate made it easier to avoid errors.
  @staticmethod
  def _get_snippet_class_key(cls: type[RavenSnippet]) -> str:
    """
    Get the registry key for a snippet class
    @ In, cls, type[RavenSnippet], the snippet class
    @ Out, key, str, the registry key
    """
    key = f"{cls.tag}"
    # TODO: delineate snippet type by additional attributes?
    if cls.subtype is not None:
      key += f"[@subType='{cls.subtype}']"
    return key

  @staticmethod
  def _get_node_key(node: ET.Element) -> str:
    """
    Get the registry key for an XML node
    @ In, node, ET.Element, the XML node
    @ Out, key, str, the registry key
    """
    key = node.tag
    if subtype := node.get("subType", None):
      key += f"[@subType='{subtype}']"
    return key


# There are many allowable distributions, each of which have their own properties. Rather than manually create classes
# for those, we can read from the RAVEN distributions input specs and dynamically create RavenSnippet classes for those
# distributions.

dist_collection = returnInputParameter()
for sub in dist_collection.subs:
  # We create a new distribution class for every RAVEN distribution class in the RAVEN input spec, then register that
  # new class with the templates.snippets.distributions module so they can be imported as expected.
  dist_class = distributions.distribution_class_from_spec(sub)
  setattr(distributions, dist_class.tag, dist_class)

# Register all RavenSnippet subclasses with the SnippetFactory
factory = SnippetFactory()
factory.register_all_subclasses(RavenSnippet)

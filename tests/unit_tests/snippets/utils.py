"""
Utility methods for RavenSnippet unit tests
@author: Jacob Bryan (@j-bryan)
@date: 2024-12-11
"""
import xml.etree.ElementTree as ET


def is_subtree_matching(snippet: ET.Element, subs: dict[str, str | dict]) -> bool:
  """
  Determines if an XML tree matches a nested dict structure of tags and text values
  @ In, snippet, ET.Element, the XML to consider
  @ In, subs, dict[str, str | dict], the structure the snippet should match
  @ Out, matches, bool, if the snippet matches the dict
  """
  if set(sub.tag for sub in snippet) != set(subs.keys()):
    return False

  for sub in snippet:
    if isinstance(subs[sub.tag], dict):
      is_subtree_matching(sub, subs[sub.tag])
    elif subs[sub.tag] != sub.text:
      return False

  return True

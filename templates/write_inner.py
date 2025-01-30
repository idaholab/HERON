from typing import Any
import xml.etree.ElementTree as ET

def increment(item: str, d: int) -> str:
  """
  Increment the right value for strings like "{%d}_{%d}"
  @ In, item, str, the full string
  @ In, d, int, the right value to set
  @ Out, item, str, the incremented item
  """
  item = item.strip().rsplit("_", 1)[0] + f"_{d}"
  return item

def alias_to_xpath(alias: str) -> str:
  """
  Convert RAVEN alias syntax to an XPath
  @ In, alias, str, alias path
  @ Out, xpath, str, the equivalent xpath
  """
  # Split the alias path by '|'
  parts = alias.split("|")

  # Initialize an empty list to hold the formatted parts
  xpath_parts = []

  # Iterate over each part
  for part in parts:
    # Check if the part contains an attribute
    if '@' in part:
      # Split the part into element and attribute
      element, attribute = part.split("@")
      # Split the attribute into name and value
      attr_name, attr_value = attribute.split(":")
      # Format and append the part to the list
      xpath_parts.append(f"{element}[@{attr_name}='{attr_value}']")
    else:
      # If no attribute, just append the element
      xpath_parts.append(part)

  # Join all parts with '/' to form the final XPath
  xpath = "/".join(xpath_parts)
  return xpath

def modifyInput(root: ET.Element, mod_dict: dict[str, Any]) -> ET.Element:
  """
  Modify the inner workflow XML
  @ In, root, ET.Element, the XML of the inner workflow
  @ In, mod_dict, dict[str, Any], modifications to make to the workflow
  @ Out, root, ET.Element, the modified XML
  """
  # Modify the inner XML based on values passed in in the mod_dict dictionary. The keys of mod_dict are
  # paths which point to an element in the XML tree, and the text of that element is set to the key's value.
  for k, v in mod_dict.items():
    # The entries in mod_dict are given in a RAVEN-specific pathing format with with pipe-separated ('|')
    # portions of the XML path. Converting those RAVEN paths to XPaths lets us directly search for the
    # intended XML element in the tree.
    xpath = alias_to_xpath(k)
    root.find(xpath).text = str(v)
    # special handling for denoises
    if "denoises" in xpath and (limit_node := root.find('.//samplerInit/limit')) is not None:
      limit_node.text = str(v)
  return root

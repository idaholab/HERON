# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
A script to autoload information from HYBRID to HERON.

This script autoloads the needed economic information about a specific component or an element of the grid system from HYBRID text files to a HERON input xml file.
It takes the following command line argument:
1. HERON_input file: the initial HERON xml file (pre_heron_input) before loading any information from HYRBID
To run the script, use the following command:
python hybrid2heron_economic.py path/to/HERON/xml/file
for example, the command looks like this:
python hybrid2heron_economic.py pre_heron_input.xml
The inital HERON xml file (e.g. pre_heron_input.xml) must have:
1- "Components" node : <Components>
2- At least one "Component" subnode under the "Components" node:
  <Component name="component_name">
3- An empty "economics" node under the "component" node with "src" attribute. For example: <economics src="path/to/HYBRID/file"></economics>
"""

#####
# Section 0
# Importing libraries and modules

import os
import sys
import csv
import argparse
from xml.etree import ElementTree as ET

#Note: xm only uses prettify which is in both xmlUtils and convert_utils
try:
  from ravenframework.utils import xmlUtils as xm
  import HERON.src
  HERON_src_path = HERON.src.__path__[0]
except ModuleNotFoundError:
  # # Importing XML utility from RAVEN to make the output XML file looks pretty
  this_file = os.path.abspath(__file__)
  #Note, drop everything after last occurence of HERON, then add HERON/src
  HERON_src_path = "HERON".join(this_file.split("HERON")[:-1])+"HERON/src"
  sys.path.append(HERON_src_path)
  from _utils import get_raven_loc
  sys.path.append(get_raven_loc())
  from scripts.conversionScripts import convert_utils as xm

#####
# Section 1: HYBRID Information Extraction Function


def relevant_info_from_hybrid(filepath):
  """
    Extracts the HYBRID keywords, corresponding values and comments.
    @ In, filepath, string, the path of the HYBRID text file that contains the component economic information
    @ Out, all_hybrid_info, dict, the relevant information from the HYBRID text files which includes: variables and their values and comments plus warning messages if necessary
  """
  hybrid_vars, hybrid_values, hybrid_comments = [], [], []
  with open(filepath, encoding="utf8") as f:
    for line in f:
      # if it is not a comment line
      if not line.lstrip().startswith("#"):
        if "=" in line:  # extracting relevant lines from HYBRID files
          relevant_lines = ("".join(line.split("#")[0])).split("=")
          if len(line.split("#")) == 2:  # if comments exist
            the_comment = ("".join(line.split("#")[1])).rstrip()
          else:  # if no comments exist
            the_comment = ""
          if len(line.split("=")) == 2:
            if "#" not in line.split("=")[0]:
              variable_name = relevant_lines[0].strip()
              value = relevant_lines[1].strip()
          if variable_name in Hybrid_keywords:
            hybrid_vars.append(variable_name)
            hybrid_values.append(value)
            hybrid_comments.append(the_comment)
          else:
            print(
              "\n",
              f" The variable {variable_name} in the file '{filepath}' is irrelevant",
              )

  # The Hybrid keywords that, if present, require adding more variables to the HERON input XML file to make it look complete. Since these variables, are not provided by HYBRID, they are assigned temporary default values

  demanding_hybrid_keywords = list(set(hybrid_vars) & set(Required_if))
  extra_variables, extra_values, extra_comments = [], [], []
  for i in range(len(Hybrid_keywords)):
    # for a hybrid keyword that is not already provided by HYBRID but need to be inlucded to make the HERON input file look complete
    if Hybrid_keywords[i] not in hybrid_vars:
      if Required_if[i] in demanding_hybrid_keywords:
        extra_variables.append(Hybrid_keywords[i])
        extra_variable_value = default[i]
        extra_values.append(extra_variable_value)
        extra_comments.append(
          f"Warning: The value of '{Hybrid_keywords[i]}' is not provided by HYBRID. A default value is assigned instead. It needs to be reviewed or replaced"
          )
  all_hybrid_info = dict()
  total_hybrid_variables = hybrid_vars + extra_variables
  total_hybrid_values = hybrid_values + extra_values
  total_hybrid_comments = hybrid_comments + extra_comments
  all_hybrid_info["variables"] = total_hybrid_variables
  all_hybrid_info["values"] = total_hybrid_values
  all_hybrid_info["comments"] = total_hybrid_comments
  if extra_variables:
    all_hybrid_info["warning"] = (
      "\nWarning: The folllowing list of variables were not provided by the HYBRID file:"
      + f"'{filepath}' \nand are assigned default values. The user needs to review them"
      + f" \n{extra_variables}"
    )

  else:
    all_hybrid_info["warning"] = (
      "\n",
      f" All variables were provided by the HYBRID file '{filepath}' and no default values are assigned",
    )

  return all_hybrid_info

#####
# Section 2: A function to covert HYBRID keywords to HERON nodes
# Loading the information from the HYRBID files to the corresopnding nodes, subnodes, parameters at the HERON XML file


def heron_node_from_hybrid(economics_node, hybrid_var_list, value_list, comments_list):
  """
    Creates HERON nodes from HYBRID keywords.
    @ In, economics_node, xml.etree.ElementTree.Element, the "economics" node in the HERON input xml file
    @ In, HYBRID_var_list, list, a list of the relevant HYBRID variables in the text file
    @ In, value_list, list, a list of the relevant HYBRID variables' values in the text file
    @ In, comments_list, list, a list of the relevant HYBRID variables' comments in the text file
    @ Out, economics_node, xml.etree.ElementTree.Element, the "economics" node in the HERON input xml file after being filled with nodes that are constructed based on the HYBRID text files information
  """

  # Initializing lists for HYRBID, HERON keywords, parameters,.etc to load information from the HYBRID text files
  (
    node_or_param_file,
    node_file,
    subnode_file,
    sub_subnode_file,
    belong_to_same_node_file,
    node_index,
  ) = ([], [], [], [], [], [])
  for var in hybrid_var_list:
    (
      node_or_parameter,
      nodes,
      subnodes,
      sub_subnodes,
      belonging_to_same_node,
    ) = hybrid_heron_dict[var][0:5]
    node_or_param_file.append(node_or_parameter)
    node_file.append(nodes)
    subnode_file.append(subnodes)
    sub_subnode_file.append(sub_subnodes)
    belong_to_same_node_file.append(belonging_to_same_node)
    node_index.append(belonging_to_same_node.split(","))

  # Computing the number of HERON XML file nodes under the  "economics"
  maximum_node_index = max([int(item) for sublist in node_index for item in sublist])
  try:
    for node_number in range(maximum_node_index + 1):  # Iterate over the nodes
      main_nodes, parameters = [], []
      nodes_of_same_param, nodes_of_same_subnodes, nodes_of_same_sub_subnodes = [], [], []
      subnodes_of_same_node, sub_subnodes_of_same_node, subnode_of_same_sub_subnodes = [], [], []
      nodes_values, param_values, subnodes_values, subsubnodes_values = [], [], [], []
      nodes_comments, param_comments, subnodes_comments, subsubnodes_comments = [], [], [], []

      for i in range(len(hybrid_var_list)):  # Iterate over HYBRID variables
        # Iterate over correspodning HERON keywords of the same node
        if str(node_number) in belong_to_same_node_file[i].split(","):

          if node_file[i] and not subnode_file[i]:  # nodes with no subnodes
            if node_or_param_file[i] == "N":
              main_nodes.append(node_file[i])
              nodes_values.append(value_list[i])
              nodes_comments.append(comments_list[i])

          if node_file[i] and subnode_file[i]:  # nodes with subnodes
            if sub_subnode_file [i]:  # subsubnode under a subnode under a node
              if node_or_param_file[i] == "N":
                nodes_of_same_sub_subnodes.append(node_file[i])
                subnode_of_same_sub_subnodes.append(subnode_file[i])
                sub_subnodes_of_same_node.append(sub_subnode_file[i])
                subsubnodes_values.append(value_list[i])
                subsubnodes_comments.append(comments_list[i])

            if not sub_subnode_file[i]:
              if node_or_param_file[i] == "P":  # parameters
                nodes_of_same_param.append(node_file[i])
                parameters.append(subnode_file[i])
                param_values.append(value_list[i])
                param_comments.append(comments_list[i])

              if node_or_param_file[i] == "N":  # nodes not parameters
                nodes_of_same_subnodes.append(node_file[i])
                subnodes_of_same_node.append(subnode_file[i])
                subnodes_values.append(value_list[i])
                subnodes_comments.append(comments_list[i])

      if main_nodes:
        subnode0 = ET.SubElement(economics_node, main_nodes[0])
        subnode0.text = nodes_values[0]
        if nodes_comments[0] != "":
          subnode0.append(ET.Comment(nodes_comments[0]))

      param_dict = dict(zip(parameters, param_values))
      if nodes_of_same_param:
        subnode1 = ET.SubElement(
          economics_node, nodes_of_same_param[0], param_dict
          )
        for comment in param_comments:
          parameter_comment = comment
          subnode1.append(ET.Comment(parameter_comment))
      else:
        if nodes_of_same_subnodes:
          subnode1 = ET.SubElement(economics_node, nodes_of_same_subnodes[0])
        else:
          if nodes_of_same_sub_subnodes:
            subnode1 = ET.SubElement(
              economics_node, nodes_of_same_sub_subnodes[0]
              )

      for n in range(len(subnodes_of_same_node)):
        subnode2 = ET.SubElement(subnode1, subnodes_of_same_node[n])
        subnode2.text = str(subnodes_values[n])
        if subnodes_comments[n] != "":
          subnode2.append(ET.Comment(subnodes_comments[n]))
      for m in range(len(subnode_of_same_sub_subnodes)):
        subnode2 = ET.SubElement(subnode1, subnode_of_same_sub_subnodes[m])
        if subsubnodes_comments[m] != "":
          subnode2.append(ET.Comment(subsubnodes_comments[m]))
        subnode3 = ET.SubElement(subnode2, sub_subnodes_of_same_node[m])
        subnode3.text = str(subsubnodes_values[m])


  except IndexError:
    pass

  return economics_node

#####
# Section 3: Connecting HYBRID and HERON keywords
# Load the HYBRID keywords from the csv file
# If needed, these HYBRID keywords can be modified in the csv file
keywords_path = HERON_src_path + "/Hybrid2Heron/HYBRID_HERON_keywords.csv"
with open(keywords_path, "r", encoding="utf8") as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=",")
  next(csv_reader)
  Hybrid_keywords = []  # Expected keywords of HYRBID

  # Is this HERON keywors a node or a parementer in the HERON XML file?
  # N: node, P, parameter
  HERON_Node_or_Parameter = []

  # The location of the HERON node corresponding to the HYBRID keyword is identified by HERON_Node/HERON_subnode/HERON_subsubnode as follows:
  # The main nodes under the "economics" node such as the lifetime and the cashflows
  HERON_Node = []
  HERON_Subnode, HERON_subsubnode = [], []

  # list of subnodes or pararmeters that are attached to the same node
  belong_to_same_node = []

  # Variables that are required to be included, even if not provided by HYBRID, because of the presence of other variables (To make the HERON xml input file complete)
  Required_if = []

  # default values for variables that are not provided but must be included
  default = []

  for lines in csv_reader:
    Hybrid_keywords.append(lines[0])
    HERON_Node_or_Parameter.append(lines[1])
    HERON_Node.append(lines[2])
    HERON_Subnode.append(lines[3])
    HERON_subsubnode.append(lines[4])
    belong_to_same_node.append(lines[5])
    Required_if.append(lines[6])
    default.append(lines[7])

hybrid_heron_dict = {
  i: [j, k, l, m, n, o, p]
  for i, j, k, l, m, n, o, p in zip(
  Hybrid_keywords,
  HERON_Node_or_Parameter,
  HERON_Node,
  HERON_Subnode,
  HERON_subsubnode,
  belong_to_same_node,
  Required_if,
  default,
  )
}

#####
# Section 4
# Specifying user inputs and output file
if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Autoload the component economic information from HYBRID folder to HERON input XML file"
    )
    # Inputs from the user
  parser.add_argument("HERON_input_file", help="HERON XML input file path")
  args = parser.parse_args()
  # Output file where you get the new HYBRID xml file with the new information loaded from HYBRID
  output_file = "heron_input.xml"

####
# Section 5: Autoload the HERON XML input file uing the HYBRID information
# Reading the HERON xml file
# Finding the "Compnents" node and the "economics" node under each component to fill it from HYBRID text files

# The HERON XML file tree
HERON_inp_tree = ET.parse(args.HERON_input_file)
components_list = HERON_inp_tree.findall("Components")  # The "components" node
if not components_list:
  print("\n", "The 'Components' node is not found in the HERON input xml file")

# Searching for the node "Component"
for components in components_list:
  component = components.findall("Component")
  comp_list = []  # The list of compoenents
  for comp in component:
    comp_list.append(comp.attrib["name"])
    print(f"\nThe component '{comp.attrib['name']}' is found")
    for node in comp:
      if node.tag == "economics":
        print(
          f"The 'economics' node is found at the component '{comp.attrib['name']}'"
          )
        ECO_NODE_FOUND = "True"
        if node.attrib["src"]:
          print("The attribute 'src' is found at the 'economics' node")
          children_nodes = []
          for child in node:
            children_nodes.append(child)
          if children_nodes:
              print(
                "\n",
                f"Warning: The 'economics' node at the component '{comp.attrib['name']}' is not empty so it will be left unchanged",
                )
          else:
            print(
              f"The economic information for the component '{comp.attrib['name']}' is loaded from the file: '{node.attrib['src']}'"
              )
            # check whether the file path exists
            if os.path.exists(node.attrib["src"]):
              print(f"The file path '{node.attrib['src']}' exists")
              comp.remove(node)
              print(
              "The 'economics' node in the"
              f" component '{comp.attrib['name']}' is being modified",
              )

              # If HYBRID variables list is not empty, start filling the "economics" node
              if relevant_info_from_hybrid(node.attrib["src"])["variables"]:
                node_eco = ET.SubElement(comp, "economics")
                node_eco.append(
                  ET.Comment(
                    "This 'economics' subnode is created using information from the HYBRID simulations"
                    )
                  )
                inputs = relevant_info_from_hybrid(node.attrib["src"])
                input_vars, input_vals, input_comments = (
                  inputs["variables"],
                  inputs["values"],
                  inputs["comments"],
                )
                print(inputs["warning"])

                # Applying the Hybrid2Heron function
                heron_node_from_hybrid(
                  node_eco, input_vars, input_vals, input_comments
                  )

              else:
                print(
                  f"no relevant information were found at the file {node.attrib['src']}'"
                  )

            else:
              print(
                f"Warning: The file path '{node.attrib['src']}' does not exist"
                )

        else:
          print(
            "The attribute 'src' is not found at the 'economics' node so it will be left unchanged"
            )

    if not ECO_NODE_FOUND:
      print(f"The economics node is not found at the component{comp}")

  if not comp_list:
    print("No componenet names were found")

#####
# Section 6: Print the modified HERON input new file in a pretty RAVEN-preferred format
with open(output_file, "w", encoding="utf8") as out:
  print((xm.prettify(HERON_inp_tree)), file=out)

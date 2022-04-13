"""
  This script autoloads the needed economic information about a specific component or an element of the grid system from a HYBRID folder to a HERON input xml file.
  It takes the following command line arguments:
  1. HYBRID_input folder: the path to the HYRBID folder from which the information are loaded
  2. HERON_input file: the original HERON xml file before loading any information from HYRBID

  To run the script, use the following command:
  python hybrid2heron.py path/to/HYRBID/folder path/to/HYRBID/xml/file
  for example, the command looks like this:
  python hybrid2heron.py ~/projects/raven/plugins/HYBRID/Costs/ heron_input.xml 
"""

import os
import csv
import sys
from xml.etree import ElementTree as ET
from xml.dom import minidom as MD

# Inputs from the user
Hybrid_input_folder, HERON_input_file = sys.argv[1], sys.argv[2]
# Output file where you get the new HYBRID xml file with the new information loaded from HYBRID
output_file = "new_" + (os.path.basename(HERON_input_file))

# Load the HYBRID keywords from the csv file
# If needed, these HYBRID keywords can be modified in the csv file
with open("HYBRID_keywords.csv", "r", encoding="utf8") as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=",")
  next(csv_reader)
  Hybrid_keywords = []
  for lines in csv_reader:
    Hybrid_keywords.append(lines[0])

# Listing the HERON keywords corresponding to the HYBRID keywords in the same order
HERON_keywords = [
  "depreciate",
  "capex.type",
  "capex.inflation",
  "capex.mult_target",
  "capex.taxable",
  "capex",
  "scaling_factor_x",
  "fixed_OM.type",
  "fixed_OM.inflation",
  "fixed_OM.mult_target",
  "fixed_OM.taxable",
  "fixed_OM",
  "lifetime",
  "reference_driver",
  "var_OM.type",
  "var_OM.inflation",
  "var_OM.taxable",
  "var_OM",
  "var_OM.mult_target",
  ]

# The connection between the HYBRID and HERON keywords
Hybrid2Heron_Dict = dict(zip(Hybrid_keywords, HERON_keywords))

# Defining the cashflows and corresponding parameters
cashflow_types = ["capex", "fixed_OM", "var_OM"]
capex_parameters = [
  "capex.type",
  "capex.taxable",
  "capex.inflation",
  "capex.mult_target",
  ]
fixed_OM_parameters = [
  "fixed_OM.type",
  "fixed_OM.taxable",
  "fixed_OM.inflation",
  "fixed_OM.mult_target",
  ]
var_OM_parameters = [
  "var_OM.type",
  "var_OM.taxable",
  "var_OM.inflation",
  "var_OM.mult_target",
  ]
cashflow_parameters = capex_parameters + fixed_OM_parameters\
                      + var_OM_parameters

# Defining the default values for cashflow parameters if not obtained from Hybrid
capex_default = {
  "capex.type": "one-time",
  "capex.taxable": "True",
  "capex.inflation": "none",
  "capex.mult_target": "False",
  }
fixed_OM_default = {
  "fixed_OM.type": "repeating",
  "fixed_OM.taxable": "True",
  "fixed_OM.inflation": "none",
  "fixed_OM.mult_target": "False",
  }
var_OM_default = {
  "var_OM.type": "repeating",
  "var_OM.taxable": "True",
  "var_OM.inflation": "none",
  "var_OM.mult_target": "False",
  }
cashflow_default = {**capex_default, **fixed_OM_default, **var_OM_default}


"""
  xml_node1 and xml_node1 are two functions to create two types of HERON input XML file nodes out of the available Hybrid keywords
"""
def xml_node1(keyword, parent_node):
  """xml_node1 is applicable to the component lifetime and depreciate nodes"""
  if keyword in eco_val_dict: # if keywords exist
    any_subnode = ET.SubElement(parent_node, keyword)
    any_subnode.text = eco_val_dict[keyword]
    subnode_comment = eco_comment_dict[keyword]
  else:
    subnode_comment = f"{keyword} is not provided by Hybrid"
  if subnode_comment != "": # If the comment is not empty
    any_subnode.append(ET.Comment(subnode_comment))
  return any_subnode, any_subnode.text


def xml_node2(keyword, parent_node, node_name):
  """xml_node2 is applicable to reference price, refernce driver, scaling factor"""
  if keyword in eco_val_dict:
    any_subnode1 = ET.SubElement(parent_node, node_name)
    any_subnode2 = ET.SubElement(any_subnode1, "fixed_value") # create a subnode called "fixed value"
    any_subnode2.text = eco_val_dict[keyword]
    any_subnode1_comment = eco_comment_dict[keyword]
  else:
    any_subnode1_comment = f"{keyword} is not provided by Hybrid"
  if any_subnode1_comment != (""):  # remove empty comments
    any_subnode1.append(ET.Comment(any_subnode1_comment))
  return any_subnode1, any_subnode2


def prettify(tree):
  """
    This function makes the xml file lookd prettier.It is taken from the
    xmlUtils file with minor changes"""
  pretty_output = MD.parseString(ET.tostring(tree)).toprettyxml(indent="  ")
  # loop over each "line" and toss empty ones, but for ending main nodes, insert a newline after.
  towrite = ""
  for the_line in pretty_output.split("\n"):
    if the_line.strip() == "":
      continue
    towrite += the_line.rstrip() + "\n"
    if the_line.startswith("  </"):
      towrite += "\n\n"
    if the_line.startswith("    </"):
      towrite += "\n"
  return towrite


"""
  Searching for the text files that contain the components economic information plus corresponding each file with a component text files we search for include .txt, .toml, .rtf files. 
  we search for files that contain at least one of the HYBRID keywords plus the string (.Economics)
"""

economic_information_file_names = []
economic_information_file_paths = []

for path, subdirs, files in os.walk(Hybrid_input_folder):
  for name in files:
    if name.endswith(".txt") or name.endswith(".rtf") or name.endswith(".toml"):
      file_path = os.path.join(path, name)
      file_name = os.path.splitext(name)[0]
      with open(file_path, encoding="utf8") as f:
        for i in range(len(Hybrid_keywords)):
          # The economic information file we search for are expected
          # to have these strings: ".Economics" , "=" plus (at
          # least) one of the keywords
          if Hybrid_keywords[i] and ".Economics" and "=" in f.read():
            economic_information_file_names.append(file_name)
            economic_information_file_paths.append(file_path)
          break
if economic_information_file_paths:
  print(
    "\n",
    "The components economic information are extracted from"
    " the following file(s):",
    *economic_information_file_paths,
    sep="\n",
  )
else:
  print(
    "\n",
    "The components' economic information are not found in any files"
    )
  


# Reading the HERON xml file
# Finding the "Compnents" node and the name of the componenet we are interested in
comp_list = []  # The list of compoenents
HERON_inp_tree = ET.ElementTree().parse(HERON_input_file)
components_list = HERON_inp_tree.findall("Components")

# Extracting the HYBRID keywords, corresponding values and comments from the
# HYBRID input file and asocciating these valeus and comments with HERON keywords
HERON_variables_names = []
HERON_variables_values = []
HERON_variables_comments = []
# Iterate through the components economic text files
for i in range(len(economic_information_file_paths)):
  with open(economic_information_file_paths[i], encoding="utf8") as f:
    eco_info_file_name = economic_information_file_names[i]
    for line in f:
      if "=" in line:
        relevant_lines = ("".join(line.split("#")[0])).split("=")
        if len(line.split("#")) == 2:  # if comments exist
          COMMENT = ("".join(line.split("#")[1])).rstrip()
        else:  # if no comments exist
          COMMENT = ""
        variable_name = relevant_lines[0].strip()
        value = relevant_lines[1].strip()
        if variable_name in Hybrid_keywords:
          HERON_variables_names.append(Hybrid2Heron_Dict.get(variable_name))
          HERON_variables_values.append(value)
          HERON_variables_comments.append(COMMENT)
        else:
          print(
            "\n",
            f" The variable {variable_name} in the file '{economic_information_file_names[i]}' is irrelevant"
            )

                

  # connect the HERON keywords with the HYBRID variables' values
  eco_val_dict = dict(zip(HERON_variables_names, HERON_variables_values))
  # connect the HERON keywords with the HYBRID variables' comments
  eco_comment_dict = dict(zip(HERON_variables_names, HERON_variables_comments))
  available_cashflows = list(set(cashflow_types) & set(HERON_variables_names))

  # Searching for available keywords that have to do with cashflows
  cashflow_param_values = []
  unavailable_cashflow_parameters = []
  available_cashflow_parameters = []

  for param in cashflow_parameters:
    if param in HERON_variables_names:
      cashflow_param_value = eco_val_dict.get(param)
      cashflow_param_values.append(cashflow_param_value)
      available_cashflow_parameters.append(param)
    else:
      cashflow_param_value = cashflow_default.get(param)
      cashflow_param_values.append(cashflow_param_value)
      unavailable_cashflow_parameters.append(param)
  cashflow_param_dic = dict(zip(cashflow_parameters, cashflow_param_values))

  if not components_list:
    print(
      "\n",
      "The 'Components' node is not found in the HERON input xml file"
      )

  # Searching for the node "Component"
  for components in components_list:
    component = components.findall("Component")
    for comp in component:
      comp_list.append(comp.attrib["name"])
      if comp.attrib["name"] == eco_info_file_name:  # If the component is found
        print(
          "\n",
          f"The component {eco_info_file_name} already exists in the HERON xml file",
        )
        # if the "economics" node is found in the component we are
        # interested in, it gets removed to be replaced with the new economics
        # node created from the HYBRID text file data
        for node in comp:
          if node.tag == "economics":
            print(
              "\n",
              f"The (economics) node exists in the component ({comp.attrib['name']})",
            )
            ECO_NODE_FOUND = "True"
            comp.remove(node)
            print(
              "\n",
              "The (economics) node is being replaced in the"
              f" component ({comp.attrib['name']})",
            )

        if ECO_NODE_FOUND != "True":
          print(
            "\n",
            "The economics node does not exist in the "
            f"component ({comp.attrib['name']})",
            )
          # Creating the XML component/economics node step by step
        if (eco_val_dict):
          # if the dictionary is not empty (i.e. some desired keywords exist)
          # create the "economics" node inside the component node
          node_eco = ET.SubElement(comp, "economics")

    if eco_info_file_name not in comp_list:
      comp_node = ET.SubElement(components, "Component", {"name": eco_info_file_name})
      print(
        "\n",
        f"A node is created for the component '{eco_info_file_name}' in the HERON xml file",
        )
      node_eco = ET.SubElement(comp_node, "economics")
      print(
        "\n", 
        f"The 'economics' node is created in the component '{eco_info_file_name}'",
        )

  # Creating the XML component/economics node step by step
  node_eco.append(ET.Comment
                  ("This 'economics' subnode is created using information from the HYBRID simulations" 
                   ))

  # First node to be created: component lifetime
  xml_node1("lifetime", node_eco)

  # Creating nodes and parameters for different types of cashflows
  if available_cashflows:
    for c in range(len(available_cashflows)):
      cashflow = available_cashflows[c]

      subnode_cashflow = ET.SubElement(node_eco,"CashFlow",
        {
          "name": cashflow,
          "type": (cashflow_param_dic.get(f"{cashflow}.type")),
          "taxable": (cashflow_param_dic.get(f"{cashflow}.taxable")),
          "inflation": (cashflow_param_dic.get(f"{cashflow}.inflation")),
          "mult_target": (cashflow_param_dic.get(f"{cashflow}.mult_target")),
        })

      parameters = [
        f"{cashflow}.type",
        f"{cashflow}.taxable",
        f"{cashflow}.inflation",
        f"{cashflow}.mult_target",
      ]

      WARNING_MSG = (
        "The following paramters are set to default values and not"
        " provided by HYBRID: "
        + str(list(set(unavailable_cashflow_parameters) & set(parameters)))
      )
      # Creating comments if exist
      for parameter in list(set(available_cashflow_parameters) & set(parameters)):
        print(parameter)
        comment_cash = eco_comment_dict[parameter]
        subnode_cashflow.append(ET.Comment(comment_cash))
      subnode_cashflow.append(ET.Comment(WARNING_MSG))

      # Creating the "driver" node as a placeholder for the user
      subnode_driver = ET.SubElement(subnode_cashflow, "driver")
      subnode_driver_variable = ET.SubElement(subnode_driver, "variable")
      subnode_driver_variable.text = "???"
      subnode_driver.append(ET.Comment(
        "The driver node is a " "placeholder and needs to be modified"
        ))

      # Creating the remaining nodes
      xml_node2(cashflow, subnode_cashflow, "reference_price")
      xml_node2("reference_driver", subnode_cashflow, "reference_driver")
      xml_node2("scaling_factor_x", subnode_cashflow, "scaling_factor_x")
      xml_node1("depreciate", subnode_cashflow)

  else:
    node_eco.append(ET.Comment("cashflows are not provided by Hybrid"))


# Print the modified HERON input new file
with open(output_file, "w", encoding="utf8") as f:
  print(prettify(HERON_inp_tree), file=f)
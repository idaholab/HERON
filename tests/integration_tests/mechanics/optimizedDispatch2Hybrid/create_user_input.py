# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
A script to create a user-input file for the HYBRID user.

This script is an initial step that the HYBRID user needs to create new names, if necessary, to: 1- The optimized dispatch outputs, 2- The components' capacities.
This script extracts the optimized dispatch outputs from the dispatch prints csv file and the list of components (and their capacities) from the HERON input XML file.

It takes the following arguments:
1- The HERON input XML file
2- The optimized dispatch outputs csv file.
For example, To run the script, use the following command:
    python create_user_input.py heron_input.xml dispatch_print.csv
The output will be a user input file: "user_input.txt".
The user needs to review the user input file and change/review the HYRBID variables and capacities there.

Next, the user can run the "export2Hybrid.py" which loads the dispatch outpus from HERON to a new file that HYBRID can use. The variables names in the autoloaded HYBRID file are borrowd from "user_input.txt"
"""

#####
# Section 0
# Importing libraries and modules

import os.path
import argparse
from xml.etree import ElementTree as ET
import pandas as pd
from colorama import Fore
from colorama import Style


#####
# Section 1: Extracting components' capacities Function
def extract_capacities_heron_hybrid(heron_input, output_file):
  """
  Creates a list of components' capacities from the HERON inpue file plus creating placeholders for the components' capacities in HYRBID
  @ In, heron input, str, the HERON input XML file
  @ In, output_file, str, the text file where the list of components capacities should be printed.
  @ Out, output_file, str, text a file autoloaded with the list of components' capacities from the HERON file and placeholders for the components' capacities in HYRBID
  """

  # Creating the HERON XML file tree
  HERON_inp_tree = ET.parse(heron_input)
  # Searching for the "components" node
  components_list = HERON_inp_tree.findall("Components")
  if not components_list:
    print("The 'Components' node is not found in the HERON input xml file")
  # Searching for the node "Component"
  for components in components_list:
    component = components.findall("Component")
    comp_list = []  # The list of compoenents

    # Printing the components' capacities in the output file
    with open(output_file, "a+") as u:
      u.write(
        "\n\n\n" + "# Below is a list of the components' capacities."
        + "\n" + "# Components' list is extracted from HERON input file: " + '"' + args.HERON_input_file + '"'
        + "\n" + "# The capacities are listed in the form:"
        + "\n" + "#" + "\t\t\t\t " + "HERON_component_capacity_1 = HYBRID_component_capacity_1" + "\n"
        + "# The user is expected to change/review the HYRBID components' capacities." + "\n"
      )
      # Components' capacities for HYBRID are also printed as placeholders until the user change them
      for comp in component:
        comp_list.append(comp.attrib["name"])
        u.write(
          comp.attrib["name"]
          + "_capacity = HYBRID_"
          + comp.attrib["name"]
          + "_capacity"
          + "\n"
        )
      print("\n")
      print(len(comp_list), "components are found in:", heron_input, "\n")
      print(
        "The capacities of",
        len(comp_list),
        "components are printed at the",
        output_file,
        "\n",
      )

    if not comp_list:
      print("No components found in the HERON input xml file")


#####
# Section 2: Extracting dispatch outputs Function
def extract_dispatches_heron_hybrid(dispatch_print, output_file):
  """
  Creates a list of dispatches (components' optimized variables) from the HERON inpue file plus creating placeholders for the dispatches in HYRBID
  @ In, dispatch_print, str, the distpach csv file with a list of optimized variables
  @ In, output_file, str, the text file where the list of optimized dispatches should be printed.
  @ Out, output_file, str, a text file autoloaded with the optimized dispatches plus placeholders for the HYBRID dispatches
  """
  input_dataset = pd.read_csv(dispatch_print)
  # Extracting the HERON csv dispatches
  colNames = input_dataset.columns[
    input_dataset.columns.str.contains(pat="Dispatch", case=False)
  ]
  # Printing the dispatches in the output file
  with open(user_input_file, "a+") as u:
    u.write(
      "\n\n"
      + "# Below are the HERON variables and their corresponding HYBRID variables" + "\n"
      + "# HERON variables are extracted from the HERON dispatch outputs file"
      + "\n" + "# HERON dispatch outputs file is: "
      + '"' + args.HERON_dipatch_csv_file
      + '"' + "\n" + "# The variables are listed in the form:"
      + "\n" + "#" + "\t\t\t\t" + "HERON_variable_1 = HYBRID_variable_1"
      + "\n" + "# The user is expected to change/review the HYRBID variables"
      + "\n"
    )

    for col in colNames:
      u.write(col + " = HYBRID_" + col)
      u.write("\n")
    print(
      len(list(colNames)),
      " optimized dispatches are found in:",
      dispatch_print,
      "and printed at the",
      output_file,
      "\n",
    )


#####
# Section 3: Specifying terminal command arguments & HYBRID user-input filename
if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Create a user-input file for the HYBRID user"
  )
  # Inputs from the user
  parser.add_argument("HERON_input_file", help="HERON XML input file path")
  parser.add_argument(
    "HERON_dipatch_csv_file", help="HERON optimized dispatch output csv file"
  )
  args = parser.parse_args()

# Output file: the user input
user_input_file = "user_input.txt"
# remove old user input if exists
file_exists = os.path.exists(user_input_file)
if file_exists:
  os.remove(user_input_file)

#####
# Section 4: Creating the HYBRID user-input file
with open(user_input_file, "a+") as u:
  u.write(
    "# The file that includes all the HYBRID variables." + "\n"
    "# All the capacities and dispatches should be a subset of this file's variables." "\n" + "# The user is expected to change/review the filename"
    + "\n" + "all_variables_file = "
    + '"' + "dsfinal.txt"+ '"'
  )

extract_capacities_heron_hybrid(args.HERON_input_file, user_input_file)
extract_dispatches_heron_hybrid(args.HERON_dipatch_csv_file, user_input_file)

print(
  f"{Fore.RED}The {user_input_file} file is created for the HYBRID user to change/review it{Style.RESET_ALL}",
  "\n",
)

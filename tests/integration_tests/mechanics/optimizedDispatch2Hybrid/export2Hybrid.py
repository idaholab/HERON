# Copyright 2022, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
A script to convert the HERON optimized components' variables (optimized dispatch) to a text file compatible with HYBRID.

The user-input file, "user_input.txt", must be in the same folder.
The user must review/modify the "user_input.txt" before running this script.

This script is run with the following arguments:
1- The HERON input XML file.
2- The optimized dispatch outputs CSV file.
For example, to run this script, use the following command:
    python export2Hybrid.py heron_input.xml Debug_Run_o/dispatch_print.csv
The output will be a text file compatible with HYBRID: "hybrid_compatible_dispatch.txt"
"""

#####
# Section 0
# Importing libraries and modules

import os.path
import argparse
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET

#####
# Section 1: A Function to connect HERON/HYBRID capacities in the user-input file

def map_capacities_in_the_user_input_file(user_input_file):
  """
    Identifies which component capacity in HYBRID corresponds to which component capacity in HERON.
    @ In, user_input_file, str, the user-input text file
    @ Out, comp_capacity, dict, a dictionary that matches HERON and HYBRID components capacities
  """
  # Iterating over the user-input file
  with open(user_input_file, "r") as f:
    comp_capacity = {}
    for line in f:
      if line.startswith("all_variables_file"):
        all_var_hybrid_file = line.split("=")[1].strip().replace('"', "")
      if not line.startswith("#"):
        # Connect HYBRID/HREON capacities
        if "capacity" in line:
          comp_name = (
            line.strip().split("=")[0].replace("_capacity", "").strip()
          )
          hybrid_capacity = line.strip().split("=")[1].strip()
          comp_capacity[comp_name] = hybrid_capacity
          if os.path.exists(all_var_hybrid_file): # checkin whether HYBRID variables file exists
            with open(all_var_hybrid_file) as hyb:
              if not hybrid_capacity in hyb.read():
                print('\033[91m', f" Warning: {hybrid_capacity} is not found in {all_var_hybrid_file}", '\033[0m')
          else:
            print ('\033[91m', f" Warning: {all_var_hybrid_file} is not found", '\033[0m')
  return (comp_capacity)

#####
# Section 2: A Function to connect HERON/HYBRID dispatches in the user-input file

def map_dispatches_in_the_user_input_file(user_input_file):
  """
    Identifies which component dispatch in HYBRID corresponds to which component dispatch in HERON.
    @ In, user_input_file, str, the user-input text file
    @ Out, heron_hybrid_dispatch, dict, a dictionary that matches HERON and HYBRID dispatches
  """
  # Iterating over the user-input file
  with open(user_input_file, "r") as f:
    heron_hybrid_dispatch = {}
    for line in f:
      if line.startswith("all_variables_file"):
        all_var_hybrid_file = line.split("=")[1].strip().replace('"', "")

      if not line.startswith("#"):
        # Connect HYBRID/HREON dispatches
        if "Dispatch" in line:
          heron_dispatch = line.strip().split("=")[0].strip()
          hybrid_dispatch = line.strip().split("=")[1].strip()
          heron_hybrid_dispatch[heron_dispatch] = hybrid_dispatch
          if os.path.exists(all_var_hybrid_file): # checkin whether HYBRID variables file exists
            with open(all_var_hybrid_file) as hyb:
              if not hybrid_dispatch in hyb.read():
                print('\033[91m', f" Warning: {hybrid_dispatch} is not found in {all_var_hybrid_file}",'\033[0m')
          else:
            print ('\033[91m', f" Warning: {all_var_hybrid_file} is not found", '\033[0m')

  return (heron_hybrid_dispatch)

#####
# Section 3: A Function to obtain the values of the components' capacities from the HERON input file

def get_capacities_from_heron(heron_input_file):
  """
    Extracts the values of the components' capacities from the HERON input XML file
    @ In, heron_input_file, str, the HERON input XML file
    @ Out, comp_capacites_values, dict, the values of the components' capacities
  """

  # The HERON XML file tree
  HERON_inp_tree = ET.parse(heron_input_file)
  # Searching for the "Components" node
  components_list = HERON_inp_tree.findall("Components")  # The "components" node
  if not components_list:
    print("\n", "The 'Components' node is not found in the HERON input xml file")
  # Searching for the node "Component"
  for components in components_list:
    component = components.findall("Component")
    comp_list = []  # The list of compoenents
    comp_capacites_values = {}
    for comp in component:
      comp_list.append(comp.attrib["name"])
      # Searching for the "capacity" node
      for activity in comp:
        resource_nodes = ["produces", "demands", "stores"]
        if activity.tag in resource_nodes:
          for node in activity:
            if node.tag == "capacity":
              # get the capacity value
              for subnode in node:
                if subnode.tag == "fixed_value":
                  comp_capacites_values[
                      comp.attrib["name"]
                  ] = subnode.text
                elif subnode.tag == "sweep_values":
                  debug_value = subnode.get("debug_value")
                  comp_capacites_values[
                    comp.attrib["name"]
                  ] = debug_value
  return (comp_capacites_values)


#####
# Section 4: # Section 4: A Function to get the most interesting dataset from the HERON optimized dispatch

def get_interesting_dispatch(dispatch_print_csv):
  """
    Extracts the most interesting dataset from the CSV file we get from HERON. This CSV file includes optimized dispatches calculated at different years and samples.
    @ In, dispatch_print_csv, str, a CSV file produced by HERON and includes optimized components' time-dependent variables at different years, different samples
    @ Out, interesting_dataset, pandas.core.frame.DataFrame, time-dependent optimized variables
  """

  # Reading the input csv file from HERON
  input_dataset = pd.read_csv(dispatch_print_csv)
  # Extracting the HERON csv file dataset that are relevant: (time, year, sample, components' resources)
  colNames = input_dataset.columns[
    input_dataset.columns.str.contains(pat="Sample|Year|Time|Dispatch", case=False)
  ]
  relevant_dataset = input_dataset[colNames]

  # The dataset is split into subsets. Each group is a different sample at a different year (The sample and year are indices)
  indexNames = relevant_dataset.columns[
    relevant_dataset.columns.str.contains(pat="Sample|Year", case=False)
  ]
  sample_values = relevant_dataset[indexNames[0]].unique()
  year_values = relevant_dataset[indexNames[1]].unique()
  relevant_dataset_slice = relevant_dataset.set_index(list(indexNames))
  relevant_dataset_slice = (
    relevant_dataset_slice.sort_index()
  )  # Sorting before indexing is recommended otherwise warning messages will be displayed.

  # Iterating over samples and years
  data_sets_max_ramping_rates = np.empty((0, 3), int)
  for sample in sample_values:
    for year in year_values:
      one_sample_one_year_dataset_slice = relevant_dataset_slice.loc[sample, year]
      # calculate the change of each variable over a timestep: abs(var(t) - var(t-1))
      variable_diff = abs(one_sample_one_year_dataset_slice.diff(axis=0).dropna())
      max_diff_rates = []
    for column in range(1, len(variable_diff.columns)):
      # max rate of change: divide the variable_diff by the change in timestep and divide it by the timestep, calculate the maximum
      max_diff_rate = max(
        (variable_diff.iloc[:, column] / variable_diff.iloc[:, 0])
      )  # this outputs the max_diff_rate for each variable
      max_diff_rates.append(
        max_diff_rate
      )  # max_diff_rates is a list of the max_diff_rate for each variable
    data_sets_max_ramping_rates = np.append(
      data_sets_max_ramping_rates,
      np.array([[sample, year, max((max_diff_rates))]]),
      axis=0,
    )  # for each dataset(sample, year), the maxium of the max_diff_rate is calculated
  max_ramping_rate_ever = max(data_sets_max_ramping_rates[:, -1])

  # find which dataset has the maximum ramping rate
  for row in data_sets_max_ramping_rates:
    if (row[-1]) == max_ramping_rate_ever:
        intersting_dataset_sample, intersting_dataset_year = row[0], row[1]
    break

  interesting_dataset = relevant_dataset_slice.loc[
    intersting_dataset_sample, intersting_dataset_year
  ]
  return interesting_dataset


#####
# Section 5: Specifying terminal command arguments

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Convert the HERON optimized components' variables (optimized dispatch) to a text file that is compatible with HYBRID"
  )
  # Inputs from the user
  parser.add_argument("HERON_input_file", help="HERON XML input file path")
  parser.add_argument(
    "HERON_dipatch_csv_file", help="HERON optimized dispatch output csv file"
  )
  args = parser.parse_args()

#####
# Section 6: Creating a text file to be used by the HYBRID user

# Check whether the user-input file exists
file_exists = os.path.exists("user_input.txt")
if file_exists:
  print("\n",
        "The user_input.txt file is found")
else:
  print("The user_input.txt file is not found")

# Check if an on old output file exists and remove it
output_file = "hybrid_compatible_dispatch.txt"
if os.path.exists(output_file):
  os.remove(output_file)

# Get capacities and dispatches
hybrid_heron_capacities = (
  map_capacities_in_the_user_input_file("user_input.txt")
)
hybrid_heron_dispatches = (
  map_dispatches_in_the_user_input_file("user_input.txt")
)

with open(output_file, "a+") as f:
  f.write("# A list of components' capacities" + "\n")
  capacities_dict = get_capacities_from_heron(args.HERON_input_file)
  for key in capacities_dict:
    f.write(hybrid_heron_capacities[str(key)] + " = " + capacities_dict[key] + "\n")
  f.write("\n\n" + "# Optimized dispatches" + "\n")
  interesting_scenario = get_interesting_dispatch(args.HERON_dipatch_csv_file)
  for col in range(1, len(interesting_scenario.columns)):
    hybrid_dispatch_name = hybrid_heron_dispatches[
      str(interesting_scenario.columns[col])
    ]
    f.write("double " + str(hybrid_dispatch_name))
    f.write("\n")
    np.savetxt(f, interesting_scenario.iloc[:, [0, col]], fmt="%10.9f")
    f.write("\n")

print(f"The {output_file} is created")

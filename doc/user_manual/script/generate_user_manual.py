"""
  Populates user manual with automatic documentation.
"""
import os

import collect_tex

build_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build'))
source_path = os.path.relpath(os.path.join(build_path, '..', 'src'), start=build_path)

# def difference_strings(list_1,list_2):
#   """
#     A function to compute difference of list of strings
#     @ In, list_1: First List
#     @ Out, list_2: Second List
#   """
#   c = set(list_1).union(set(list_2))
#   d = set(list_1).intersection(set(list_2))
#   return list(c-d)

print("Populating HERON_user_manual.tex...")

filenames = collect_tex.specs_to_load.keys()

string = ''
input_template = '\\input{{{name}}}\n'
# these are directly in the src
#for txt in ['Introduction', 'Howtorun']:
#  string += input_template.format(name=os.path.join(source_path, txt))
for txt in filenames:
  string += input_template.format(name=txt)

manual = os.path.join(build_path, 'HERON_user_manual.tex')
with open(manual, 'r') as file:
  document = file.read()
  document = document.replace('%INSERT_SECTIONS_HERE', string + '\n\\clearpage')
with open(manual, 'w') as file:
  file.write(document)

print("... HERON_user_manual.tex populated")










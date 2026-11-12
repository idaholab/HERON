
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Constructs documentation files for HERON.
"""
import os
import sys

heron_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
build_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build'))
src_path = os.path.abspath(os.path.join(build_path, '..', 'src'))

sys.path.append(heron_path)
from HERON.src import Cases, Components, Placeholders
from HERON.src.ValuedParams import distributions
from HERON.src.ValuedParams.RandomVariable import MANUAL_LABEL as uncertainty_label

specs_to_load = {'Cases': ['Case'],
                 'Components': ['Component'],
                 'Placeholders': ['ARMA', 'Function'],
}

# sections written by their own functions below, in the order they appear after specs_to_load
extra_sections = ['Uncertainty']

# Introductory sections from the .tex file #####

def read(texname):
  """
    A function to read the .tex file
    @ In, texname, str, LaTeX file name
    @ Out, filler_text, str, tex contents
  """
  with open(texname, 'r') as fd:
    filler_text = fd.read()
  return filler_text

def create(module, contents):
  """
    A function to create the .tex file by
    enquiring the methods in raven.
    @ In, module, str, name of module to get specs from
    @ In, contents, list(str), list of objects to inquire
    @ Out, None
  """
  out_name = f'{module}.tex'
  intro_name = f'{module.lower()}_intro.tex'
  intro_file = os.path.join(src_path, intro_name)
  if os.path.isfile(intro_file):
    intro = read(intro_file)
  else:
    intro = ''
  to_write = f'\\section{{{module}}}'
  to_write += intro
  to_write += '\n' * 2
  mod = getattr(sys.modules[__name__], module)
  for entry in contents:
    tex = getattr(mod, entry).get_input_specs()().generateLatex()
    to_write += tex + '\n' * 2
  with open(os.path.join(build_path, out_name), 'w') as f:
    f.writelines(to_write)

def create_uncertainty():
  """
    Creates the .tex file for the <uncertainty> section, which documents each supported distribution
    once. Every node that accepts <uncertainty> refers to this section by its label.
    @ In, None
    @ Out, None
  """
  to_write = f'\\section{{Uncertainty}}\n\\label{{{uncertainty_label}}}\n'
  to_write += read(os.path.join(src_path, 'uncertainty_intro.tex'))
  to_write += '\n\\subsection{Supported Distributions}\n'
  to_write += 'The \\xmlNode{uncertainty} node recognizes the following distributions:\n'
  to_write += '\\begin{itemize}\n'
  for dist in distributions.SUPPORTED:
    to_write += distributions.make_spec(dist).generateLatex(recDepth=1) + '\n'
  to_write += '\\end{itemize}\n'
  with open(os.path.join(build_path, 'Uncertainty.tex'), 'w') as f:
    f.writelines(to_write)

###create the tex files using the members in the list_of_files
for module_name, objects_list in specs_to_load.items():
  create(module_name, objects_list)
create_uncertainty()

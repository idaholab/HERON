#!/usr/bin/python
import sys,os
import shutil as sh
path=os.path.dirname(os.path.abspath(__file__))
path2=os.path.join(path,'../src')
sys.path.append(path2)
import Components, Cases, Economics
import _utils as hutils
framework_path = hutils.get_raven_loc()
sys.path.append(framework_path)
from utils import InputData

list_of_files = ['Components', 'Cases', 'Economics']

# Introductory sections from the .tex file #####

def read(texname):
  """ 
  A function to read the .tex file
  @ In, texname: LaTex file name 
  @ Out, filler_text: Text
  """

  with open(path+texname,'r') as fd:
    filler_text=fd.read()
  return filler_text

def create(module,cls,filename):
  """ 
  A function to create the .tex file by 
  enquiring the methods in raven.
  @ In, module: Name of the module to be enquired
  @ In, cls: Class name
  @ In, filename: Filename
  @ Out, Creates LaTex files
  """
  attribute = getattr(module,cls)
  stepSpec = attribute.get_input_specs()()
  tex = stepSpec.generateLatex()
  fname = str(filename) +'.tex'
  if fname == 'Cases.tex':
    filler_text_case = read('/CaseIn.tex')
    tex='\\section' + '{'+str(filename) +' ' + 'Introduction' +'}' +filler_text_case +'\n'*2 +tex
  if fname == 'Economics.tex':
    filler_text_economics = read('/EconIn.tex')
    tex='\\section' +'{' +str(filename) +' ' +'Introduction' +'}' +filler_text_economics +'\n'*2 +tex
  with open(fname, 'w') as f:
    f.writelines(tex)
  return
###create the tex files using the members in the list_of_files
for i, file_name in enumerate(list_of_files):
  #text_one has the file name and the file name has the 
  #class of the same name but without an 's'. Therefore text_two has the 
  #class name.#
  text_one = list_of_files[i]
  text_two = text_one[:-1]+''
  text_one = text_one.replace(" ' ", "")
  text_two = text_two.replace(" ' ", "")
  if text_one == 'Components':
    create(Components,text_two,list_of_files[i])
  elif text_one == 'Cases':
    create(Cases,text_two,list_of_files[i])
  elif text_one == 'Economics':
    create(Economics,'CashFlow',list_of_files[i])

### move files to src####
for i, file_name in enumerate(list_of_files):
  if not os.path.exists('src'):
    os.makedirs('src') 
  if os.path.exists('src'):
    sh.move(os.path.join(path,str(list_of_files[i])+'.tex'),os.path.join(path,'src'))




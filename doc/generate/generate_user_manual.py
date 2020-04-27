from string import punctuation
import glob
import os,sys
import Enquire
import shutil as sh
from distutils.dir_util import copy_tree
import re
import version 
path = os.path.dirname(os.path.abspath(__file__))
path_to_how = os.path.join(path,'..')

def difference_strings(list_1,list_2):
  """
    A function to compute difference of list of strings
    @ In, list_1: First List
    @ Out, list_2: Second List
  """
  c = set(list_1).union(set(list_2))
  d = set(list_1).intersection(set(list_2))
  return list(c-d)

print("Printing the HERON-Manual.tex...")

if not os.path.exists(os.path.join(path,'..','pdf')):
  print("This is true")
  os.makedirs(os.path.join(path,'..','pdf'))
sh.copy(os.path.join(path,'..','HERON_user_manual.tex'),os.path.join(path,'..','pdf','HERON_user_manual.tex'))
filenames = glob.glob('*.tex')
string=str()
for txt in filenames:
  string = string +'\\' +'input' +'{' + path + '/' +str(txt) +'}' +'\n'
string = '\\' +'input' +'{' + path_to_how + '/' + 'Howtorun' +'}' +'\n' +string
string = '\\' +'input' +'{' + path_to_how + '/' +'Introduction' +'}' +'\n'+string

with open(os.path.join(path,'..','pdf','HERON_user_manual.tex'),'r') as file:
  Document = file.read()
  Document = Document.replace('%\clearpage',string+'\clearpage')
with open(os.path.join(path,'..','pdf','HERON_user_manual.tex'),'w') as file:
  file.write(Document)
print("Printing tex file is done")










from string import punctuation
import glob
import os,sys
import Enquire
import shutil as sh
from distutils.dir_util import copy_tree
import re
import version 
path = os.path.dirname(os.path.abspath(__file__))
path_to_how = os.path.join(path,'..','..')
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
sh.copy(os.path.join(path,'..','..','HERON_user_manual.tex'),os.path.join(path,'HERON_user_manual.tex'))
filenames = Enquire.list_of_files
string=str()
for txt in filenames:
  string = string +'\\' +'input' +'{' + path + '/' +str(txt) +'}' +'\n'
string = '\\' +'input' +'{' + path_to_how + '/' + 'Howtorun.tex' +'}' +'\n' +string
string = '\\' +'input' +'{' + path_to_how + '/' +'Introduction.tex' +'}' +'\n'+string
with open(os.path.join(path,'HERON_user_manual.tex'),'r') as file:
  Document = file.read()
  Document = Document.replace('%\clearpage',string+'\clearpage')
with open(os.path.join(path,'HERON_user_manual.tex'),'w') as file:
  file.write(Document)
version.parse()
print("Printing tex file is done")










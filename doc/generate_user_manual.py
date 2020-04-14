from string import punctuation
import glob
import os,sys
import Enquire
from version import git_tag
import shutil as sh
from distutils.dir_util import copy_tree
import re
#Version = git_tag()
path = os.path.dirname(os.path.abspath(__file__))
path_to_src = os.path.join(path,'src')
print("Printing the HERON-Manual.tex...")
os.chdir(path_to_src)
filenames = glob.glob('*.tex')
os.chdir('..')
if not os.path.exists(os.path.join(path,'pdf')):
  os.makedirs(os.path.join(path,'pdf'))
def difference_strings(list_1,list_2):
  """
  A function to compute difference of list of strings
  @ In, list_1: First List
  @ Out, list_2: Second List
  """
  c = set(list_1).union(set(list_2))
  d = set(list_1).intersection(set(list_2))
  return list(c-d)
####Copy###
for i, file_name in enumerate(filenames):
  sh.copy(path_to_src+'/'+filenames[i],filenames[i])
###########
for i, file_name in enumerate(filenames):
  file_name= open(path_to_src+'/'+filenames[i],'r+')
  file_name_2 = open(os.path.join(path,'pdf/'+'pdf'+filenames[i]),'w')
  
  for line in file_name:
    file_name_2.write(line)
string=str()
filesall = glob.glob(os.path.basename(os.path.join(path,'*.tex')))
files_to_copy = difference_strings(filesall,filenames)
for txt in filenames:
  if txt != 'HERON_user_manual.tex':
    string = string +'\\' +'input' +'{' +'pdf' +str(txt) +'}' +'\n'
string = '\\' +'input' +'{' +'pdfHowtorun' +'}' +'\n' +string
string = '\\' +'input' +'{' +'pdfIntroduction' +'}' +'\n'+string
for files in files_to_copy:
  sh.copy(os.path.join(path,files),os.path.join(path,'pdf/'+'pdf'+files))
sh.copy(os.path.join(path,'INLreport.cls'),os.path.join(path,'pdf/'+'INLreport.cls'))
sh.copy(os.path.join(path,'raven_user_manual.toc'),os.path.join(path,'pdf/'+'raven_user_manual.toc'))
sh.copy(os.path.join(path,'HERON_user_manual.bib'),os.path.join(path,'pdf/'+'HERON_user_manual.bib'))
sh.copy(os.path.join(path,'HERON_user_manual.bbl'),os.path.join(path,'pdf/'+'HERON_user_manual.bbl'))
with open(os.path.join(path,'pdf/'+'pdfHERON_user_manual.tex'),'r') as file:
  Document = file.read()
  Document = Document.replace('%\clearpage',string+'\clearpage')
with open(os.path.join(path,'pdf/'+'pdfHERON_user_manual.tex'),'w') as file:
  file.write(Document)
if not os.path.exists('pics'):
  os.makedirs('pics')
copy_tree(path+'/pics',path+'/pdf/pics')
sh.rmtree(path+'/src')
print("Printing tex file is done")










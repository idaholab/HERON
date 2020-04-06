from string import punctuation
import glob
import os,sys
import Enquire
import shutil as sh
from distutils.dir_util import copy_tree
import re
path = os.path.dirname(os.path.abspath(__file__))
path_to_src = os.path.join(path,'src')
print("Printing the HERON-Manual.tex...")
os.chdir(path_to_src)
filenames = glob.glob('*.tex')
os.chdir('..')
if not os.path.exists(path+'/pdf'):
  os.makedirs(path+'/pdf')
####Copy###
for i, file_name in enumerate(filenames):
  sh.copy(path_to_src+'/'+filenames[i],filenames[i])
###########
for i, file_name in enumerate(filenames):
  file_name= open(path_to_src+'/'+filenames[i],'r+')
  file_name_2 = open(path+'/pdf/'+'pdf'+filenames[i],'w')
  
  for line in file_name:
    file_name_2.write(line)
string=str()
for txt in filenames:
  if txt != 'HERON_user_manual.tex':
    string = string +'\\' +'input' +'{' +'pdf' +str(txt) +'}' +'\n'
string = '\\' +'input' +'{' +'pdfHowtorun' +'}' +'\n' +string
string = '\\' +'input' +'{' +'pdfIntroduction' +'}' +'\n'+string
sh.copy(path+'/Introduction.tex',path+'/pdf/'+'pdfIntroduction.tex')
sh.copy(path+'/Howtorun.tex',path+'/pdf/'+'pdfHowtorun.tex')
sh.copy(path+'/HERON_user_manual.tex',path+'/pdf/'+'HERON_user_manual.tex')
sh.copy(path+'/INLreport.cls',path+'/pdf/'+'INLreport.cls')
sh.copy(path+'/raven_user_manual.toc',path+'/pdf/'+'raven_user_manual.toc')
sh.copy(path+'/HERON_user_manual.bib',path+'/pdf/'+'HERON_user_manual.bib')
sh.copy(path+'/HERON_user_manual.bbl',path+'/pdf/'+'HERON_user_manual.bbl')
with open(path+'/pdf/'+'HERON_user_manual.tex','r') as file:
  Document = file.read()
  Document = Document.replace('%\clearpage',string+'\clearpage')
with open(path+'/pdf/'+'HERON_user_manual.tex','w') as file:
  file.write(Document)
if not os.path.exists('pics'):
  os.makedirs('pics')
copy_tree(path+'/pics',path+'/pdf/pics')
sh.rmtree(path+'/src')
print("Printing tex file is done")










from string import punctuation
import glob
import os,sys
import time
import Enquire
import shutil as sh
from distutils.dir_util import copy_tree
import re
print("")
path=os.path.dirname(os.path.abspath(__file__))
path=path+'/../'
pathtosrc=path+'/src'

print("Printing the HERON-Manual...")


text= """ \\begin{document}
    \maketitle
    \SANDmain
    \clearpage
    \providecommand*{\phantomsection}{}
    \phantomsection
    \\addcontentsline{toc}{section}{References}
    \\bibliographystyle{ieeetr}
    \\bibliography{raven_user_manual}
    \end{document}"""

sys.path.append(pathtosrc)
filenames= glob.glob('*.tex')

os.chdir(path)

if not os.path.exists('pdf'):
    os.makedirs('pdf')
os.chdir('pdf')

####Copy###

for i in range(0,len(filenames)):
    sh.copy('../src/'+filenames[i],filenames[i])
###########
for i in range(0,len(filenames)):
    fd = open('../src/'+filenames[i],'r+')
    fd2 = open('pdf'+filenames[i],'w')
    

    for line in fd:
        fd2.write(line)


string=str()

for txt in filenames:

    if txt != 'HERON_user_manual.tex':


        string=string+'\\'+'input'+'{'+'pdf'+str(txt)+'}'+'\n'
#ttt
#####Added introduction too####
if not os.path.exists('pdfIntroduction.tex'):

    intro = open('pdfIntroduction.tex','w')
    intro2 = open('pdfHowtorun.tex','w')
    string = '\\'+'input'+'{'+'pdfHowtorun'+'}'+'\n'+string
    string = '\\'+'input'+'{'+'pdfIntroduction'+'}'+'\n'+string
    string = '\\'+'tableofcontents'+string

    intro.close()
    intro2.close()

##################################
sh.copy('../Introduction.tex','pdfIntroduction.tex')
sh.copy('../Howtorun.tex','pdfHowtorun.tex')
sh.copy('../HERON_user_manual.tex','HERON_user_manual.tex')
sh.copy('../INLreport.cls','INLreport.cls')
sh.copy('../raven_user_manual.toc','raven_user_manual.toc')
Docu = open('HERON_user_manual.tex','a')
idx = (text.find('\clearpage'))
text = (text[:idx])+string+text[idx:]

Docu.write(text)
Docu.close()

if not os.path.exists('pics'):
    os.makedirs('pics')
copy_tree(path+'/pics',path+'/pdf/pics')
os.system('pdflatex -interaction=batchmode HERON_user_manual.tex ') #>/dev/null')


###Remove the unwanted files####
for txt in filenames:

    if txt != 'HERON_user_manual.tex':
        os.remove(txt)
sh.rmtree(path+'pdf'+'/pics')
sh.rmtree(path+'__pycache__')
os.remove(path+'pdf'+'/pdfIntroduction.tex')
os.remove(path+'pdf'+'/pdfHowtorun.tex')
#################################
print("Printing is done")









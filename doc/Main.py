from string import punctuation
import glob
import os,sys
import time
import Enquire
import shutil as sh
from distutils.dir_util import copy_tree
import re
path=os.path.dirname(os.path.abspath(__file__))#'/Users/gaira/Desktop/Heron/HERON_user_Manual2'#os.path.dirname(__file__)
path=path+'/../'
#print(path)
#time.sleep(2000)
pathtosrc=path+'/src'
#print(pathtosrc)
#time.sleep(2000)



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
print(filenames)

os.chdir(path)





if not os.path.exists('pdf'):
    os.makedirs('pdf')
os.chdir('pdf')

print(os.getcwd())
#time.sleep(200)

####Copy###

for i in range(0,len(filenames)):
    sh.copy('../src/'+filenames[i],filenames[i])
###########
for i in range(0,len(filenames)):
    fd=open('../src/'+filenames[i],'r+')
    fd2=open('pdf'+filenames[i],'w')
    

    for line in fd:
        #print(re.search(r'_',line))
        #time.sleep(1)
        if re.search(r'_',line) == None:
            fd2.write(line)
        if re.search(r'_',line):
            #time.sleep(1)
            line2=line.replace("_",'\\'+'_')#(line.replace("_",'$'+'\\'+'_'+'$'))
            

            #print(re.search(r'{}',line))
            #print (line2)
            fd2.write(line2)
            #time.sleep(1)


##create a string to be appended###

string=str()

for txt in filenames:

    if txt!='HERON_user_manual.tex':

        string=string+'\\'+'input'+'{'+'pdf'+str(txt)+'}'+'\n'

#####Added introduction too####
if not os.path.exists('pdfIntroduction.tex'):
    #print('Not here')
    #time.sleep(2000)
    intro=open('pdfIntroduction.tex','w')
    intro.write('\\section'+'{'+'Introduction'+'}')
    string='\\'+'input'+'{'+'pdfIntroduction'+'}'+'\n'+string
    intro.close()
##################################



sh.copy('../HERON_user_manual.tex','HERON_user_manual.tex')
sh.copy('../INLreport.cls','INLreport.cls')
sh.copy('../raven_user_manual.toc','raven_user_manual.toc')
Docu=open('HERON_user_manual.tex','a')
idx=(text.find('\clearpage'))
text=(text[:idx])+string+text[idx:]
#print(text)
Docu.write(text)
Docu.close()

if not os.path.exists('pics'):
    os.makedirs('pics')
#path=os.path.dirname(__file__)
#print(path)
#time.sleep(2000)
copy_tree(path+'/pics',path+'/pdf/pics')
os.system('pdflatex HERON_user_manual.tex')

###Remove the unwanted files####
for txt in filenames:

    if txt!='HERON_user_manual.tex':
        os.remove(txt)

sh.rmtree(path+'pdf'+'/pics')
sh.rmtree(path+'__pycache__')
#################################









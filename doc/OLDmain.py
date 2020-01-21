from tex2py import tex2py
from string import punctuation
import glob
import os,sys
import time
import CreateLatex
import shutil as sh
from distutils.dir_util import copy_tree
path='/Users/gaira/Desktop/Heron/HERON_user_Manual2'#os.path.dirname(__file__)
print(path)
#sys.path.append(path+'/../src')
#import Components


#frameworkPath = "/Users/gaira/egret/raven/framework"#os.path.join(os.path.dirname(__file__), *(['..']*4), 'framework')
#sys.path.append(frameworkPath)

#from utils import InputData



####################################
# Test InputData creating LaTeX
#
# load libraries for all of RAVEN
#import Driver
# test MultiRun Step


# write tex



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

print(text)
#time.sleep(200)



def getText(section):
    
    for token in section.descendants:
        if isinstance(token, str):
            #print(str(token))

            Document.write(str(token))
            Document.write('\n')
def clean(x):
    line=[]
    for i in range(0,len(x)):
        corpus =open(x[i],'r')
        line.append( corpus.readline())
        corpus.close()
    return line
###Open File and start pasrsing to the sample tex files
def Tree(texfile):
    with open('../src/'+texfile) as f: data = f.read()
    toc = tex2py(data)
    print(str(toc.sections))
    return toc
###This file need to be built with Paul's programs

#time.sleep(200)
#path =path
pathtosrc=path+'/../src'
print(pathtosrc)
os.chdir(pathtosrc)
filestocreate=['Components.py','Cases.py','Economics.py']#glob.glob('*.py')
#print(filestocreate)
os.chdir(path)

for txt in filestocreate:
    temp=(txt.split('.py'))
    print(temp[0])
    #time.sleep(1)
    Doc=open(temp[0]+'.tex','w')
    temp2=temp[0]
    string='\section'+'{'+str(temp2)+'}'
    Doc.write(string)
    Doc.close()

#'/Users/gaira/Desktop/myHeron/egret/Docs/HERON_user_Manual2'#os.path.dirname(__file__)

#time.sleep(200)
sys.path.append(path)
filenames= glob.glob('*.tex')
print(filenames)


###Clean files
line=clean(filenames)
print(len(line),len(filenames))



if not os.path.exists('pdf'):
    os.makedirs('pdf')
os.chdir('pdf')


##create a string to be appended###

string=str()

for txt in filenames:

    if txt!='HERON_user_manual.tex':

        string=string+'\\'+'input'+'{'+str(txt)+'}'+'\n'





for i in range(0,len(line)):
    Document = open(filenames[i], 'a')
    print(filenames[i])

    if i==0:
        Document.seek(0)
        heading=line[i]
        Document.truncate()
    else:
        heading=line[i]
        Document.write(heading)
        #print(heading)
        #time.sleep(1)
    if filenames[i]!='HERON_user_manual.tex':
        toc=Tree(filenames[i])
        getText(toc) 

    if i==len(line)-1:
        #os.remove('__init__.tex')
        sh.copy('../HERON_user_manual.tex','HERON_user_manual.tex')
        sh.copy('../INLreport.cls','INLreport.cls')
        Docu=open('HERON_user_manual.tex','a')
        idx=(text.find('\clearpage'))
        text=(text[:idx])+string+text[idx:]
        print(text)
        Docu.write(text)
        Docu.close()

if not os.path.exists('pics'):
    os.makedirs('pics')
path=os.path.dirname(__file__)
print(path)
copy_tree(path+'../pics','/Users/gaira/Desktop/Heron/HERON_user_Manual2/pdf/pics')
os.system('pdflatex HERON_user_manual.tex')










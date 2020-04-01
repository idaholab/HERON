#!/usr/bin/python
import sys,os
### path to your raven folder ####
frameworkPath = "/Users/gaira/Optimizer/OpT/raven/framework"
sys.path.append(frameworkPath)
from utils import InputData
path=os.path.dirname(__file__)
path2=path+'/../src'
sys.path.append(path2)
import Components, Cases, Economics

List_of_files=['Components', 'Cases', 'Economics']


os.chdir(path+'/../doc')
if not os.path.exists('src'):
    os.makedirs('src')
os.chdir(path+'/../doc/src')
#### Read filler text #####

def read(fn):
    fd = os.open(path+fn, os.O_RDONLY) 
    fillertext = os.read(fd,10000)
    encoding ='utf-8'
    fillertext = str(fillertext,encoding)
    return fillertext

def create(x1,x2,x3):
        
        temp_3 = getattr(x1,x2)


        stepSpec = temp_3.get_input_specs()()

        tex = stepSpec.generateLatex()
        fName = str(x3)+'.tex'
        if fName == 'Cases.tex':
            fillertextc=read('/CaseIn.tex')
            tex='\\section'+'{'+str(x3)+' '+'Introduction'+'}'+fillertextc+'\n'*4+tex
        if fName == 'Economics.tex':
            fillertexte=read('/EconIn.tex')
            tex='\\section'+'{'+str(x3)+' '+'Introduction'+'}'+fillertexte+'\n'*4+tex

    
        with open(fName, 'w') as f:
            f.writelines(tex)

for i in range(0,len(List_of_files)):

    temp = List_of_files[i]
    temp2 = temp[:-1]+''
    temp = temp.replace(" ' ", "")
    temp2 = temp2.replace(" ' ", "")

    if temp == 'Components':
        create(Components,temp2,List_of_files[i])
    elif temp == 'Cases':
        create(Cases,temp2,List_of_files[i])
    elif temp == 'Economics':
        create(Economics,'CashFlow',List_of_files[i])


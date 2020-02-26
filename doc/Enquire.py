#!/usr/bin/python
import sys,os
import time
###Added Raven Path#####
"Will automate path later"
frameworkPath = "/Users/gaira/Desktop/myHeron/egret/raven/framework"#os.path.join(os.path.dirname(__file__), *(['..']*4), 'framework')
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

fillertext='Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'

def create(x,x2,x3):
        
        temp_3 = getattr(x,x2)#temp.temp2.get_input_specs()()
        stepSpec = temp_3.get_input_specs()()
        tex = stepSpec.generateLatex()
        fName = str(x3)+'.tex'
        tex='\\section'+'{'+str(x3)+' '+'Introduction'+'}'+fillertext+'\n'*4+tex
     
    
        with open(fName, 'w') as f:
            f.writelines(tex)
 
  

for i in range(0,len(List_of_files)):

    temp=List_of_files[i]
    temp2=temp[:-1]+''
    temp=temp.replace(" ' ", "")
    temp2=temp2.replace(" ' ", "")
  

    if temp=='Components':
        create(Components,temp2,List_of_files[i])
    elif temp=='Cases':
        create(Cases,temp2,List_of_files[i])
    elif temp=='Economics':
        create(Economics,'CashFlow',List_of_files[i])

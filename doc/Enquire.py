#!/usr/bin/python
import sys,os
import time
frameworkPath = "/Users/gaira/Desktop/myHeron/egret/raven/framework"#os.path.join(os.path.dirname(__file__), *(['..']*4), 'framework')
sys.path.append(frameworkPath)

from utils import InputData
path=os.path.dirname(__file__)
path2=path+'/../src'
#print(path)
#time.sleep(2000)
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
        #print(tex)
    
        with open(fName, 'w') as f:
            f.writelines(tex)
        #f=open(fName,'r+')
        #f.seek(0,0) 
        #f.write('\\section'+'{'+str(x3)+'Introduction'+'}'+'\n')
        #f.close()
  

for i in range(0,len(List_of_files)):

    temp=List_of_files[i]
    temp2=temp[:-1]+''
    temp=temp.replace(" ' ", "")
    temp2=temp2.replace(" ' ", "")
    #print(temp,temp2)

    if temp=='Components':
        create(Components,temp2,List_of_files[i])
    elif temp=='Cases':
        create(Cases,temp2,List_of_files[i])
    elif temp=='Economics':
        create(Economics,'CashFlow',List_of_files[i])
        #create(Economics,'CashFlowUser',List_of_files[i])


#  f.writelines(tex)
#git tag -a INL/EXT-01-00002 -m "my version2"#
from subprocess import PIPE, run
import os
def git_tag():
    a = run(["git","tag"],stdout=PIPE, stderr=PIPE, universal_newlines=True)
    a = a.stdout
    Latest_version = (a[len(a)-17:len(a)])
    return(Latest_version)
Version = str(git_tag())
with open(os.path.join('HERON_user_manual.tex'),'r') as file:
  Document = file.read()
  line = Document.find('\SANDnum')
  line =Document[line:line+26]
  Document = Document.replace(line,'\SANDnum'+'{'+Version+'}')
with open(os.path.join('HERON_user_manual.tex'),'w') as file:
  file.write(Document)
print("This is the latest version of HERON", Version)







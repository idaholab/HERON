import os
import shutil as sh
import glob
path=os.path.dirname(os.path.abspath(__file__))
file_to_remove=glob.glob(path+'/pdf'+'/*')
for file in file_to_remove:
  if file != path+"/pdf"+"/pdfHERON_user_manual.pdf":
    if file == path+"/pdf"+"/pics":
      sh.rmtree(path+"/pdf"+"/pics")
    elif file != path+"/pdf"+"/pics":
      os.remove(file)

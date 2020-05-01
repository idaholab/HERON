import shutil as sh
import os
path = os.path.dirname(os.path.abspath(__file__))
path_to_pdf = os.path.join(path,'pdf')
sh.copy(os.path.join(path,'HERON_user_manual.pdf'),os.path.join(path_to_pdf,'HERON_user_manual.pdf'))
os.remove("HERON_user_manual.pdf")
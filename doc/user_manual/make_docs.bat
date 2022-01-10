ECHO Starting to compile manual...
chmod u+x script/copy_tex.sh
cd script
copy_tex.sh
cd ..
python script/generate_user_manual.py
cd build
pdflatex -interaction=nonstopmode HERON_user_manual.tex
cd ..
bibtex build/HERON_user_manual
cd build
pdflatex -interaction=nonstopmode HERON_user_manual.tex
pdflatex -interaction=nonstopmode HERON_user_manual.tex
cd ..
cp -f build/HERON_user_manual.pdf pdf/
ECHO User manual build complete.
ECHO Cleaning build
cd ..
rm -rf build
ECHO Done


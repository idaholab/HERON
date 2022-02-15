ECHO Starting to compile manual...
IF EXIST build\ (
  ECHO build folder already exists
)
ELSE (
  ECHO creating build folder
  MD build
)
ECHO copying .tex files to build folder
cd ..
COPY user_manual\src\HERON_user_manual.tex %cd%\user_manual\build\HERON_user_manual.tex
COPY user_manual\src\HERON_user_manual.bib %cd%\user_manual\build\HERON_user_manual.bib
cd user_manual
python script/generate_user_manual.py
cd build
pdflatex -interaction=nonstopmode HERON_user_manual.tex
cd ..
bibtex build/HERON_user_manual
cd build
pdflatex -interaction=nonstopmode HERON_user_manual.tex
pdflatex -interaction=nonstopmode HERON_user_manual.tex
cd ..
ECHO moving user manual to pdf folder
COPY /Y build\HERON_user_manual.pdf pdf\HERON_user_manual.pdf
ECHO User manual build complete.
ECHO Cleaning build
RD /S/Q build
ECHO Done


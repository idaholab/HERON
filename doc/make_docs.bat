ECHO Starting to compile user manual...
cd user_manual
:: check if build folder exists and create if needed
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
:: run python script to generate .tex files
cd user_manual
python script/generate_user_manual.py
:: build pdf from .tex files
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
COPY /Y pdf\HERON_user_manual.pdf ..\pdfs\HERON_user_manual.pdf
ECHO User manual build complete.
ECHO Cleaning build
RD /S/Q build pdf
ECHO Done with user manual
cd ..
ECHO User manual can be found in %cd%\pdfs

ECHO Building Software Quality Assurance documents ...
cd sqa
ECHO %cd%
IF EXIST sqa_built_documents\ (
  ECHO sqa_built_documents folder already exists
)
ELSE (
  ECHO creating sqa_built_documents folder
  MD sqa_built_documents
)
:: copy Configuration Item documents
COPY /Y CIlist\*.docx sqa_built_documents\
:: build each document
ECHO ... building in sdd
cd sdd
pdflatex -interaction=nonstopmode heron_software_design_description.tex
cd ..
bibtex ssd/heron_software_design_description
cd sdd
pdflatex -interaction=nonstopmode heron_software_design_description.tex
pdflatex -interaction=nonstopmode heron_software_design_description.tex
:: copy to sqa_built_documents
COPY /Y *.pdf ..\sqa_built_documents\
:: clean up
DEL /Q *.aux *.log *.out *.pdf *.toc
cd ..
ECHO ... building in rtr
cd rtr
python createSQAtracebilityMatrix.py -i ..\srs\requirements_list.xml -o traceability_matrix.tex
pdflatex -interaction=nonstopmode heron_requirements_traceability_matrix.tex
pdflatex -interaction=nonstopmode heron_requirements_traceability_matrix.tex
pdflatex -interaction=nonstopmode heron_requirements_traceability_matrix.tex
:: copy to sqa_built_documents
COPY /Y *.pdf ..\sqa_built_documents\
:: clean up
DEL /Q *.aux *.log *.out *.pdf *.toc traceability_matrix.tex
cd ..
ECHO ... building in srs
cd srs
python readRequirementsAndCreateLatex.py -i requirements_list.xml -o requirements.tex
pdflatex -interaction=nonstopmode heron_software_requirements_specifications.tex
pdflatex -interaction=nonstopmode heron_software_requirements_specifications.tex
pdflatex -interaction=nonstopmode heron_software_requirements_specifications.tex
:: copy to sqa_built_documents
COPY /Y *.pdf ..\sqa_built_documents\
:: clean up
DEL /Q *.aux *.log *.out *.pdf *.toc requirements.tex
cd ..
ECHO ... building in srs_rtr_combined
cd srs_rtr_combined
python ..\srs\readRequirementsAndCreateLatex.py -i ..\srs\requirements_list.xml -o ..\srs\requirements.tex
python ..\rtr\createSQAtracebilityMatrix.py -i ..\srs\requirements_list.xml -o ..\rtr\traceability_matrix.tex
pdflatex -interaction=nonstopmode heron_software_requirements_specifications_and_traceability.tex
pdflatex -interaction=nonstopmode heron_software_requirements_specifications_and_traceability.tex
pdflatex -interaction=nonstopmode heron_software_requirements_specifications_and_traceability.tex
:: copy to sqa_built_documents
COPY /Y *.pdf ..\sqa_built_documents\
:: clean up
DEL /Q *.aux *.log *.out *.pdf *.toc ..\srs\requirements.tex ..\rtr\traceability_matrix.tex
PAUSE

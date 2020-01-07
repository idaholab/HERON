LATEX_FLAGS=-interaction=nonstopmode

all: heron_software_requirements_specifications.pdf

heron_software_requirements_specifications.pdf : heron_software_requirements_specifications.tex heron_software_requirements_specifications.bib intro.tex ../../version.tex ../sdd/heronDepAndLimitations.tex
	python readRequirementsAndCreateLatex.py -i requirements_list.xml -o requirements.tex 
	pdflatex $(LATEX_FLAGS) heron_software_requirements_specifications.tex
	pdflatex $(LATEX_FLAGS) heron_software_requirements_specifications.tex
	pdflatex $(LATEX_FLAGS) heron_software_requirements_specifications.tex

../../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../../version.tex

default: *.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof heron_software_requirements_specifications.pdf
LATEX_FLAGS=-interaction=nonstopmode

all: default heron_software_design_description.pdf

heron_software_design_description.pdf : heron_software_design_description.tex heron_software_design_description.bib heronDesignStakeHoldersAndConcerns.tex heronIntro.tex ../../version.tex heronSoftwareDesing.tex heronDepAndLimitations.tex otherDesignDocs.tex dataInterfaceSystemControl.tex
	pdflatex $(LATEX_FLAGS) heron_software_design_description.tex
	bibtex heron_software_design_description
	pdflatex $(LATEX_FLAGS) heron_software_design_description.tex
	pdflatex $(LATEX_FLAGS) heron_software_design_description.tex

../../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../../version.tex

default: *.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof heron_software_design_description.pdf
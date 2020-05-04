All: Run Create Manual_draft Clean Copy
Run:
	python generate_user_manual.py

Create:
	mkdir pdf
Manual_draft:
	 -latexmk -f -pdf -pdflatex="pdflatex -interaction=nonstopmode" HERON_user_manual.tex 
	#-pdflatex -interaction=nonstopmode HERON_user_manual.tex

Clean:
	latexmk -c

Copy:
	python Make_copy.py








	 
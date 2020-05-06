SRCFILE=HERON_user_manual
# SRC_FILES=../src/HERON_user_manual.tex ../src/HERON_user_manual.bib ../src/Howtorun.tex ../src/Introduction.tex

.PHONY: all build

all: copy populate build finish

copy:
	script/copy_tex.sh

populate:
	python script/generate_user_manual.py

build: ../version.tex
	cd build; pdflatex -interaction=nonstopmode $(SRCFILE).tex
	cp -f build/HERON_user_manual.pdf pdf/

finish:
	echo done

clean:
	rm -rf build pdf

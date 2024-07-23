#!/bin/bash

declare -a files=(heron_user_manual)

clean_files() {
  rm -rf build pdf
}

gen_files () {
  ./script/copy_tex.sh
  python script/generate_user_manual.py
  cd build
	for file in "${files[@]}"
	do
    pdflatex -interaction=nonstopmode $file.tex
    pdflatex -interaction=nonstopmode $file.tex
    pdflatex -interaction=nonstopmode $file.tex
	done
  cd ../
  cp -f build/HERON_user_manual.pdf pdf/


  # git log -1 --format="%H %an %aD" .. > ../version.tex
  # python ../../scripts/library_handler.py manual > libraries.tex
  # bash.exe ./create_command.sh
  # bash.exe ./create_pip_commands.sh
	# for file in "${files[@]}"
	# do
	# 	# Generate files.
  #       pdflatex -interaction=nonstopmode $file.tex
  #       bibtex $file
	# pdflatex -interaction=nonstopmode $file.tex
	# pdflatex -interaction=nonstopmode $file.tex

	# done
}

clean_files
gen_files

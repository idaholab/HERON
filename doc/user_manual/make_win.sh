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

}

clean_files
gen_files

#!/bin/bash
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

SCRIPT_NAME=`readlink $0`
if test -x "$SCRIPT_NAME";
then
    SCRIPT_DIRNAME=`dirname $SCRIPT_NAME`
else
    SCRIPT_DIRNAME=`dirname $0`
fi
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
cd $SCRIPT_DIR

VERB=0
for i in "$@"
do
  if [[ $i == "--verbose" ]]
  then
    VERB=1
    echo Entering verbose mode...
  fi
done

# clear old dir
if [[ 1 -eq $VERB ]]
then
  echo Removing old manuals ...
  rm -Rvf pdfs
  mkdir pdfs
else
  rm -Rvf pdfs > /dev/null
  mkdir pdfs > /dev/null
fi

# load raven libraries
if [[ 1 -eq $VERB ]]; then echo Loading RAVEN libraries ...; fi
SRC_DIR=`dirname $SCRIPT_DIR`/src
RAVEN_DIR=$(python $SRC_DIR/_utils.py get_raven_loc)
source $RAVEN_DIR/scripts/establish_conda_env.sh --load --quiet


# get new git version information
## TODO modify for tagged versions
if [[ 1 -eq $VERB ]]; then echo Loading HERON version information ...; fi
if git log -1 --format="%H %an\\\\%aD" . > /dev/null
then
    git log -1 --format="%H %an\\\\%aD" . > new_version.tex
    if [[ -f "version.tex" ]]
    then
      if diff new_version.tex version.tex > /dev/null
      then
          echo No change in version.tex
      else
          mv new_version.tex version.tex > /dev/null
      fi
    else
      mv new_version.tex version.tex > /dev/null
    fi
    if [[ -f "new_version.tex" ]]; then rm new_version.tex > /dev/null; fi
fi

# build manuals
echo ''
echo Building manuals ...
for DIR in user_manual; do
    cd $DIR
    echo ... building in $DIR...
    if [ "$(uname)" == "Darwin" ] || [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]
    then
      if [[ 1 -eq $VERB ]]
      then
        make; MADE=$?
      else
        make > /dev/null; MADE=$?
      fi
    elif [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ]  || [  "$(expr substr $(uname -s) 1 4)" == "MSYS" ]
    then
      if [[ 1 -eq $VERB ]]
      then
        bash.exe make_win.sh; MADE=$?
      else
        bash.exe make_win.sh > /dev/null; MADE=$?
      fi
    fi
    if [[ 0 -eq $MADE ]]; then
        echo .. ... successfully made docs in $DIR
        cp pdf/*pdf ../pdfs
        rm -rf build pdf
    else
        echo ... ... failed to make docs in $DIR
        exit -1
    fi
    cd $SCRIPT_DIR
done
echo Manuals complete. Documents can be found in $SCRIPT_DIR/pdfs

echo ''
echo Building Software Quality Assurance documents ...
cd sqa
./make_docs.sh; MADE=$?
if [[ 0 -eq $MADE ]]; then
    echo ... Successfully made SQA docs.
else
    echo ... Failed to make SQA docs.
    exit -1
fi
echo SQA documents complete. Documents can be found in $SCRIPT_DIR/sqa/sqa_built_documents

cd $SCRIPT_DIR
echo ''
echo done.

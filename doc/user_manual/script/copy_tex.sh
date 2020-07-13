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
if [[ ! -d "../build" ]]; then mkdir "../build"; fi
if [[ ! -d "../pdf" ]]; then mkdir "../pdf"; fi
cp ../src/HERON_user_manual.tex ../build/
cp ../src/HERON_user_manual.bib ../build/

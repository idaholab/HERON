#!/bin/bash
SCRIPT_DIRNAME=`dirname $0`
HERON_DIR=`(cd $SCRIPT_DIRNAME/..; pwd)`
cd $HERON_DIR
RAVEN_DIR=`python -c 'from src._utils import get_raven_loc; print(get_raven_loc())'`

source $HERON_DIR/coverage_scripts/initialize_coverage.sh

#coverage help run
SRC_DIR=`(cd src && pwd)`

# get display var
DISPLAY_VAR=`(echo $DISPLAY)`
# reset it
export DISPLAY=

export COVERAGE_RCFILE="$SRC_DIR/../coverage_scripts/.coveragerc"
SOURCE_DIRS=($SRC_DIR,$SRC_DIR/../templates/)
OMIT_FILES=($SRC_DIR/dispatch/twin_pyomo_test.py,$SRC_DIR/dispatch/twin_pyomo_test_rte.py,$SRC_DIR/dispatch/twin_pyomo_limited_ramp.py,$SRC_DIR/ArmaBypass.py)
EXTRA="--source=${SOURCE_DIRS[@]} --omit=${OMIT_FILES[@]} --parallel-mode "
export COVERAGE_FILE=`pwd`/.coverage

coverage erase
($RAVEN_DIR/run_tests "$@" --re=HERON/tests --python-command="coverage run $EXTRA " || echo run_tests done but some tests failed)

#get DISPLAY BACK
DISPLAY=$DISPLAY_VAR

## Prepare data and generate the html documents
coverage combine
coverage html

# See report_py_coverage.sh file for explanation of script separation
(bash $HERON_DIR/coverage_scripts/report_py_coverage.sh --data-file=$COVERAGE_FILE --coverage-rc-file=$COVERAGE_RCFILE)

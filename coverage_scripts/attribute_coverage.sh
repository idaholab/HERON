#!/bin/bash

# This script serves as a wrapper around the check_py_coverage.sh script, running coverage on tests individually
# and creating a coverage report that records which tests run each line

SCRIPT_DIRNAME=`dirname $(readlink -f "$0")`
HERON_DIR=`(cd $SCRIPT_DIRNAME/..; pwd)`
cd $HERON_DIR
RAVEN_DIR=`python -c 'from src._utils import get_raven_loc; print(get_raven_loc())'`

source $HERON_DIR/coverage_scripts/initialize_coverage.sh

SRC_DIR=`(cd src && pwd)`
export COVERAGE_RCFILE="$SRC_DIR/../coverage_scripts/.coveragerc"

# Get list of tests on which to run coverage
GCT_ARGS="--get-all-tests --get-test-names --tests-dir=$HERON_DIR/tests"
TEST_NAMES_STR=$(cd $RAVEN_DIR/developer_tools; python get_coverage_tests.py $GCT_ARGS)
IFS=' '
read -ra TEST_NAMES_ARR <<< "$TEST_NAMES_STR"

coverage erase

TEST_NUM=0
TEST_TOT=${#TEST_NAMES_ARR[@]}
for TEST_NAME in "${TEST_NAMES_ARR[@]}"
do
  ((TEST_NUM++))

  echo "Running command to check individual test coverage ($TEST_NUM/$TEST_TOT):"
  echo ./coverage_scripts/check_py_coverage.sh "$@" --coverage-run-only --re=\"$TEST_NAME\" --coverage-clargs=\"--context="$TEST_NAME"\"
  CPC_OUT=$(./coverage_scripts/check_py_coverage.sh "$@" --coverage-run-only --re="$TEST_NAME" --coverage-clargs="--context='$TEST_NAME'")
  if [[ $? -ne 0 ]]
  then
    echo "Failure in check_py_coverage run:"
    echo "$CPC_OUT"
  fi
done

coverage combine
coverage html --show-contexts

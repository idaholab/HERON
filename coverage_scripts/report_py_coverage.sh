#!/bin/bash

# To be run after a run of check_py_coverage.sh
# This script has been separated from check_py_coverage.sh for the github action, so that
#   the output of run_tests within check_py_coverage.sh can be printed, but the report
#   value can be caught and put in an annotation. This is necessary because calling
#   "coverage report --format=total" directly in the yaml does not work

SCRIPT_DIRNAME=`dirname $0`
HERON_DIR=`(cd $SCRIPT_DIRNAME/..; pwd)`
cd $HERON_DIR

source coverage_scripts/initialize_coverage.sh

# read command-line arguments
ARGS=()
for A in "$@"
do
  case $A in
    --data-file=*)
      export COVERAGE_FILE="${A#--data-file=}"  # Removes "--data-file=" and puts path into env variable
      ;;
    --coverage-rc-file=*)
      export COVERAGE_RCFILE="${A#--coverage-rc-file=}"  # See above
      ;;
    *)
      ARGS+=("$A")
      ;;
  esac
done

COV_VAL=`coverage report --format=total "${ARGS[@]}"`
if [[ $COV_VAL = "No data to report." ]]
then
  echo "Could not find data file with coverage results."
  exit 0
fi

echo "Coverage for this repository is now $COV_VAL%."

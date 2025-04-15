#!/bin/bash
SCRIPT_DIRNAME=`dirname $(readlink -f "$0")`
HERON_DIR=`(cd $SCRIPT_DIRNAME/..; pwd)`
cd $HERON_DIR
RAVEN_DIR=`python -c 'from src._utils import get_raven_loc; print(get_raven_loc())'`

# read command-line arguments
ARGS=()
COVERAGE_RUN_ONLY=0 # Default to running supporting coverage commands
export REGEX=`cd tests; pwd` # Default regex value for run_tests
if [[ "$REGEX" == "/c/"* ]] # The path should be in Windows format if it's a Windows path
then
  REGEX="C:${REGEX:2}"
  REGEX="${REGEX//\//\\}"
fi
for A in "$@"
do
  case $A in
    --coverage-run-only)
      export COVERAGE_RUN_ONLY=1 # Do not run supporting coverage commands
      ;;
    --re=*)
      export REGEX="${A#--re=}" # Removes "--re=" and puts regex value into env variable
      ;;
    --coverage-clargs=*)
      export COV_CLARGS="${A#--coverage-clargs=}"
      ;;
    *)
      ARGS+=("$A")
      ;;
  esac
done

if [ $COVERAGE_RUN_ONLY -eq 0 ]
then
  source $HERON_DIR/coverage_scripts/initialize_coverage.sh

  coverage erase
fi

#coverage help run
SRC_DIR=`(cd src && pwd)`
if [[ "$SRC_DIR" == "/c/"* ]] # It's a Windows path
then
  SRC_DIR="C:${SRC_DIR:2}" # coverage.py is picky about this for --source and --omit
fi

export COVERAGE_RCFILE="$SRC_DIR/../coverage_scripts/.coveragerc"
SOURCE_DIRS=($SRC_DIR,$SRC_DIR/../templates/)
OMIT_FILES=($SRC_DIR/dispatch/twin_pyomo_test.py,$SRC_DIR/dispatch/twin_pyomo_test_rte.py,$SRC_DIR/dispatch/twin_pyomo_limited_ramp.py,$SRC_DIR/ArmaBypass.py)
EXTRA="--source=${SOURCE_DIRS[@]} --omit=${OMIT_FILES[@]} --parallel-mode "
export COVERAGE_FILE=`pwd`/.coverage

($RAVEN_DIR/run_tests "${ARGS[@]}" --re="$REGEX" --python-command="coverage run $COV_CLARGS $EXTRA ")
TESTS_SUCCESS=$?

if [ $COVERAGE_RUN_ONLY -eq 0 ]
then
  ## Prepare data and generate the html documents
  coverage combine
  coverage html

  # See report_py_coverage.sh file for explanation of script separation
  (bash $HERON_DIR/coverage_scripts/report_py_coverage.sh --data-file=$COVERAGE_FILE --coverage-rc-file=$COVERAGE_RCFILE)
fi

if [ $TESTS_SUCCESS -ne 0 ]
then
  echo "run_tests finished but some tests failed"
fi

exit $TESTS_SUCCESS

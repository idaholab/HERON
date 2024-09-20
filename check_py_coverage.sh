#!/bin/bash
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
RAVEN_DIR=`python -c 'from src._utils import get_raven_loc; print(get_raven_loc())'`
source $RAVEN_DIR/scripts/establish_conda_env.sh --quiet --load
RAVEN_LIBS_PATH=`conda env list | awk -v rln="$RAVEN_LIBS_NAME" '$0 ~ rln {print $NF}'`
BUILD_DIR=${BUILD_DIR:=$RAVEN_LIBS_PATH/build}
INSTALL_DIR=${INSTALL_DIR:=$RAVEN_LIBS_PATH}
PYTHON_CMD=${PYTHON_CMD:=python}
JOBS=${JOBS:=1}
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
DOWNLOADER='curl -C - -L -O '

ORIGPYTHONPATH="$PYTHONPATH"

update_python_path ()
{
    if ls -d $INSTALL_DIR/lib/python*
    then
        export PYTHONPATH=`ls -d $INSTALL_DIR/lib/python*/site-packages/`:"$ORIGPYTHONPATH"
    fi
}

update_python_path
PATH=$INSTALL_DIR/bin:$PATH

if which coverage
then
    echo coverage already available, skipping building it.
else
    if curl http://www.energy.gov > /dev/null
    then
       echo Successfully got data from the internet
    else
       echo Could not connect to internet
    fi

    cd $BUILD_DIR
    #SHA256=56e448f051a201c5ebbaa86a5efd0ca90d327204d8b059ab25ad0f35fbfd79f1
    $DOWNLOADER https://files.pythonhosted.org/packages/ef/05/31553dc038667012853d0a248b57987d8d70b2d67ea885605f87bcb1baba/coverage-7.5.4.tar.gz
    tar -xvzf coverage-7.5.4.tar.gz
    cd coverage-7.5.4
    (unset CC CXX; $PYTHON_CMD setup.py install --prefix=$INSTALL_DIR)
fi

update_python_path

cd $SCRIPT_DIR

#coverage help run
SRC_DIR=`(cd src && pwd)`

# get display var
DISPLAY_VAR=`(echo $DISPLAY)`
# reset it
export DISPLAY=

export COVERAGE_RCFILE="$SRC_DIR/../tests/.coveragerc"
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
($SCRIPT_DIR/report_py_coverage.sh --data-file=$COVERAGE_FILE --coverage-rc-file=$COVERAGE_RCFILE)

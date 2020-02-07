#!/bin/bash

PYTHON=${PYTHON:-"python"}
HOME_DIR=$1
cd ./mmaction/ops/

echo "Building roi align op..."
cd ./roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building package resample2d"
cd ../resample2d_package
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building package trajectory_conv..."
cd ../trajectory_conv_package
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

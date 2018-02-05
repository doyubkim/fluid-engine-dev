#!/usr/bin/env bash

# Copyright (c) 2018 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.

GIT_COMMIT_ID=`git rev-parse HEAD`
GIT_COMMIT_ID_6=${GIT_COMMIT_ID:0:6}
if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR=.
fi
$BUILD_DIR/bin/time_perf_tests --benchmark_format=csv > time_perf_tests_$GIT_COMMIT_ID_6.csv

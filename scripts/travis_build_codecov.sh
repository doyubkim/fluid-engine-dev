#!/usr/bin/env bash

set -e

export NUM_JOBS=1

sudo apt-get install -yq lcov curl

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_COVERAGE=ON
make unit_tests
lcov -c -i -d src/tests/unit_tests -o base.info
bin/unit_tests
lcov -c -d src/tests/unit_tests -o test.info
lcov -a base.info -a test.info -o coverage.info
lcov -r coverage.info '/usr/*' -o coverage.info
lcov -r coverage.info '*/external/*' -o coverage.info
lcov -r coverage.info '*/src/tests/*' -o coverage.info
lcov -l coverage.info
bash <(curl -s https://codecov.io/bash) || echo "Codecov did not collect coverage reports"

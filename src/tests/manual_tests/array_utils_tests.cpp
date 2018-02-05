// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array_utils.h>
#include <jet/array2.h>
#include <jet/vector2.h>

using namespace jet;

JET_TESTS(ArrayUtils);

JET_BEGIN_TEST_F(ArrayUtils, ExtralateToRegion2) {
    Array2<double> data(128, 192, 0.0);
    Array2<char> valid(128, 192, 0);

    for (int j = 0; j < 192; ++j) {
        for (int i = 0; i < 128; ++i) {
            Vector2D pt(i / 128.0, j / 128.0);

            data(i, j) = std::sin(4 * kPiD * pt.x) * std::sin(4 * kPiD * pt.y);

            if (pt.distanceTo(Vector2D(0.5, 0.5)) < 0.15 ||
                pt.distanceTo(Vector2D(0.5, 0.9)) < 0.15) {
                valid(i, j) = 1;
            } else {
                valid(i, j) = 0;
            }
        }
    }

    saveData(data.constAccessor(), "data0.npy");
    saveData(valid.constAccessor(), "valid0.npy");

    extrapolateToRegion(
        data.constAccessor(), valid.constAccessor(), 10, data.accessor());

    saveData(data.constAccessor(), "data10.npy");
    saveData(valid.constAccessor(), "valid10.npy");
}
JET_END_TEST_F

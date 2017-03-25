// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/matrix.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Matrix, Constructors) {
    Matrix<double, 2, 3> mat;

    EXPECT_EQ(6u, mat.elements.size());

    for (double elem : mat.elements) {
        EXPECT_DOUBLE_EQ(0.0, elem);
    }
}

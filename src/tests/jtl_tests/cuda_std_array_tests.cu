// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_std_array.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(CudaStdArray, Constructors) {
    {
        CudaStdArray<int, 3> a;
        EXPECT_EQ(a[0], 0);
        EXPECT_EQ(a[1], 0);
        EXPECT_EQ(a[2], 0);
    }

    {
        CudaStdArray<int, 3> a(1, 2, 3);
        EXPECT_EQ(a[0], 1);
        EXPECT_EQ(a[1], 2);
        EXPECT_EQ(a[2], 3);
    }

    {
        std::array<int, 3> a = {1, 2, 3};
        CudaStdArray<int, 3> b(a);
        EXPECT_EQ(b[0], 1);
        EXPECT_EQ(b[1], 2);
        EXPECT_EQ(b[2], 3);
    }

    {
        jet::Vector<int, 3> a(1, 2, 3);
        CudaStdArray<int, 3> b(a);
        EXPECT_EQ(b[0], 1);
        EXPECT_EQ(b[1], 2);
        EXPECT_EQ(b[2], 3);
    }
}

TEST(CudaStdArray, Fill) {
    CudaStdArray<int, 3> a;
    a.fill(5);
    EXPECT_EQ(a[0], 5);
    EXPECT_EQ(a[1], 5);
    EXPECT_EQ(a[2], 5);
}

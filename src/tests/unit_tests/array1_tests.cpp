// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array.h>
#include <jet/parallel.h>
#include <jet/serialization.h>

#include <gtest/gtest.h>

#include <vector>

using namespace jet;

TEST(Array1, Serialization) {
    Array1<float> arr1 = {1.f, 2.f, 3.f, 4.f};

    // Serialize to in-memoery stream
    std::vector<uint8_t> buffer1;
    serialize<float>(arr1.view(), &buffer1);

    // Deserialize to non-zero array
    Array1<float> arr2 = {5.f, 6.f, 7.f};
    deserialize(buffer1, &arr2);
    EXPECT_EQ(4u, arr2.length());
    EXPECT_EQ(1.f, arr2[0]);
    EXPECT_EQ(2.f, arr2[1]);
    EXPECT_EQ(3.f, arr2[2]);
    EXPECT_EQ(4.f, arr2[3]);

    // Serialize zero-sized array
    Array1<float> arr3;
    serialize<float>(arr3.view(), &buffer1);

    // Deserialize to non-zero array
    deserialize(buffer1, &arr3);
    EXPECT_EQ(0u, arr3.length());
}

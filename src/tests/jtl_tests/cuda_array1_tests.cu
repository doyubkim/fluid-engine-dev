// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#if 0
#include <jet/_cuda_array.h>
#include <jet/_cuda_array_view.h>

#include <gtest/gtest.h>

using namespace jet;

namespace {

thrust::host_vector<float> makeVector(std::initializer_list<float> lst) {
    thrust::host_vector<float> vec;
    for (float v : lst) {
        vec.push_back(v);
    }
    return vec;
}

}  // namespace

TEST(NewCudaArray1, Constructors) {
    NewCudaArray1<float> arr0;
    EXPECT_EQ(0u, arr0.length());

    NewCudaArray1<float> arr1(9, 1.5f);
    EXPECT_EQ(9u, arr1.length());
    for (size_t i = 0; i < arr1.length(); ++i) {
        EXPECT_FLOAT_EQ(1.5f, arr1[i]);
    }

    Array1<float> arr9({1.0f, 2.0f, 3.0f});
    ArrayView1<float> view9(arr9);
    NewCudaArray1<float> arr10(view9);
    EXPECT_EQ(3u, arr10.length());
    for (size_t i = 0; i < arr10.length(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr10[i]);
    }
    NewCudaArray1<float> arr11(arr9.view());
    EXPECT_EQ(3u, arr11.length());
    for (size_t i = 0; i < arr11.length(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr11[i]);
    }

    NewCudaArray1<float> arr2(arr1.view());
    EXPECT_EQ(9u, arr2.length());
    for (size_t i = 0; i < arr2.length(); ++i) {
        EXPECT_FLOAT_EQ(1.5f, arr2[i]);
    }

    NewCudaArray1<float> arr3({1.0f, 2.0f, 3.0f});
    EXPECT_EQ(3u, arr3.length());
    for (size_t i = 0; i < arr3.length(); ++i) {
        float a = arr3[i];
        EXPECT_FLOAT_EQ(1.0f + i, arr3[i]);
    }

    NewCudaArray1<float> arr8(std::vector<float>{1.0f, 2.0f, 3.0f});
    EXPECT_EQ(3u, arr8.length());
    for (size_t i = 0; i < arr8.length(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr8[i]);
    }
    /*
        NewCudaArray1<float> arr4(makeVector({1.0f, 2.0f, 3.0f}));
        EXPECT_EQ(3u, arr4.length());
        for (size_t i = 0; i < arr4.length(); ++i) {
            EXPECT_FLOAT_EQ(1.0f + i, arr4[i]);
        }

        NewCudaArray1<float> arr5(
            thrust::device_vector<float>(makeVector({1.0f, 2.0f, 3.0f})));
        EXPECT_EQ(3u, arr5.length());
        for (size_t i = 0; i < arr5.length(); ++i) {
            EXPECT_FLOAT_EQ(1.0f + i, arr5[i]);
        }
    */
    NewCudaArray1<float> arr6(arr8);
    EXPECT_EQ(3u, arr6.length());
    for (size_t i = 0; i < arr8.length(); ++i) {
        EXPECT_FLOAT_EQ(arr8[i], arr6[i]);
    }

    NewCudaArray1<float> arr7 = std::move(arr6);
    EXPECT_EQ(3u, arr7.length());
    EXPECT_EQ(0u, arr6.length());
    for (size_t i = 0; i < arr6.length(); ++i) {
        EXPECT_FLOAT_EQ(arr6[i], arr7[i]);
    }
}

TEST(NewCudaArray1, Append) {
    // Cuda + scalar
    {
        NewCudaArray1<float> arr1({1.0f, 2.0f, 3.0f});
        arr1.append(4.0f);
        arr1.append(5.0f);
        EXPECT_EQ(5u, arr1.length());
        for (size_t i = 0; i < arr1.length(); ++i) {
            float a = arr1[i];
            EXPECT_FLOAT_EQ(1.0f + i, arr1[i]);
        }
    }

    // Cuda + Cuda
    {
        NewCudaArray1<float> arr1({1.0f, 2.0f, 3.0f});
        NewCudaArray1<float> arr2({4.0f, 5.0f});
        arr1.append(arr2);
        EXPECT_EQ(5u, arr1.length());
        for (size_t i = 0; i < arr1.length(); ++i) {
            float a = arr1[i];
            EXPECT_FLOAT_EQ(1.0f + i, arr1[i]);
        }
    }

    // Cuda + Cpu
    {
        NewCudaArray1<float> arr1({1.0f, 2.0f, 3.0f});
        Array1<float> arr2({4.0f, 5.0f});
        arr1.append(arr2);
        EXPECT_EQ(5u, arr1.length());
        for (size_t i = 0; i < arr1.length(); ++i) {
            float a = arr1[i];
            EXPECT_FLOAT_EQ(1.0f + i, arr1[i]);
        }
    }
}

TEST(NewCudaArray1, View) {
    NewCudaArray1<float> arr(15, 3.14f);
    NewCudaArrayView1<float> view = arr.view();
    EXPECT_EQ(15u, view.length());
    EXPECT_EQ(arr.data(), view.data());
    for (size_t i = 0; i < 15; ++i) {
        float val = arr.handle().ptr[i];
        EXPECT_FLOAT_EQ(3.14f, val);
    }
}
#endif

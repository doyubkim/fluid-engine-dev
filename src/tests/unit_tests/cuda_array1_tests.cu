// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#include <jet/cuda_array1.h>
#include <jet/cuda_array_view1.h>

#include <gtest/gtest.h>

using namespace jet;
using namespace experimental;

namespace {

thrust::host_vector<float> makeVector(std::initializer_list<float> lst) {
    thrust::host_vector<float> vec;
    for (float v : lst) {
        vec.push_back(v);
    }
    return vec;
}

}

TEST(CudaArray1, Constructors) {
    CudaArray1<float> arr0;
    EXPECT_EQ(0u, arr0.size());

    CudaArray1<float> arr1(9, 1.5f);
    EXPECT_EQ(9u, arr1.size());
    for (size_t i = 0; i < arr1.size(); ++i) {
        EXPECT_FLOAT_EQ(1.5f, arr1[i]);
    }

    Array1<float> arr9({1.0f, 2.0f, 3.0f});
    ArrayView1<float> view9(arr9);
    CudaArray1<float> arr10(view9);
    EXPECT_EQ(3u, arr10.size());
    for (size_t i = 0; i < arr10.size(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr10[i]);
    }
    CudaArray1<float> arr11(arr9);
    EXPECT_EQ(3u, arr11.size());
    for (size_t i = 0; i < arr11.size(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr11[i]);
    }

    CudaArray1<float> arr2(arr1.view());
    EXPECT_EQ(9u, arr2.size());
    for (size_t i = 0; i < arr2.size(); ++i) {
        EXPECT_FLOAT_EQ(1.5f, arr2[i]);
    }

    CudaArray1<float> arr3({ 1.0f, 2.0f, 3.0f });
    EXPECT_EQ(3u, arr3.size());
    for (size_t i = 0; i < arr3.size(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr3[i]);
    }

    CudaArray1<float> arr8(std::vector<float>{ 1.0f, 2.0f, 3.0f });
    EXPECT_EQ(3u, arr8.size());
    for (size_t i = 0; i < arr8.size(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr8[i]);
    }

    CudaArray1<float> arr4(makeVector({1.0f, 2.0f, 3.0f}));
    EXPECT_EQ(3u, arr4.size());
    for (size_t i = 0; i < arr4.size(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr4[i]);
    }

    CudaArray1<float> arr5(
        thrust::device_vector<float>(makeVector({ 1.0f, 2.0f, 3.0f })));
    EXPECT_EQ(3u, arr5.size());
    for (size_t i = 0; i < arr5.size(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr5[i]);
    }

    CudaArray1<float> arr6(arr5);
    EXPECT_EQ(3u, arr6.size());
    for (size_t i = 0; i < arr5.size(); ++i) {
        EXPECT_FLOAT_EQ(arr5[i], arr6[i]);
    }

    CudaArray1<float> arr7 = std::move(arr6);
    EXPECT_EQ(3u, arr7.size());
    EXPECT_EQ(0u, arr6.size());
    for (size_t i = 0; i < arr6.size(); ++i) {
        EXPECT_FLOAT_EQ(arr6[i], arr7[i]);
    }
}

TEST(CudaArray1, View) {
    CudaArray1<float> arr(15, 3.14f);
    CudaArrayView1<float> view = arr.view();
    EXPECT_EQ(15u, view.size());
    EXPECT_EQ(arr.data(), view.data());
    for (size_t i = 0; i < 15; ++i) {
        EXPECT_FLOAT_EQ(3.14f, view[i]);
    }
}

#endif  // JET_USE_CUDA

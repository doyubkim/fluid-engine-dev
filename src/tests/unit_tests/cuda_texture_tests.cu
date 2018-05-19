// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_array1.h>
#include <jet/cuda_array_view1.h>
#include <jet/cuda_texture.h>

#include <thrust/for_each.h>

#include <gtest/gtest.h>

using namespace jet;

template <typename T>
struct CopyTextureToArray {
    cudaTextureObject_t srcTex = 0;
    T* dstArr = nullptr;

    CopyTextureToArray(cudaTextureObject_t srcTex_, T* dstArr_)
        : srcTex(srcTex_), dstArr(dstArr_) {}

    JET_CUDA_DEVICE void operator()(size_t i) {
        dstArr[i] = tex1D<T>(srcTex, i + 0.5f);
    }
};

TEST(CudaTexture1, Constructors) {
    CudaTexture1<float> cudaTex0;
    EXPECT_EQ(0u, cudaTex0.size());
    EXPECT_EQ(0, cudaTex0.textureObject());

    CudaArray1<float> cudaArr1 = {1.0f, 2.0f, 3.0f, 4.0f};
    CudaTexture1<float> cudaTex1(cudaArr1);
    EXPECT_EQ(4u, cudaTex1.size());
    EXPECT_NE(0, cudaTex1.textureObject());

    CudaArray1<float> cudaArr1_2(4);
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.size()),
        CopyTextureToArray<float>(cudaTex1.textureObject(), cudaArr1_2.data()));
    for (size_t i = 0; i < cudaArr1.size(); ++i) {
        EXPECT_FLOAT_EQ(cudaArr1[i], cudaArr1_2[i]);
    }
}

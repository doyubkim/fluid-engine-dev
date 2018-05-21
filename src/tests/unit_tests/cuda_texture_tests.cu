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

namespace {

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

template <typename T>
struct CopyTextureToArray2 {
    cudaTextureObject_t srcTex = 0;
    T* dstArr = nullptr;
    size_t width = 0;

    CopyTextureToArray2(cudaTextureObject_t srcTex_, T* dstArr_, size_t width_)
        : srcTex(srcTex_), dstArr(dstArr_), width(width_) {}

    JET_CUDA_DEVICE void operator()(size_t j) {
        for (size_t i = 0; i < width; ++i) {
            dstArr[i + width * j] = tex2D<T>(srcTex, i + 0.5f, j + 0.5f);
        }
    }
};

template <typename T>
struct CopyTextureToArray3 {
    cudaTextureObject_t srcTex = 0;
    T* dstArr = nullptr;
    size_t width = 0;
    size_t height = 0;

    CopyTextureToArray3(cudaTextureObject_t srcTex_, T* dstArr_, size_t width_,
                        size_t height_)
        : srcTex(srcTex_), dstArr(dstArr_), width(width_), height(height_) {}

    JET_CUDA_DEVICE void operator()(size_t k) {
        for (size_t j = 0; j < height; ++j) {
            for (size_t i = 0; i < width; ++i) {
                dstArr[i + width * (j + height * k)] =
                    tex3D<T>(srcTex, i + 0.5f, j + 0.5f, k + 0.5f);
            }
        }
    }
};

}  // namespace

TEST(CudaTexture1, Constructors) {
    // Default ctor
    CudaTexture1<float> cudaTex0;
    EXPECT_EQ(0u, cudaTex0.size());
    EXPECT_EQ(0, cudaTex0.textureObject());

    // Ctor with view
    CudaArray1<float> cudaArr1 = {1.0f, 2.0f, 3.0f, 4.0f};
    CudaTexture1<float> cudaTex1(cudaArr1);
    EXPECT_EQ(4u, cudaTex1.size());
    ASSERT_NE(0, cudaTex1.textureObject());

    CudaArray1<float> cudaArr1_2(cudaArr1.size());
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.size()),
        CopyTextureToArray<float>(cudaTex1.textureObject(), cudaArr1_2.data()));
    for (size_t i = 0; i < cudaArr1.size(); ++i) {
        EXPECT_FLOAT_EQ(cudaArr1[i], cudaArr1_2[i]);
    }

    // Copy ctor
    CudaTexture1<float> cudaTex2(cudaTex1);
    EXPECT_EQ(4u, cudaTex2.size());
    ASSERT_NE(0, cudaTex2.textureObject());
    ASSERT_NE(cudaTex1.textureObject(), cudaTex2.textureObject());

    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.size()),
        CopyTextureToArray<float>(cudaTex2.textureObject(), cudaArr1_2.data()));
    for (size_t i = 0; i < cudaArr1.size(); ++i) {
        EXPECT_FLOAT_EQ(cudaArr1[i], cudaArr1_2[i]);
    }

    // Move ctor
    CudaTexture1<float> cudaTex3 = std::move(cudaTex2);
    EXPECT_EQ(0u, cudaTex2.size());
    EXPECT_EQ(0, cudaTex2.textureObject());
    EXPECT_EQ(4u, cudaTex3.size());
    ASSERT_NE(0, cudaTex3.textureObject());

    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.size()),
        CopyTextureToArray<float>(cudaTex3.textureObject(), cudaArr1_2.data()));
    for (size_t i = 0; i < cudaArr1.size(); ++i) {
        EXPECT_FLOAT_EQ(cudaArr1[i], cudaArr1_2[i]);
    }
}

TEST(CudaTexture2, Constructors) {
    // Default ctor
    CudaTexture2<float> cudaTex0;
    EXPECT_EQ(0u, cudaTex0.width());
    EXPECT_EQ(0u, cudaTex0.height());
    EXPECT_EQ(0, cudaTex0.textureObject());

    // Ctor with view
    CudaArray2<float> cudaArr1 = {{1.0f, 2.0f, 3.0f, 4.0f},
                                  {5.0f, 6.0f, 7.0f, 8.0f}};
    CudaTexture2<float> cudaTex1(cudaArr1);
    EXPECT_EQ(4u, cudaTex1.width());
    EXPECT_EQ(2u, cudaTex1.height());
    ASSERT_NE(0, cudaTex1.textureObject());

    CudaArray2<float> cudaArr1_2(cudaArr1.size());
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.height()),
        CopyTextureToArray2<float>(cudaTex1.textureObject(), cudaArr1_2.data(),
                                   cudaArr1_2.width()));
    for (size_t j = 0; j < cudaArr1.height(); ++j) {
        for (size_t i = 0; i < cudaArr1.width(); ++i) {
            EXPECT_FLOAT_EQ(cudaArr1(i, j), cudaArr1_2(i, j));
        }
    }

    // Copy ctor
    CudaTexture2<float> cudaTex2(cudaTex1);
    EXPECT_EQ(4u, cudaTex2.width());
    EXPECT_EQ(2u, cudaTex2.height());
    ASSERT_NE(0, cudaTex2.textureObject());
    ASSERT_NE(cudaTex1.textureObject(), cudaTex2.textureObject());

    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.height()),
        CopyTextureToArray2<float>(cudaTex2.textureObject(), cudaArr1_2.data(),
                                   cudaArr1_2.width()));
    for (size_t j = 0; j < cudaArr1.height(); ++j) {
        for (size_t i = 0; i < cudaArr1.width(); ++i) {
            EXPECT_FLOAT_EQ(cudaArr1(i, j), cudaArr1_2(i, j));
        }
    }

    // Move ctor
    CudaTexture2<float> cudaTex3 = std::move(cudaTex2);
    EXPECT_EQ(0u, cudaTex2.width());
    EXPECT_EQ(0u, cudaTex2.height());
    EXPECT_EQ(0, cudaTex2.textureObject());
    EXPECT_EQ(4u, cudaTex3.width());
    EXPECT_EQ(2u, cudaTex3.height());
    ASSERT_NE(0, cudaTex3.textureObject());

    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.height()),
        CopyTextureToArray2<float>(cudaTex3.textureObject(), cudaArr1_2.data(),
                                   cudaArr1_2.width()));
    for (size_t j = 0; j < cudaArr1.height(); ++j) {
        for (size_t i = 0; i < cudaArr1.width(); ++i) {
            EXPECT_FLOAT_EQ(cudaArr1(i, j), cudaArr1_2(i, j));
        }
    }
}

TEST(CudaTexture3, Constructors) {
    // Default ctor
    CudaTexture3<float> cudaTex0;
    EXPECT_EQ(0u, cudaTex0.width());
    EXPECT_EQ(0u, cudaTex0.height());
    EXPECT_EQ(0u, cudaTex0.depth());
    EXPECT_EQ(0, cudaTex0.textureObject());

    // Ctor with view
    CudaArray3<float> cudaArr1 = {
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}},
        {{13.f, 14.f, 15.f, 16.f},
         {17.f, 18.f, 19.f, 20.f},
         {21.f, 22.f, 23.f, 24.f}}};

    CudaTexture3<float> cudaTex1(cudaArr1);
    EXPECT_EQ(4u, cudaTex1.width());
    EXPECT_EQ(3u, cudaTex1.height());
    EXPECT_EQ(2u, cudaTex1.depth());
    ASSERT_NE(0, cudaTex1.textureObject());

    CudaArray3<float> cudaArr1_2(cudaArr1.size());
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.depth()),
        CopyTextureToArray3<float>(cudaTex1.textureObject(), cudaArr1_2.data(),
                                   cudaArr1_2.width(), cudaArr1_2.height()));
    for (size_t k = 0; k < cudaArr1.depth(); ++k) {
        for (size_t j = 0; j < cudaArr1.height(); ++j) {
            for (size_t i = 0; i < cudaArr1.width(); ++i) {
                EXPECT_FLOAT_EQ(cudaArr1(i, j, k), cudaArr1_2(i, j, k));
            }
        }
    }

    // Copy ctor
    CudaTexture3<float> cudaTex2(cudaTex1);
    EXPECT_EQ(4u, cudaTex2.width());
    EXPECT_EQ(3u, cudaTex2.height());
    EXPECT_EQ(2u, cudaTex2.depth());
    ASSERT_NE(0, cudaTex2.textureObject());
    ASSERT_NE(cudaTex1.textureObject(), cudaTex2.textureObject());

    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.depth()),
        CopyTextureToArray3<float>(cudaTex2.textureObject(), cudaArr1_2.data(),
                                   cudaArr1_2.width(), cudaArr1_2.height()));
    for (size_t k = 0; k < cudaArr1.depth(); ++k) {
        for (size_t j = 0; j < cudaArr1.height(); ++j) {
            for (size_t i = 0; i < cudaArr1.width(); ++i) {
                EXPECT_FLOAT_EQ(cudaArr1(i, j, k), cudaArr1_2(i, j, k));
            }
        }
    }

    // Move ctor
    CudaTexture3<float> cudaTex3 = std::move(cudaTex2);
    EXPECT_EQ(0u, cudaTex2.width());
    EXPECT_EQ(0u, cudaTex2.height());
    EXPECT_EQ(0u, cudaTex2.depth());
    EXPECT_EQ(0, cudaTex2.textureObject());
    EXPECT_EQ(4u, cudaTex3.width());
    EXPECT_EQ(3u, cudaTex3.height());
    EXPECT_EQ(2u, cudaTex3.depth());
    ASSERT_NE(0, cudaTex3.textureObject());

    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(cudaArr1.depth()),
        CopyTextureToArray3<float>(cudaTex3.textureObject(), cudaArr1_2.data(),
                                   cudaArr1_2.width(), cudaArr1_2.height()));
    for (size_t k = 0; k < cudaArr1.depth(); ++k) {
        for (size_t j = 0; j < cudaArr1.height(); ++j) {
            for (size_t i = 0; i < cudaArr1.width(); ++i) {
                EXPECT_FLOAT_EQ(cudaArr1(i, j, k), cudaArr1_2(i, j, k));
            }
        }
    }
}

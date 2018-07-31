// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_array.h>
#include <jet/cuda_array_view.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(CudaArray2, Constructors) {
    {
        CudaArray2<float> arr;
        EXPECT_EQ(0u, arr.width());
        EXPECT_EQ(0u, arr.height());
    }
    {
        CudaArray2<float> arr(CudaStdArray<size_t, 2>(3, 7));
        EXPECT_EQ(3u, arr.width());
        EXPECT_EQ(7u, arr.height());
        for (size_t i = 0; i < 21; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }
    }
    {
        CudaArray2<float> arr(CudaStdArray<size_t, 2>(1, 9), 1.5f);
        EXPECT_EQ(1u, arr.width());
        EXPECT_EQ(9u, arr.height());
        for (size_t i = 0; i < 9; ++i) {
            EXPECT_FLOAT_EQ(1.5f, arr[i]);
        }
    }
    {
        CudaArray2<float> arr(5, 2);
        EXPECT_EQ(5u, arr.width());
        EXPECT_EQ(2u, arr.height());
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }
    }
    {
        CudaArray2<float> arr(3, 4, 7.f);
        EXPECT_EQ(3u, arr.width());
        EXPECT_EQ(4u, arr.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ(7.f, arr[i]);
        }
    }
    {
        CudaArray2<float> arr({{1.f, 2.f, 3.f, 4.f},
                               {5.f, 6.f, 7.f, 8.f},
                               {9.f, 10.f, 11.f, 12.f}});
        EXPECT_EQ(4u, arr.width());
        EXPECT_EQ(3u, arr.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr[i]);
        }
    }
    {
        Array2<float> arr({{1.f, 2.f, 3.f, 4.f},
                           {5.f, 6.f, 7.f, 8.f},
                           {9.f, 10.f, 11.f, 12.f}});
        CudaArray2<float> arr2(arr);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }
    {
        CudaArray2<float> arr({{1.f, 2.f, 3.f, 4.f},
                               {5.f, 6.f, 7.f, 8.f},
                               {9.f, 10.f, 11.f, 12.f}});
        CudaArray2<float> arr2(arr);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }
    {
        CudaArray2<float> arr({{1.f, 2.f, 3.f, 4.f},
                               {5.f, 6.f, 7.f, 8.f},
                               {9.f, 10.f, 11.f, 12.f}});
        CudaArrayView2<float> arrVew(arr.data(), arr.size());
        EXPECT_EQ(4u, arrVew.width());
        EXPECT_EQ(3u, arrVew.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arrVew[i]);
        }
    }
}

TEST(CudaArray2, At) {
    {
        float values[12] = {0.f, 1.f, 2.f, 3.f, 4.f,  5.f,
                            6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
        CudaArray2<float> arr(4, 3);
        for (size_t i = 0; i < 12; ++i) {
            arr[i] = values[i];
        }

        // Test row-major
        EXPECT_FLOAT_EQ(0.f, arr(0, 0));
        EXPECT_FLOAT_EQ(1.f, arr(1, 0));
        EXPECT_FLOAT_EQ(2.f, arr(2, 0));
        EXPECT_FLOAT_EQ(3.f, arr(3, 0));
        EXPECT_FLOAT_EQ(4.f, arr(0, 1));
        EXPECT_FLOAT_EQ(5.f, arr(1, 1));
        EXPECT_FLOAT_EQ(6.f, arr(2, 1));
        EXPECT_FLOAT_EQ(7.f, arr(3, 1));
        EXPECT_FLOAT_EQ(8.f, arr(0, 2));
        EXPECT_FLOAT_EQ(9.f, arr(1, 2));
        EXPECT_FLOAT_EQ(10.f, arr(2, 2));
        EXPECT_FLOAT_EQ(11.f, arr(3, 2));
    }
}

TEST(CudaArray2, CopyFrom) {
    // From Array
    {
        Array2<float> arr({{1.f, 2.f, 3.f, 4.f},
                           {5.f, 6.f, 7.f, 8.f},
                           {9.f, 10.f, 11.f, 12.f}});
        CudaArray2<float> arr2(2, 5);

        arr2.copyFrom(arr);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From ArrayView
    {
        Array2<float> arr({{1.f, 2.f, 3.f, 4.f},
                           {5.f, 6.f, 7.f, 8.f},
                           {9.f, 10.f, 11.f, 12.f}});
        CudaArray2<float> arr2(2, 5);

        arr2.copyFrom(arr.view());
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From CudaArray
    {
        CudaArray2<float> arr({{1.f, 2.f, 3.f, 4.f},
                               {5.f, 6.f, 7.f, 8.f},
                               {9.f, 10.f, 11.f, 12.f}});
        CudaArray2<float> arr2(2, 5);

        arr2.copyFrom(arr);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From CudaArrayView
    {
        CudaArray2<float> arr({{1.f, 2.f, 3.f, 4.f},
                               {5.f, 6.f, 7.f, 8.f},
                               {9.f, 10.f, 11.f, 12.f}});
        CudaArray2<float> arr2(2, 5);

        arr2.copyFrom(arr.view());
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }
}

TEST(CudaArray2, CopyTo) {
    // To Array
    {
        CudaArray2<float> arr({{1.f, 2.f, 3.f, 4.f},
                               {5.f, 6.f, 7.f, 8.f},
                               {9.f, 10.f, 11.f, 12.f}});
        Array2<float> arr2(2, 5);

        arr.copyTo(arr2);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // To ArrayView
    {
        CudaArray2<float> arr({{1.f, 2.f, 3.f, 4.f},
                               {5.f, 6.f, 7.f, 8.f},
                               {9.f, 10.f, 11.f, 12.f}});
        Array2<float> arr2(4, 3);

        arr.copyTo(arr2.view());
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From CudaArray
    {
        CudaArray2<float> arr({{1.f, 2.f, 3.f, 4.f},
                               {5.f, 6.f, 7.f, 8.f},
                               {9.f, 10.f, 11.f, 12.f}});
        CudaArray2<float> arr2(2, 5);

        arr.copyTo(arr2);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From CudaArrayView
    {
        CudaArray2<float> arr({{1.f, 2.f, 3.f, 4.f},
                               {5.f, 6.f, 7.f, 8.f},
                               {9.f, 10.f, 11.f, 12.f}});
        CudaArray2<float> arr2(4, 3);

        arr.copyTo(arr2.view());
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }
}

TEST(CudaArray2, Fill) {
    CudaArray2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});

    arr.fill(42.0f);
    EXPECT_EQ(4u, arr.width());
    EXPECT_EQ(3u, arr.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(42.0f, arr[i]);
    }
}

TEST(CudaArray2, Resize) {
    {
        CudaArray2<float> arr;
        arr.resize(CudaStdArray<size_t, 2>(2, 9));
        EXPECT_EQ(2u, arr.width());
        EXPECT_EQ(9u, arr.height());
        for (size_t i = 0; i < 18; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }

        arr.resize(CudaStdArray<size_t, 2>(8, 13), 4.f);
        cudaDeviceSynchronize();
        EXPECT_EQ(8u, arr.width());
        EXPECT_EQ(13u, arr.height());
        for (size_t i = 0; i < 8; ++i) {
            for (size_t j = 0; j < 13; ++j) {
                if (i < 2 && j < 9) {
                    EXPECT_FLOAT_EQ(0.f, arr(i, j));
                } else {
                    EXPECT_FLOAT_EQ(4.f, arr(i, j));
                }
            }
        }
    }
    {
        CudaArray2<float> arr;
        arr.resize(7, 6);
        EXPECT_EQ(7u, arr.width());
        EXPECT_EQ(6u, arr.height());
        for (size_t i = 0; i < 42; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }

        arr.resize(1, 9, 3.f);
        EXPECT_EQ(1u, arr.width());
        EXPECT_EQ(9u, arr.height());
        for (size_t i = 0; i < 1; ++i) {
            for (size_t j = 0; j < 9; ++j) {
                if (j < 6) {
                    EXPECT_FLOAT_EQ(0.f, arr(i, j));
                } else {
                    EXPECT_FLOAT_EQ(3.f, arr(i, j));
                }
            }
        }
    }
}

TEST(CudaArray2, Clear) {
    CudaArray2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});

    arr.clear();
    EXPECT_EQ(0u, arr.width());
    EXPECT_EQ(0u, arr.height());
}

TEST(CudaArray2, Swap) {
    CudaArray2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});
    CudaArray2<float> arr2(2, 5, 42.f);

    arr.swap(arr2);

    EXPECT_EQ(2u, arr.width());
    EXPECT_EQ(5u, arr.height());
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(42.0f, arr[i]);
    }

    EXPECT_EQ(4u, arr2.width());
    EXPECT_EQ(3u, arr2.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
    }
}

TEST(CudaArray2, View) {
    CudaArray2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});

    auto view = arr.view();

    EXPECT_EQ(4u, view.width());
    EXPECT_EQ(3u, view.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, view[i]);
    }
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(arr(i, j), view(i, j));
        }
    }

    const auto& arrRef = arr;
    auto constView = arrRef.view();

    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, constView[i]);
    }
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(arr(i, j), constView(i, j));
        }
    }

    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            view(i, j) = float(i + 4 * j);
        }
    }

    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(float(i + 4 * j), arr(i, j));
            EXPECT_FLOAT_EQ(float(i + 4 * j), constView(i, j));
        }
    }
}

TEST(CudaArray2, AssignmentOperator) {
    CudaArray2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});
    CudaArray2<float> arr2(2, 5, 42.f);

    arr2 = arr;

    EXPECT_EQ(4u, arr.width());
    EXPECT_EQ(3u, arr.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, arr[i]);
    }

    EXPECT_EQ(4u, arr2.width());
    EXPECT_EQ(3u, arr2.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
    }
}

TEST(CudaArray2, MoveOperator) {
    CudaArray2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});
    CudaArray2<float> arr2(2, 5, 42.f);

    arr2 = std::move(arr);

    EXPECT_EQ(0u, arr.width());
    EXPECT_EQ(0u, arr.height());
    EXPECT_EQ(nullptr, arr.data());
    EXPECT_EQ(4u, arr2.width());
    EXPECT_EQ(3u, arr2.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
    }
}

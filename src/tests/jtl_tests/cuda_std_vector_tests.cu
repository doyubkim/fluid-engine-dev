// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_std_vector.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(CudaStdVector, Constructors) {
    {
        CudaStdVector<int> vec;
        EXPECT_EQ(0u, vec.size());
        EXPECT_EQ(nullptr, vec.data());
    }

    {
        CudaStdVector<int> vec(5, 3);
        EXPECT_EQ(5u, vec.size());
        std::vector<int> ans;
        vec.copyTo(ans);
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(3, ans[i]);
        }
    }

    {
        std::vector<int> host({1, 2, 3, 4, 5});
        CudaStdVector<int> vec(host);
        std::vector<int> ans;
        vec.copyTo(ans);
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(host[i], ans[i]);
        }
    }

    {
        std::vector<int> host({1, 2, 3, 4, 5});
        CudaStdVector<int> vec(host);
        CudaStdVector<int> vec2(vec);
        std::vector<int> ans;
        vec2.copyTo(ans);
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(host[i], ans[i]);
        }
    }

    {
        std::vector<int> host({1, 2, 3, 4, 5});
        CudaStdVector<int> vec(host);
        CudaStdVector<int> vec2 = std::move(vec);

        EXPECT_EQ(0u, vec.size());
        EXPECT_EQ(nullptr, vec.data());

        std::vector<int> ans;
        vec2.copyTo(ans);
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(host[i], ans[i]);
        }
    }
}

TEST(CudaStdVector, Getters) {
    std::vector<int> host({1, 2, 3, 4, 5});
    CudaStdVector<int> vec(host);

    EXPECT_NE(nullptr, vec.data());
    EXPECT_EQ(5u, vec.size());

    const auto& vecRef = vec;
    EXPECT_NE(nullptr, vecRef.data());
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(host[i], vecRef.at(i));
    }

    std::vector<int> ans;
    vec.copyTo(ans);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(host[i], ans[i]);
    }
}

TEST(CudaStdVector, Modifiers) {
    std::vector<int> host({1, 2, 3, 4, 5});
    CudaStdVector<int> vec(host);

    vec.at(0) = 9;
    vec.at(1) = 8;
    vec.at(2) = 7;
    vec.at(3) = 6;
    vec.at(4) = 5;
    const auto& vecRef = vec;
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(9 - i, vec.at(i));
        EXPECT_EQ(9 - i, vecRef.at(i));
    }

    vec.fill(10);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(10, vec.at(i));
    }

    vec.clear();
    EXPECT_EQ(0u, vec.size());
    EXPECT_EQ(nullptr, vec.data());

    vec.resizeUninitialized(4);
    EXPECT_EQ(4u, vec.size());

    vec.resize(7, 3);
    EXPECT_EQ(7u, vec.size());
    for (int i = 4; i < 7; ++i) {
        EXPECT_EQ(3, vec.at(i));
    }

    CudaStdVector<int> vec2(host);
    vec.swap(vec2);

    EXPECT_EQ(7u, vec2.size());
    for (int i = 4; i < 7; ++i) {
        EXPECT_EQ(3, vec2.at(i));
    }
    EXPECT_EQ(5u, vec.size());
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(host[i], vec.at(i));
    }

    vec.push_back(6);
    EXPECT_EQ(6u, vec.size());
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(host[i], vec.at(i));
    }
    EXPECT_EQ(6, vec.at(5));

    vec2.copyFrom(host);
    vec.append(vec2);

    EXPECT_EQ(11u, vec.size());
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(host[i], vec.at(i));
    }
    EXPECT_EQ(6, vec.at(5));
    for (int i = 6; i < 11; ++i) {
        EXPECT_EQ(host[i - 6], vec.at(i));
    }

    vec.copyFrom(vec2);
    EXPECT_EQ(5u, vec.size());
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(host[i], vec.at(i));
    }
}

TEST(CudaStdVector, Operators) {
    std::vector<int> host({1, 2, 3, 4, 5});
    CudaStdVector<int> vec(host);

    vec[0] = 9;
    vec[1] = 8;
    vec[2] = 7;
    vec[3] = 6;
    vec[4] = 5;
    const auto& vecRef = vec;
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(9 - i, vec[i]);
        EXPECT_EQ(9 - i, vecRef[i]);
    }

    vec = host;
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(host[i], vec[i]);
    }

    CudaStdVector<int> vec2(host);
    vec.fill(42);
    vec = vec2;
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(host[i], vec[i]);
    }
}

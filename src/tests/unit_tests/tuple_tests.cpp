// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/tuple.h>

#include <gtest/gtest.h>

using namespace jet;

namespace {

template <typename T, size_t N>
void expectEqual(const std::array<T, N>& expected, const Tuple<T, N>& actual) {
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

}  // namespace

TEST(Tuple, Constructors) {
    FloatN<5> t0;
    expectEqual({{0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}, t0);

    FloatN<5> t1{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    expectEqual({{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}}, t1);

    Float1 t2;
    expectEqual({{0}}, t2);

    Float1 t3{42};
    expectEqual({{42}}, t3);

    Double2 t4;
    expectEqual({{0, 0}}, t4);

    Double2 t5{1.0, 3.0};
    expectEqual({{1, 3}}, t5);

    Byte3 t6;
    expectEqual({{0, 0, 0}}, t6);

    Byte3 t7{5, 7, 9};
    expectEqual({{5, 7, 9}}, t7);

    Long4 t8;
    expectEqual({{0, 0, 0, 0}}, t8);

    Long4 t9{4, 3, 2, 1};
    expectEqual({{4, 3, 2, 1}}, t9);
}

TEST(Tuple, GetterSetter) {
    Long4 l4t{4, 3, 2, 1};
    EXPECT_EQ(4, l4t[0]);
    EXPECT_EQ(3, l4t[1]);
    EXPECT_EQ(2, l4t[2]);
    EXPECT_EQ(1, l4t[3]);

    l4t[0] = 5;
    l4t[1] = 6;
    l4t[2] = 7;
    l4t[3] = 8;
    EXPECT_EQ(5, l4t.x);
    EXPECT_EQ(6, l4t.y);
    EXPECT_EQ(7, l4t.z);
    EXPECT_EQ(8, l4t.w);
}

TEST(Tuple, Add) {
    IntN<5> a{0, 1, 2, 3, 4};
    IntN<5> b{5, 6, 7, 8, 9};
    expectEqual({{5, 7, 9, 11, 13}}, a + b);
    expectEqual({{4, 5, 6, 7, 8}}, -1 + b);
    expectEqual({{3, 4, 5, 6, 7}}, a + 3);

    Byte1 c{5};
    Byte1 d{4};
    expectEqual({{9}}, c + d);
    expectEqual({{7}}, int8_t{3} + d);
    expectEqual({{14}}, c + int8_t{9});

    Short2 e{0, 1};
    Short2 f{5, 6};
    expectEqual({{5, 7}}, e + f);
    expectEqual({{4, 5}}, int16_t{-1} + f);
    expectEqual({{3, 4}}, e + int16_t{3});

    Int3 g{0, 1, 2};
    Int3 h{5, 6, 7};
    expectEqual({{5, 7, 9}}, g + h);
    expectEqual({{4, 5, 6}}, -1 + h);
    expectEqual({{3, 4, 5}}, g + 3);

    Long4 i{0, 1, 2, 3};
    Long4 j{5, 6, 7, 8};
    expectEqual({{5, 7, 9, 11}}, i + j);
    expectEqual({{4, 5, 6, 7}}, -1LL + j);
    expectEqual({{3, 4, 5, 6}}, i + 3LL);
}

TEST(Tuple, Subtract) {
    IntN<5> a{0, 1, 2, 3, 4};
    IntN<5> b{5, 6, 7, 8, 9};
    expectEqual({{-5, -5, -5, -5, -5}}, a - b);
    expectEqual({{-6, -7, -8, -9, -10}}, -1 - b);
    expectEqual({{-3, -2, -1, 0, 1}}, a - 3);

    Int1 c{5};
    Int1 d{4};
    expectEqual({{1}}, c - d);
    expectEqual({{-1}}, 3 - d);
    expectEqual({{-4}}, c - 9);

    Short2 e{0, 1};
    Short2 f{5, 6};
    expectEqual({{-5, -5}}, e - f);
    expectEqual({{-6, -7}}, int16_t{-1} - f);
    expectEqual({{-3, -2}}, e - int16_t{3});

    Int3 g{0, 1, 2};
    Int3 h{5, 6, 7};
    expectEqual({{-5, -5, -5}}, g - h);
    expectEqual({{-6, -7, -8}}, -1 - h);
    expectEqual({{-3, -2, -1}}, g - 3);

    Long4 i{0, 1, 2, 3};
    Long4 j{5, 6, 7, 8};
    expectEqual({{-5, -5, -5, -5}}, i - j);
    expectEqual({{-6, -7, -8, -9}}, -1LL - j);
    expectEqual({{-3, -2, -1, 0}}, i - 3LL);
}

TEST(Tuple, Compare) {
    Long4 a{0, 1, 2, 3};
    Long4 b{5, 6, 7, 8};
    Long4 c{5, 6, 7, 8};

    EXPECT_NE(a, b);
    EXPECT_EQ(b, c);
}

TEST(Tuple, Utilities) {
    IntN<5> a{5, 6, 7, 8, 9};

    int sum = accumulate(a, 0);
    EXPECT_EQ(35, sum);

    int mul = product(a, 1);
    EXPECT_EQ(15120, mul);

    IntN<5> b{6, 3, 9, 1, 4};
    auto c = min(a, b);
    EXPECT_EQ(IntN<5>(5, 3, 7, 1, 4), c);

    auto d = max(a, b);
    EXPECT_EQ(IntN<5>(6, 6, 9, 8, 9), d);

    Float3 f3{2.f, 4.f, 1.f}, low{3.f, -1.f, 0.f}, high{5.f, 2.f, 3.f};
    auto clampedF3 = clamp<float, 3, Float3>(f3, low, high);
    EXPECT_EQ(Float3(3.f, 2.f, 1.f), clampedF3);

    Float4 e{2.5f, 4.1f, 1.9f, 2.0f};
    auto f = ceil(e);
    EXPECT_EQ(Float4(3.0f, 5.0f, 2.0f, 2.0f), f);

    auto g = floor(e);
    EXPECT_EQ(Float4(2.0f, 4.0f, 1.0f, 2.0f), g);

    fill(a, 4);
    EXPECT_EQ(IntN<5>(4, 4, 4, 4, 4), a);
}

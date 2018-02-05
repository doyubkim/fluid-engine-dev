// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/pde.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Pde, Upwind1)
{
    double values[3] = {0.0, 1.0, -1.0};
	auto result = upwind1(values, 0.5);

    EXPECT_DOUBLE_EQ(2.0, result[0]);
	EXPECT_DOUBLE_EQ(-4.0, result[1]);

	double d0 = upwind1(values, 2.0, true);
	double d1 = upwind1(values, 2.0, false);

	EXPECT_DOUBLE_EQ(0.5, d0);
	EXPECT_DOUBLE_EQ(-1.0, d1);
}

TEST(Pde, Cd2)
{
    double values[3] = {0.0, 1.0, -1.0};
	double result = cd2(values, 0.5);

    EXPECT_DOUBLE_EQ(-1.0, result);
}

TEST(Pde, Eno3)
{
    double values0[7] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	auto result0 = eno3(values0, 0.5);

	// Sanity check for linear case
	EXPECT_DOUBLE_EQ(2.0, result0[0]);
	EXPECT_DOUBLE_EQ(2.0, result0[1]);

	double d0 = eno3(values0, 2.0, true);
	double d1 = eno3(values0, 2.0, false);

	EXPECT_DOUBLE_EQ(0.5, d0);
	EXPECT_DOUBLE_EQ(0.5, d1);

	// Unit-step function
	double values1[7] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
	auto result1 = eno3(values1, 0.5);

	// Check monotonicity
	EXPECT_LE(0.0, result1[0]);
	EXPECT_DOUBLE_EQ(0.0, result1[1]);
}

TEST(Pde, Weno5)
{
    double values0[7] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	auto result0 = weno5(values0, 0.5);

	// Sanity check for linear case
	EXPECT_DOUBLE_EQ(2.0, result0[0]);
	EXPECT_DOUBLE_EQ(2.0, result0[1]);

	double d0 = weno5(values0, 2.0, true);
	double d1 = weno5(values0, 2.0, false);

	EXPECT_DOUBLE_EQ(0.5, d0);
	EXPECT_DOUBLE_EQ(0.5, d1);

	// Unit-step function
	double values1[7] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
	auto result1 = weno5(values1, 0.5);

	// Check monotonicity
	EXPECT_LE(0.0, result1[0]);
	EXPECT_LE(std::fabs(result1[1]), 1e-10);
}

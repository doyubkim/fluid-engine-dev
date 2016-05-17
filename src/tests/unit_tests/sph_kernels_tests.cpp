// Copyright (c) 2016 Doyub Kim

#include <jet/sph_kernels2.h>
#include <jet/sph_kernels3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(SphStdKernel2, Constructors)
{
	SphStdKernel2 kernel;
	EXPECT_DOUBLE_EQ(0.0, kernel.h);

	SphStdKernel2 kernel2(3.0);
	EXPECT_DOUBLE_EQ(3.0, kernel2.h);
}

TEST(SphStdKernel2, KernelFunction)
{
	SphStdKernel2 kernel(10.0);

	double prevValue = kernel(0.0);

	for (int i = 1; i <= 10; ++i)
	{
		double value = kernel(static_cast<double>(i));
		EXPECT_LT(value, prevValue);
	}
}

TEST(SphStdKernel2, FirstDerivative)
{
	SphStdKernel2 kernel(10.0);

	double value0 = kernel.firstDerivative(0.0);
	double value1 = kernel.firstDerivative(5.0);
	double value2 = kernel.firstDerivative(10.0);
	EXPECT_DOUBLE_EQ(0.0, value0);
	EXPECT_DOUBLE_EQ(0.0, value2);
	EXPECT_LT(value1, value0);
}

TEST(SphSpikyKernel2, Constructors)
{
	SphSpikyKernel2 kernel;
	EXPECT_DOUBLE_EQ(0.0, kernel.h);

	SphSpikyKernel2 kernel2(3.0);
	EXPECT_DOUBLE_EQ(3.0, kernel2.h);
}

TEST(SphSpikyKernel2, KernelFunction)
{
	SphSpikyKernel2 kernel(10.0);

	double prevValue = kernel(0.0);

	for (int i = 1; i <= 10; ++i)
	{
		double value = kernel(static_cast<double>(i));
		EXPECT_LT(value, prevValue);
	}
}

TEST(SphSpikyKernel2, FirstDerivative)
{
	SphSpikyKernel2 kernel(10.0);

	double value0 = kernel.firstDerivative(0.0);
	double value1 = kernel.firstDerivative(5.0);
	double value2 = kernel.firstDerivative(10.0);
	EXPECT_LT(value0, value1);
	EXPECT_LT(value1, value2);
}

TEST(SphSpikyKernel2, SecondDerivative)
{
	SphSpikyKernel2 kernel(10.0);

	double value0 = kernel.secondDerivative(0.0);
	double value1 = kernel.secondDerivative(5.0);
	double value2 = kernel.secondDerivative(10.0);
	EXPECT_LT(value1, value0);
	EXPECT_LT(value2, value1);
}

TEST(SphStdKernel3, Constructors)
{
	SphStdKernel3 kernel;
	EXPECT_DOUBLE_EQ(0.0, kernel.h);

	SphStdKernel3 kernel2(3.0);
	EXPECT_DOUBLE_EQ(3.0, kernel2.h);
}

TEST(SphStdKernel3, KernelFunction)
{
	SphStdKernel3 kernel(10.0);

	double prevValue = kernel(0.0);

	for (int i = 1; i <= 10; ++i)
	{
		double value = kernel(static_cast<double>(i));
		EXPECT_LT(value, prevValue);
	}
}

TEST(SphStdKernel3, FirstDerivative)
{
	SphStdKernel3 kernel(10.0);

	double value0 = kernel.firstDerivative(0.0);
	double value1 = kernel.firstDerivative(5.0);
	double value2 = kernel.firstDerivative(10.0);
	EXPECT_DOUBLE_EQ(0.0, value0);
	EXPECT_DOUBLE_EQ(0.0, value2);
	EXPECT_LT(value1, value0);
}

TEST(SphSpikyKernel3, Constructors)
{
	SphSpikyKernel3 kernel;
	EXPECT_DOUBLE_EQ(0.0, kernel.h);

	SphSpikyKernel3 kernel2(3.0);
	EXPECT_DOUBLE_EQ(3.0, kernel2.h);
}

TEST(SphSpikyKernel3, KernelFunction)
{
	SphSpikyKernel3 kernel(10.0);

	double prevValue = kernel(0.0);

	for (int i = 1; i <= 10; ++i)
	{
		double value = kernel(static_cast<double>(i));
		EXPECT_LT(value, prevValue);
	}
}

TEST(SphSpikyKernel3, FirstDerivative)
{
	SphSpikyKernel3 kernel(10.0);

	double value0 = kernel.firstDerivative(0.0);
	double value1 = kernel.firstDerivative(5.0);
	double value2 = kernel.firstDerivative(10.0);
	EXPECT_LT(value0, value1);
	EXPECT_LT(value1, value2);
}

TEST(SphSpikyKernel3, SecondDerivative)
{
	SphSpikyKernel3 kernel(10.0);

	double value0 = kernel.secondDerivative(0.0);
	double value1 = kernel.secondDerivative(5.0);
	double value2 = kernel.secondDerivative(10.0);
	EXPECT_LT(value1, value0);
	EXPECT_LT(value2, value1);
}

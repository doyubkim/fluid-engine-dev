// Copyright (c) 2016 Doyub Kim

#include <jet/sph_system_data3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(SphSystemData3, Parameters) {
    SphSystemData3 data;

    data.setTargetDensity(123.0);
    data.setTargetSpacing(0.549);
    data.setRelativeKernelRadius(2.5);

    EXPECT_EQ(123.0, data.targetDensity());
    EXPECT_EQ(0.549, data.targetSpacing());
    EXPECT_EQ(0.549, data.radius());
    EXPECT_EQ(2.5, data.relativeKernelRadius());
    EXPECT_DOUBLE_EQ(2.5 * 0.549, data.kernelRadius());

    data.setRadius(0.413);
    EXPECT_EQ(0.413, data.targetSpacing());
    EXPECT_EQ(0.413, data.radius());
    EXPECT_EQ(2.5, data.relativeKernelRadius());
    EXPECT_DOUBLE_EQ(2.5 * 0.413, data.kernelRadius());
}

TEST(SphSystemData3, Particles) {
    SphSystemData3 data;

    data.setTargetSpacing(1.0);
    data.setRelativeKernelRadius(1.0);

    data.addParticle(Vector3D(0, 0, 0));
    data.addParticle(Vector3D(1, 0, 0));

    data.buildNeighborSearcher();
    data.updateDensities();

    // See if we get symmetric density profile
    auto den = data.densities();
    EXPECT_LT(0.0, den[0]);
    EXPECT_EQ(den[0], den[1]);

    Array1<double> values = {1.0, 1.0};
    double midVal = data.interpolate(
        Vector3D(0.5, 0, 0), values.constAccessor());
    EXPECT_LT(0.0, midVal);
    EXPECT_GT(1.0, midVal);
}

// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/sph_system_data2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(SphSystemData2, Parameters) {
    SphSystemData2 data;

    data.setTargetDensity(123.0);
    data.setTargetSpacing(0.549);
    data.setRelativeKernelRadius(2.5);

    EXPECT_EQ(123.0, data.targetDensity());
    EXPECT_EQ(0.549, data.targetSpacing());
    EXPECT_EQ(0.549, data.radius());
    EXPECT_EQ(2.5, data.relativeKernelRadius());
    EXPECT_DOUBLE_EQ(2.5 * 0.549, data.kernelRadius());

    data.setKernelRadius(1.9);
    EXPECT_DOUBLE_EQ(1.9, data.kernelRadius());
    EXPECT_DOUBLE_EQ(1.9 / 2.5, data.targetSpacing());

    data.setRadius(0.413);
    EXPECT_EQ(0.413, data.targetSpacing());
    EXPECT_EQ(0.413, data.radius());
    EXPECT_EQ(2.5, data.relativeKernelRadius());
    EXPECT_DOUBLE_EQ(2.5 * 0.413, data.kernelRadius());

    data.setMass(2.0 * data.mass());
    EXPECT_DOUBLE_EQ(246.0, data.targetDensity());
}

TEST(SphSystemData2, Particles) {
    SphSystemData2 data;

    data.setTargetSpacing(1.0);
    data.setRelativeKernelRadius(1.0);

    data.addParticle(Vector2D(0, 0));
    data.addParticle(Vector2D(1, 0));

    data.buildNeighborSearcher();
    data.updateDensities();

    // See if we get symmetric density profile
    auto den = data.densities();
    EXPECT_LT(0.0, den[0]);
    EXPECT_EQ(den[0], den[1]);

    Array1<double> values = {1.0, 1.0};
    double midVal = data.interpolate(Vector2D(0.5, 0), values.constAccessor());
    EXPECT_LT(0.0, midVal);
    EXPECT_GT(1.0, midVal);
}

TEST(SphSystemData2, Serialization) {
    SphSystemData2 data;

    ParticleSystemData2::VectorData positions = {
        {0.3, 0.5},
        {0.6, 0.8},
        {0.1, 0.8},
        {0.7, 0.9},
        {0.3, 0.2},
        {0.8, 0.3},
        {0.8, 0.5},
        {0.4, 0.9},
        {0.8, 0.6},
        {0.2, 0.9},
        {0.1, 0.2},
        {0.6, 0.9},
        {0.2, 0.2},
        {0.5, 0.6},
        {0.8, 0.4},
        {0.4, 0.2},
        {0.2, 0.3},
        {0.8, 0.6},
        {0.2, 0.8},
        {1.0, 0.5}
    };
    data.addParticles(positions);

    data.setTargetDensity(123.0);
    data.setTargetSpacing(0.549);
    data.setRelativeKernelRadius(2.5);

    size_t a0 = data.addScalarData(2.0);
    size_t a1 = data.addScalarData(9.0);
    size_t a2 = data.addVectorData({1.0, -3.0});

    std::vector<uint8_t> buffer;
    data.serialize(&buffer);

    SphSystemData2 data2;
    data2.deserialize(buffer);

    EXPECT_EQ(123.0, data2.targetDensity());
    EXPECT_EQ(0.549, data2.targetSpacing());
    EXPECT_EQ(0.549, data2.radius());
    EXPECT_EQ(2.5, data2.relativeKernelRadius());
    EXPECT_DOUBLE_EQ(2.5 * 0.549, data2.kernelRadius());

    EXPECT_EQ(positions.size(), data2.numberOfParticles());
    auto as0 = data2.scalarDataAt(a0);
    for (size_t i = 0; i < positions.size(); ++i) {
        EXPECT_DOUBLE_EQ(2.0, as0[i]);
    }

    auto as1 = data2.scalarDataAt(a1);
    for (size_t i = 0; i < positions.size(); ++i) {
        EXPECT_DOUBLE_EQ(9.0, as1[i]);
    }

    auto as2 = data2.vectorDataAt(a2);
    for (size_t i = 0; i < positions.size(); ++i) {
        EXPECT_DOUBLE_EQ(1.0, as2[i].x);
        EXPECT_DOUBLE_EQ(-3.0, as2[i].y);
    }

    const auto& neighborLists = data.neighborLists();
    const auto& neighborLists2 = data2.neighborLists();
    EXPECT_EQ(neighborLists.size(), neighborLists2.size());

    for (size_t i = 0; i < neighborLists.size(); ++i) {
        const auto& neighbors = neighborLists[i];
        const auto& neighbors2 = neighborLists2[i];
        EXPECT_EQ(neighbors.size(), neighbors2.size());

        for (size_t j = 0; j < neighbors.size(); ++j) {
            EXPECT_EQ(neighbors[j], neighbors2[j]);
        }
    }
}

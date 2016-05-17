// Copyright (c) 2016 Doyub Kim

#include <jet/particle_system_data3.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(ParticleSystemData3, Resize) {
    ParticleSystemData3 particleSystem;
    particleSystem.resize(12);

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
}

TEST(ParticleSystemData3, AddScalarData) {
    ParticleSystemData3 particleSystem;
    particleSystem.resize(12);

    size_t a0 = particleSystem.addScalarData(2.0);
    size_t a1 = particleSystem.addScalarData(9.0);

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
    EXPECT_EQ(0u, a0);
    EXPECT_EQ(1u, a1);

    auto as0 = particleSystem.scalarDataAt(a0);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_DOUBLE_EQ(2.0, as0[i]);
    }

    auto as1 = particleSystem.scalarDataAt(a1);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_DOUBLE_EQ(9.0, as1[i]);
    }
}

TEST(ParticleSystemData3, AddVectorData) {
    ParticleSystemData3 particleSystem;
    particleSystem.resize(12);

    size_t a0 = particleSystem.addVectorData(Vector3D(2.0, 4.0, -1.0));
    size_t a1 = particleSystem.addVectorData(Vector3D(9.0, -2.0, 5.0));

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
    EXPECT_EQ(0u, a0);
    EXPECT_EQ(1u, a1);

    auto as0 = particleSystem.vectorDataAt(a0);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(Vector3D(2.0, 4.0, -1.0), as0[i]);
    }

    auto as1 = particleSystem.vectorDataAt(a1);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(Vector3D(9.0, -2.0, 5.0), as1[i]);
    }
}

TEST(ParticleSystemData3, AddParticles) {
    ParticleSystemData3 particleSystem;
    particleSystem.resize(12);

    particleSystem.addParticles(
        Array1<Vector3D>(
            {Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0)}).accessor(),
        Array1<Vector3D>(
            {Vector3D(7.0, 8.0, 9.0), Vector3D(8.0, 7.0, 6.0)}).accessor(),
        Array1<Vector3D>(
            {Vector3D(5.0, 4.0, 3.0), Vector3D(2.0, 1.0, 3.0)}).accessor());

    EXPECT_EQ(14u, particleSystem.numberOfParticles());
    auto p = particleSystem.positions();
    auto v = particleSystem.velocities();
    auto f = particleSystem.forces();

    EXPECT_EQ(Vector3D(1.0, 2.0, 3.0), p[12]);
    EXPECT_EQ(Vector3D(4.0, 5.0, 6.0), p[13]);
    EXPECT_EQ(Vector3D(7.0, 8.0, 9.0), v[12]);
    EXPECT_EQ(Vector3D(8.0, 7.0, 6.0), v[13]);
    EXPECT_EQ(Vector3D(5.0, 4.0, 3.0), f[12]);
    EXPECT_EQ(Vector3D(2.0, 1.0, 3.0), f[13]);
}

TEST(ParticleSystemData3, AddParticlesException) {
    ParticleSystemData3 particleSystem;
    particleSystem.resize(12);

    try {
        particleSystem.addParticles(
            Array1<Vector3D>(
                {Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0)}).accessor(),
            Array1<Vector3D>(
                {Vector3D(7.0, 8.0, 9.0)}).accessor(),
            Array1<Vector3D>(
                {Vector3D(5.0, 4.0, 3.0), Vector3D(2.0, 1.0, 3.0)}).accessor());

        EXPECT_FALSE(true) << "Invalid argument shoudl throw exception.";
    }
    catch (std::invalid_argument) {
        // Do nothing -- expected exception
    }

    EXPECT_EQ(12u, particleSystem.numberOfParticles());

    try {
        particleSystem.addParticles(
            Array1<Vector3D>(
                {Vector3D(1.0, 2.0, 3.0), Vector3D(4.0, 5.0, 6.0)}).accessor(),
            Array1<Vector3D>(
                {Vector3D(7.0, 8.0, 9.0), Vector3D(2.0, 1.0, 3.0)}).accessor(),
            Array1<Vector3D>(
                {Vector3D(5.0, 4.0, 3.0)}).accessor());

        EXPECT_FALSE(true) << "Invalid argument shoudl throw exception.";
    }
    catch (std::invalid_argument) {
        // Do nothing -- expected exception
    }

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
}

TEST(ParticleSystemData3, BuildNeighborSearcher) {
    ParticleSystemData3 particleSystem;
    ParticleSystemData3::VectorData positions = {
        {0.1, 0.0, 0.4},
        {0.6, 0.2, 0.6},
        {1.0, 0.3, 0.4},
        {0.9, 0.2, 0.2},
        {0.8, 0.4, 0.9},
        {0.1, 0.6, 0.2},
        {0.8, 0.0, 0.5},
        {0.9, 0.8, 0.2},
        {0.3, 0.5, 0.2},
        {0.1, 0.6, 0.6},
        {0.1, 0.2, 0.1},
        {0.2, 0.0, 0.0},
        {0.2, 0.6, 0.1},
        {0.1, 0.3, 0.7},
        {0.9, 0.7, 0.6},
        {0.4, 0.5, 0.1},
        {0.1, 0.1, 0.6},
        {0.7, 0.8, 1.0},
        {0.6, 0.9, 0.4},
        {0.7, 0.7, 0.0}
    };
    particleSystem.addParticles(positions);

    const double radius = 0.4;
    particleSystem.buildNeighborSearcher(radius);

    auto neighborSearcher = particleSystem.neighborSearcher();
    const Vector3D searchOrigin = {0.1, 0.2, 0.3};
    std::vector<size_t> found;
    neighborSearcher->forEachNearbyPoint(
        searchOrigin,
        radius,
        [&](size_t i, const Vector3D&) {
            found.push_back(i);
        });

    for (size_t ii = 0; ii < positions.size(); ++ii) {
        if (searchOrigin.distanceTo(positions[ii]) <= radius) {
            EXPECT_TRUE(
                found.end() != std::find(found.begin(), found.end(), ii));
        }
    }
}

TEST(ParticleSystemData3, BuildNeighborLists) {
    ParticleSystemData3 particleSystem;
    ParticleSystemData3::VectorData positions = {
        {0.7, 0.2, 0.2},
        {0.7, 0.8, 1.0},
        {0.9, 0.4, 0.0},
        {0.5, 0.1, 0.6},
        {0.6, 0.3, 0.8},
        {0.1, 0.6, 0.0},
        {0.5, 1.0, 0.2},
        {0.6, 0.7, 0.8},
        {0.2, 0.4, 0.7},
        {0.8, 0.5, 0.8},
        {0.0, 0.8, 0.4},
        {0.3, 0.0, 0.6},
        {0.7, 0.8, 0.3},
        {0.0, 0.7, 0.1},
        {0.6, 0.3, 0.8},
        {0.3, 0.2, 1.0},
        {0.3, 0.5, 0.6},
        {0.3, 0.9, 0.6},
        {0.9, 1.0, 1.0},
        {0.0, 0.1, 0.6}
    };
    particleSystem.addParticles(positions);

    const double radius = 0.4;
    particleSystem.buildNeighborSearcher(radius);
    particleSystem.buildNeighborLists(radius);

    const auto& neighborLists = particleSystem.neighborLists();
    EXPECT_EQ(positions.size(), neighborLists.size());

    for (size_t i = 0; i < neighborLists.size(); ++i) {
        const auto& neighbors = neighborLists[i];
        for (size_t ii = 0; ii < positions.size(); ++ii) {
            if (ii != i && positions[ii].distanceTo(positions[i]) <= radius) {
                EXPECT_TRUE(
                    neighbors.end()
                    != std::find(neighbors.begin(), neighbors.end(), ii));
            }
        }
    }
}

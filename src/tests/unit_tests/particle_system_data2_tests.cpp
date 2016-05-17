// Copyright (c) 2016 Doyub Kim

#include <jet/particle_system_data2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(ParticleSystemData2, Resize) {
    ParticleSystemData2 particleSystem;
    particleSystem.resize(12);

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
}

TEST(ParticleSystemData2, AddScalarData) {
    ParticleSystemData2 particleSystem;
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

TEST(ParticleSystemData2, AddVectorData) {
    ParticleSystemData2 particleSystem;
    particleSystem.resize(12);

    size_t a0 = particleSystem.addVectorData(Vector2D(2.0, 4.0));
    size_t a1 = particleSystem.addVectorData(Vector2D(9.0, -2.0));

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
    EXPECT_EQ(0u, a0);
    EXPECT_EQ(1u, a1);

    auto as0 = particleSystem.vectorDataAt(a0);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(Vector2D(2.0, 4.0), as0[i]);
    }

    auto as1 = particleSystem.vectorDataAt(a1);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(Vector2D(9.0, -2.0), as1[i]);
    }
}

TEST(ParticleSystemData2, AddParticles) {
    ParticleSystemData2 particleSystem;
    particleSystem.resize(12);

    particleSystem.addParticles(
        Array1<Vector2D>({Vector2D(1.0, 2.0), Vector2D(4.0, 5.0)}).accessor(),
        Array1<Vector2D>({Vector2D(7.0, 8.0), Vector2D(8.0, 7.0)}).accessor(),
        Array1<Vector2D>({Vector2D(5.0, 4.0), Vector2D(2.0, 1.0)}).accessor());

    EXPECT_EQ(14u, particleSystem.numberOfParticles());
    auto p = particleSystem.positions();
    auto v = particleSystem.velocities();
    auto f = particleSystem.forces();

    EXPECT_EQ(Vector2D(1.0, 2.0), p[12]);
    EXPECT_EQ(Vector2D(4.0, 5.0), p[13]);
    EXPECT_EQ(Vector2D(7.0, 8.0), v[12]);
    EXPECT_EQ(Vector2D(8.0, 7.0), v[13]);
    EXPECT_EQ(Vector2D(5.0, 4.0), f[12]);
    EXPECT_EQ(Vector2D(2.0, 1.0), f[13]);
}

TEST(ParticleSystemData2, AddParticlesException) {
    ParticleSystemData2 particleSystem;
    particleSystem.resize(12);

    try {
        particleSystem.addParticles(
            Array1<Vector2D>(
                {Vector2D(1.0, 2.0), Vector2D(4.0, 5.0)}).accessor(),
            Array1<Vector2D>(
                {Vector2D(7.0, 8.0)}).accessor(),
            Array1<Vector2D>(
                {Vector2D(5.0, 4.0), Vector2D(2.0, 1.0)}).accessor());

        EXPECT_FALSE(true) << "Invalid argument shoudl throw exception.";
    }
    catch (std::invalid_argument) {
        // Do nothing -- expected exception
    }

    EXPECT_EQ(12u, particleSystem.numberOfParticles());

    try {
        particleSystem.addParticles(
            Array1<Vector2D>(
                {Vector2D(1.0, 2.0), Vector2D(4.0, 5.0)}).accessor(),
            Array1<Vector2D>(
                {Vector2D(7.0, 8.0), Vector2D(2.0, 1.0)}).accessor(),
            Array1<Vector2D>(
                {Vector2D(5.0, 4.0)}).accessor());

        EXPECT_FALSE(true) << "Invalid argument shoudl throw exception.";
    }
    catch (std::invalid_argument) {
        // Do nothing -- expected exception
    }

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
}

TEST(ParticleSystemData2, BuildNeighborSearcher) {
    ParticleSystemData2 particleSystem;
    ParticleSystemData2::VectorData positions = {
        {0.5, 0.7},
        {0.1, 0.5},
        {0.3, 0.1},
        {0.2, 0.6},
        {0.9, 0.7},
        {0.2, 0.5},
        {0.5, 0.8},
        {0.2, 0.3},
        {0.9, 0.1},
        {0.6, 0.8},
        {0.1, 0.7},
        {0.4, 0.5},
        {0.5, 0.9},
        {0.7, 0.9},
        {0.2, 0.8},
        {0.5, 0.5},
        {0.4, 0.1},
        {0.2, 0.4},
        {0.1, 0.6},
        {0.9, 0.8}
    };
    particleSystem.addParticles(positions);

    const double radius = 0.4;
    particleSystem.buildNeighborSearcher(radius);

    auto neighborSearcher = particleSystem.neighborSearcher();
    const Vector2D searchOrigin = {0.1, 0.2};
    std::vector<size_t> found;
    neighborSearcher->forEachNearbyPoint(
        searchOrigin,
        radius,
        [&](size_t i, const Vector2D&) {
            found.push_back(i);
        });

    for (size_t ii = 0; ii < positions.size(); ++ii) {
        if (searchOrigin.distanceTo(positions[ii]) <= radius) {
            EXPECT_TRUE(
                found.end() != std::find(found.begin(), found.end(), ii));
        }
    }
}

TEST(ParticleSystemData2, BuildNeighborLists) {
    ParticleSystemData2 particleSystem;
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

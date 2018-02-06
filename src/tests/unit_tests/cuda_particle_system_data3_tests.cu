// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#include <jet/cuda_particle_system_data3.h>
#include <jet/cuda_utils.h>

#include <gtest/gtest.h>

using namespace jet;
using namespace experimental;

namespace {

Vector4F makeVector4F(float4 f) { return Vector4F{f.x, f.y, f.z, f.w}; }

struct ForEachCallback {
    int* count;

    ForEachCallback(int* cnt) : count(cnt) {}

    template <typename Index, typename Float4>
    JET_CUDA_HOST_DEVICE void operator()(size_t i, Float4 o, Index j,
                                         Float4 pt) {
        count[i] += 1;
    }
};

}  // namespace

TEST(CudaParticleSystemData3, Constructors) {
    CudaParticleSystemData3 particleSystem;
    EXPECT_EQ(0u, particleSystem.numberOfParticles());

    CudaParticleSystemData3 particleSystem2(100);
    EXPECT_EQ(100u, particleSystem2.numberOfParticles());

    size_t a0 = particleSystem2.addFloatData(2.0f);
    size_t a1 = particleSystem2.addFloatData(9.0f);
    size_t a2 = particleSystem2.addVectorData({1.0f, -3.0f, 5.0f, 4.0f});
    size_t a3 = particleSystem2.addIntData(8);

    CudaParticleSystemData3 particleSystem3(particleSystem2);
    EXPECT_EQ(100u, particleSystem3.numberOfParticles());
    auto as0 = particleSystem3.floatDataAt(a0);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(2.0f, as0[i]);
    }

    auto as1 = particleSystem3.floatDataAt(a1);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(9.0f, as1[i]);
    }

    auto as2 = particleSystem3.vectorDataAt(a2);
    for (size_t i = 0; i < 100; ++i) {
        float4 f = as2[i];
        EXPECT_FLOAT_EQ(1.0f, f.x);
        EXPECT_FLOAT_EQ(-3.0f, f.y);
        EXPECT_FLOAT_EQ(5.0f, f.z);
        EXPECT_FLOAT_EQ(4.0f, f.w);
    }

    auto as3 = particleSystem3.intDataAt(a3);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(8, as3[i]);
    }
}

TEST(CudaParticleSystemData3, Resize) {
    CudaParticleSystemData3 particleSystem;
    particleSystem.resize(12);

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
}

TEST(CudaParticleSystemData3, AddFloatData) {
    CudaParticleSystemData3 particleSystem;
    particleSystem.resize(12);

    size_t a0 = particleSystem.addFloatData(2.0f);
    size_t a1 = particleSystem.addFloatData(9.0f);

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
    EXPECT_EQ(0u, a0);
    EXPECT_EQ(1u, a1);

    auto as0 = particleSystem.floatDataAt(a0);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(2.0f, as0[i]);
    }

    auto as1 = particleSystem.floatDataAt(a1);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(9.0f, as1[i]);
    }
}

TEST(CudaParticleSystemData3, AddVectorData) {
    CudaParticleSystemData3 particleSystem;
    particleSystem.resize(12);

    size_t a0 = particleSystem.addVectorData(Vector4F(2.0f, 4.0f, -1.0f, 9.0f));
    size_t a1 = particleSystem.addVectorData(Vector4F(9.0f, -2.0f, 5.0f, 7.0f));

    EXPECT_EQ(12u, particleSystem.numberOfParticles());
    EXPECT_EQ(2u, a0);
    EXPECT_EQ(3u, a1);

    auto as0 = particleSystem.vectorDataAt(a0);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(Vector4F(2.0f, 4.0f, -1.0f, 9.0f), makeVector4F(as0[i]));
    }

    auto as1 = particleSystem.vectorDataAt(a1);
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(Vector4F(9.0f, -2.0f, 5.0f, 7.0f), makeVector4F(as1[i]));
    }
}

TEST(CudaParticleSystemData3, AddParticles) {
    CudaParticleSystemData3 particleSystem;
    particleSystem.resize(12);

    particleSystem.addParticles(
        Array1<Vector4F>({Vector4F(1.0f, 2.0f, 3.0f, 4.0f),
                          Vector4F(4.0f, 5.0f, 6.0f, 7.0f)}),
        Array1<Vector4F>({Vector4F(7.0f, 8.0f, 9.0f, 10.0f),
                          Vector4F(8.0f, 7.0f, 6.0f, 5.0f)}));

    EXPECT_EQ(14u, particleSystem.numberOfParticles());
    auto p = particleSystem.positions();
    auto v = particleSystem.velocities();

    EXPECT_EQ(Vector4F(1.0f, 2.0f, 3.0f, 4.0f), makeVector4F(p[12]));
    EXPECT_EQ(Vector4F(4.0f, 5.0f, 6.0f, 7.0f), makeVector4F(p[13]));
    EXPECT_EQ(Vector4F(7.0f, 8.0f, 9.0f, 10.0f), makeVector4F(v[12]));
    EXPECT_EQ(Vector4F(8.0f, 7.0f, 6.0f, 5.0f), makeVector4F(v[13]));
}

TEST(CudaParticleSystemData3, BuildNeighborSearcher) {
    CudaParticleSystemData3 particleSystem;
    Array1<Vector4F> positions = {
        Vector4F{0.1f, 0.0f, 0.4f, 0.0f}, Vector4F{0.6f, 0.2f, 0.6f, 0.0f},
        Vector4F{1.0f, 0.3f, 0.4f, 0.0f}, Vector4F{0.9f, 0.2f, 0.2f, 0.0f},
        Vector4F{0.8f, 0.4f, 0.9f, 0.0f}, Vector4F{0.1f, 0.6f, 0.2f, 0.0f},
        Vector4F{0.8f, 0.0f, 0.5f, 0.0f}, Vector4F{0.9f, 0.8f, 0.2f, 0.0f},
        Vector4F{0.3f, 0.5f, 0.2f, 0.0f}, Vector4F{0.1f, 0.6f, 0.6f, 0.0f},
        Vector4F{0.1f, 0.2f, 0.1f, 0.0f}, Vector4F{0.2f, 0.0f, 0.0f, 0.0f},
        Vector4F{0.2f, 0.6f, 0.1f, 0.0f}, Vector4F{0.1f, 0.3f, 0.7f, 0.0f},
        Vector4F{0.9f, 0.7f, 0.6f, 0.0f}, Vector4F{0.4f, 0.5f, 0.1f, 0.0f},
        Vector4F{0.1f, 0.1f, 0.6f, 0.0f}, Vector4F{0.7f, 0.8f, 1.0f, 0.0f},
        Vector4F{0.6f, 0.9f, 0.4f, 0.0f}, Vector4F{0.7f, 0.7f, 0.0f, 0.0f}};
    particleSystem.addParticles(positions);

    float radius = 0.4f;
    particleSystem.buildNeighborSearcher(radius);

    auto searcher = particleSystem.neighborSearcher();
    Vector4F o{0.1f, 0.2f, 0.3f, 0.0f};
    CudaArray1<float4> searchOrigin(1, toFloat4(o));
    CudaArray1<int> count(1, 0);
    searcher->forEachNearbyPoint(searchOrigin.view(), radius,
                                 ForEachCallback(count.data()));

    int ans = 0;
    for (auto p : positions) {
        if ((p - o).length() <= radius) {
            ans++;
        }
    }

    EXPECT_EQ(ans, count[0]);
}

TEST(CudaParticleSystemData3, BuildNeighborLists) {
    CudaParticleSystemData3 particleSystem;
    Array1<Vector4F> positions = {
        Vector4F{0.1f, 0.0f, 0.4f, 0.0f}, Vector4F{0.6f, 0.2f, 0.6f, 0.0f},
        Vector4F{1.0f, 0.3f, 0.4f, 0.0f}, Vector4F{0.9f, 0.2f, 0.2f, 0.0f},
        Vector4F{0.8f, 0.4f, 0.9f, 0.0f}, Vector4F{0.1f, 0.6f, 0.2f, 0.0f},
        Vector4F{0.8f, 0.0f, 0.5f, 0.0f}, Vector4F{0.9f, 0.8f, 0.2f, 0.0f},
        Vector4F{0.3f, 0.5f, 0.2f, 0.0f}, Vector4F{0.1f, 0.6f, 0.6f, 0.0f},
        Vector4F{0.1f, 0.2f, 0.1f, 0.0f}, Vector4F{0.2f, 0.0f, 0.0f, 0.0f},
        Vector4F{0.2f, 0.6f, 0.1f, 0.0f}, Vector4F{0.1f, 0.3f, 0.7f, 0.0f},
        Vector4F{0.9f, 0.7f, 0.6f, 0.0f}, Vector4F{0.4f, 0.5f, 0.1f, 0.0f},
        Vector4F{0.1f, 0.1f, 0.6f, 0.0f}, Vector4F{0.7f, 0.8f, 1.0f, 0.0f},
        Vector4F{0.6f, 0.9f, 0.4f, 0.0f}, Vector4F{0.7f, 0.7f, 0.0f, 0.0f}};
    particleSystem.addParticles(positions);

    float radius = 0.4f;
    particleSystem.buildNeighborSearcher(radius);
    particleSystem.buildNeighborLists(radius);

    Array1<size_t> ansNeighborStarts(positions.size());
    Array1<size_t> ansNeighborEnds(positions.size());

    for (size_t i = 0; i < positions.size(); ++i) {
        size_t cnt = 0;
        for (size_t j = 0; j < positions.size(); ++j) {
            if (i != j && (positions[i] - positions[j]).length() <= radius) {
                ++cnt;
            }
        }
        ansNeighborStarts[i] = cnt;
    }

    ansNeighborEnds[0] = ansNeighborStarts[0];
    for (size_t i = 1; i < ansNeighborStarts.size(); ++i) {
        ansNeighborEnds[i] = ansNeighborEnds[i - 1] + ansNeighborStarts[i];
    }
    std::transform(ansNeighborEnds.begin(), ansNeighborEnds.end(),
                   ansNeighborStarts.begin(), ansNeighborStarts.begin(),
                   std::minus<size_t>());

    auto cuNeighborStarts = particleSystem.neighborStarts();
    auto cuNeighborEnds = particleSystem.neighborEnds();

    for (size_t i = 0; i < ansNeighborStarts.size(); ++i) {
        EXPECT_EQ(ansNeighborStarts[i], cuNeighborStarts[i]) << i;
        EXPECT_EQ(ansNeighborEnds[i], cuNeighborEnds[i]) << i;
    }

    auto cuNeighborLists = particleSystem.neighborLists();
    for (size_t i = 0; i < ansNeighborStarts.size(); ++i) {
        size_t start = ansNeighborStarts[i];
        size_t end = ansNeighborEnds[i];
        for (size_t jj = start; jj < end; ++jj) {
            size_t j = cuNeighborLists[jj];
            EXPECT_LE((positions[i] - positions[j]).length(), radius);
        }
    }
}

#endif  // JET_USE_CUDA

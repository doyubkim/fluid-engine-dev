// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#include <jet/cuda_particle_system_data3.h>

#include <gtest/gtest.h>

using namespace jet;
using namespace experimental;

namespace {

Vector4F makeVector4F(float4 f) { return Vector4F{f.x, f.y, f.z, f.w}; }

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

#endif  // JET_USE_CUDA

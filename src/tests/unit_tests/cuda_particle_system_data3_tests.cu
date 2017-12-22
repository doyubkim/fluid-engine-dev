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

TEST(CudaParticleSystemData3, Constructors) {
    CudaParticleSystemData3 particles1;
    EXPECT_EQ(0u, particles1.numberOfParticles());

    CudaParticleSystemData3 particles2(15);
    EXPECT_EQ(15u, particles2.numberOfParticles());

    CudaParticleSystemData3 particles3(15);
    EXPECT_EQ(15u, particles3.numberOfParticles());
}

#endif  // JET_USE_CUDA

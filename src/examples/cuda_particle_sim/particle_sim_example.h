// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_CUDA_PARTICLE_SIM_PARTICLE_SIM_EXAMPLE_H_
#define SRC_EXAMPLES_CUDA_PARTICLE_SIM_PARTICLE_SIM_EXAMPLE_H_

#include <jet.viz/jet.viz.h>

class ParticleSimExample {
 public:
    ParticleSimExample() = default;
    virtual ~ParticleSimExample() = default;

    virtual void setup(jet::viz::GlfwWindow* window) = 0;

    virtual void onGui(jet::viz::GlfwWindow* window);
};

typedef std::shared_ptr<ParticleSimExample> ParticleSimExamplePtr;

#endif  // SRC_EXAMPLES_CUDA_PARTICLE_SIM_PARTICLE_SIM_EXAMPLE_H_

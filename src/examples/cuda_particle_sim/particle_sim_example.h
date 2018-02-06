// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_CUDA_PARTICLE_SIM_PARTICLE_SIM_EXAMPLE_H_
#define SRC_EXAMPLES_CUDA_PARTICLE_SIM_PARTICLE_SIM_EXAMPLE_H_

#include <jet.viz/jet.viz.h>
#include <jet/animation.h>

class ParticleSimExample {
 public:
    ParticleSimExample(const jet::Frame& frame);
    virtual ~ParticleSimExample() = default;

    void setup(jet::viz::GlfwWindow* window);

    void gui(jet::viz::GlfwWindow* window);

    void update();

 protected:
    virtual void onSetup(jet::viz::GlfwWindow* window);

    virtual void onGui(jet::viz::GlfwWindow* window);

    virtual void onUpdate(const jet::Frame& frame);

 private:
    jet::Frame _frame;
};

typedef std::shared_ptr<ParticleSimExample> ParticleSimExamplePtr;

#endif  // SRC_EXAMPLES_CUDA_PARTICLE_SIM_PARTICLE_SIM_EXAMPLE_H_

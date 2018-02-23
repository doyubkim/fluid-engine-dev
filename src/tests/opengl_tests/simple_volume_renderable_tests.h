// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_TESTS_OPENGL_TESTS_SIMPLE_VOLUME_RENDERABLE_TESTS_H_
#define SRC_TESTS_OPENGL_TESTS_SIMPLE_VOLUME_RENDERABLE_TESTS_H_

#include "opengl_tests.h"

#include <jet.viz/jet.viz.h>

class SimpleVolumeRenderableTests final : public OpenGLTests {
 public:
    SimpleVolumeRenderableTests() = default;

    void setup(jet::viz::GlfwWindow* window) override;

    void onGui(jet::viz::GlfwWindow* window) override;

 private:
     jet::viz::SimpleVolumeRenderablePtr _renderable;
};

#endif  // SRC_TESTS_OPENGL_TESTS_SIMPLE_VOLUME_RENDERABLE_TESTS_H_

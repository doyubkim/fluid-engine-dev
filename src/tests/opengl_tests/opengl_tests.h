// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_TESTS_OPENGL_TESTS_OPENGL_TESTS_H_
#define SRC_TESTS_OPENGL_TESTS_OPENGL_TESTS_H_

#include <jet.viz/jet.viz.h>

class OpenGLTests {
 public:
    OpenGLTests() = default;
    virtual ~OpenGLTests() = default;

    virtual void setup(jet::viz::GlfwWindow* window) = 0;

    virtual void onGui(jet::viz::GlfwWindow* window);
};

typedef std::shared_ptr<OpenGLTests> OpenGLTestsPtr;

#endif  // SRC_TESTS_OPENGL_TESTS_OPENGL_TESTS_H_

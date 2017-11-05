// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_TESTS_OPENGL_TESTS_OPENGL_TESTS_H_
#define SRC_TESTS_OPENGL_TESTS_OPENGL_TESTS_H_

#include <jet.viz/jet.viz.h>

namespace jet {

namespace viz {

class OpenGLTests {
 public:
    OpenGLTests() = default;
    virtual ~OpenGLTests() = default;

    virtual void setup(Renderer* renderer) = 0;
};

typedef std::shared_ptr<OpenGLTests> OpenGLTestsPtr;

}  // namespace viz

}  // namespace jet

#endif  // SRC_TESTS_OPENGL_TESTS_OPENGL_TESTS_H_

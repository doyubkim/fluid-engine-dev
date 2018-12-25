// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_GL

#include <common.h>

#include <jet.gfx/gl_renderer.h>

namespace jet {

namespace gfx {

GLRenderer::GLRenderer() {}

GLRenderer::~GLRenderer() {}

void GLRenderer::render() {
    Vector4F bgColor = backgroundColor().castTo<float>();
    glClearColor(bgColor.x, bgColor.y, bgColor.z, bgColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

}  // namespace gfx

}  // namespace jet

#endif  // JET_USE_GL

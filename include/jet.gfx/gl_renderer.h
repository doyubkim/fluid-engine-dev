// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_GL_RENDERER_H_
#define INCLUDE_JET_GFX_GL_RENDERER_H_

#ifdef JET_USE_GL

#include <jet.gfx/renderer.h>

namespace jet {

namespace gfx {

class GLRenderer : public Renderer {
 public:
    GLRenderer();
    virtual ~GLRenderer();

    void render() override;
};

using GLRendererPtr = std::shared_ptr<GLRenderer>;

}  // namespace gfx

}  // namespace jet

#endif  // INCLUDE_JET_GFX_GL_RENDERER_H_

#endif  // JET_USE_GL

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_RENDERER_H_
#define INCLUDE_JET_GFX_RENDERER_H_

#include <jet.gfx/camera.h>
#include <jet.gfx/viewport.h>

#include <memory>

namespace jet {

namespace gfx {

class Renderer {
 public:
    Renderer();
    virtual ~Renderer();

    virtual void render() = 0;

    const CameraPtr& camera() const;

    void setCamera(const CameraPtr& camera);

    const Vector4D& backgroundColor() const;

    void setBackgroundColor(const Vector4D& color);

 private:
    CameraPtr _camera;
    Vector4D _bgColor;
};

using RendererPtr = std::shared_ptr<Renderer>;

}  // namespace gfx

}  // namespace jet

#endif  // INCLUDE_JET_GFX_RENDERER_H_

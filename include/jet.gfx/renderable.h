// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_RENDERABLE_H_
#define INCLUDE_JET_GFX_RENDERABLE_H_

#include <memory>

namespace jet {
namespace gfx {

class Renderer;

class Renderable {
 public:
    Renderable();

    virtual ~Renderable();

    void render(Renderer* renderer);

protected:
    virtual void onInitializeGpuResources(Renderer* renderer) = 0;

    virtual void onRender(Renderer* renderer) = 0;

    void invalidateGpuResources();

private:
    bool _gpuResourcesInitialized = false;
};

//! Shared pointer type for Renderable.
using RenderablePtr = std::shared_ptr<Renderable>;

}  // namespace gfx
}  // namespace jet

#endif  // INCLUDE_JET_GFX_RENDERABLE_H_

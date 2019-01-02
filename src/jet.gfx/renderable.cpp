// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/renderable.h>

namespace jet {
namespace gfx {

Renderable::Renderable() {}

Renderable::~Renderable() {}

void Renderable::render(jet::gfx::Renderer *renderer) {
    if (!_gpuResourcesInitialized) {
        onInitializeGpuResources(renderer);
        _gpuResourcesInitialized = true;
    }
    onRender(renderer);
}

void Renderable::invalidateGpuResources() {
    _gpuResourcesInitialized = false;
}

}  // namespace gfx
}  // namespace jet

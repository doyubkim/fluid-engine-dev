// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_METAL_RENDERER_H_
#define INCLUDE_JET_GFX_METAL_RENDERER_H_

#include <jet/macros.h>

#ifdef JET_MACOSX

#include <jet.gfx/renderer.h>

namespace jet {

namespace gfx {

class MetalWindow;
class MetalPrivateDevice;
class MetalPrivateCommandQueue;

class MetalRenderer : public Renderer {
 public:
    MetalRenderer(MetalWindow* window);

    virtual ~MetalRenderer();

    void render() override;

    MetalPrivateDevice* device() const;

    MetalPrivateCommandQueue* commandQueue() const;

 private:
    MetalWindow* _window = nullptr;
    MetalPrivateDevice* _device = nullptr;
    MetalPrivateCommandQueue* _commandQueue = nullptr;
};

using MetalRendererPtr = std::shared_ptr<MetalRenderer>;

}  // namespace gfx

}  // namespace jet

#endif  // INCLUDE_JET_GFX_METAL_RENDERER_H_

#endif  // JET_MACOSX

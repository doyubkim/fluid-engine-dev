// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#import "metal_view.h"
#import "mtlpp_wrappers.h"

#import <jet.gfx/metal_renderer.h>
#import <jet.gfx/metal_window.h>

#import <MetalKit/MetalKit.h>

using namespace jet;
using namespace gfx;

namespace {

mtlpp::Drawable getCurrentDrawable(const MetalWindow* window) {
    return ns::Handle{
        (__bridge void*)((__bridge MTKView*)window->view()->GetPtr())
            .currentDrawable};
}

mtlpp::RenderPassDescriptor getCurrentRenderPassDescriptor(
    const MetalWindow* window) {
    return ns::Handle{
        (__bridge void*)((__bridge MTKView*)window->view()->GetPtr())
            .currentRenderPassDescriptor};
}

}  // namespace

MetalRenderer::MetalRenderer(MetalWindow* window) : _window(window) {
    // Create device
    _device =
        new MetalPrivateDevice(mtlpp::Device::CreateSystemDefaultDevice());

    // Create command queue
    _commandQueue =
        new MetalPrivateCommandQueue(_device->value.NewCommandQueue());
}

MetalRenderer::~MetalRenderer() {
    delete _device;
    delete _commandQueue;
}

void MetalRenderer::render() {
    mtlpp::CommandBuffer commandBuffer = _commandQueue->value.CommandBuffer();

    mtlpp::RenderPassDescriptor renderPassDesc =
            getCurrentRenderPassDescriptor(_window);
    if (renderPassDesc) {
        const auto& bg = backgroundColor();
        renderPassDesc.GetColorAttachments()[0].SetClearColor(mtlpp::ClearColor(bg.x, bg.y, bg.z, bg.w));

        mtlpp::RenderCommandEncoder renderCommandEncoder =
                commandBuffer.RenderCommandEncoder(renderPassDesc);

        // TODO: Actual rendering

        renderCommandEncoder.EndEncoding();
        commandBuffer.Present(getCurrentDrawable(_window));
    }

    commandBuffer.Commit();
    commandBuffer.WaitUntilCompleted();
}

MetalPrivateDevice* MetalRenderer::device() const { return _device; }

MetalPrivateCommandQueue* MetalRenderer::commandQueue() const {
    return _commandQueue;
}

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/persp_camera.h>
#include <jet.gfx/renderer.h>

namespace jet {
namespace gfx {

Renderer::Renderer() { _camera = std::make_shared<PerspCamera>(); }

Renderer::~Renderer() {}

void Renderer::render() {
    onRenderBegin();

    // TODO: Actual rendering
    //    for (auto &renderable : _renderables) {
    //        renderable->render(this);
    //    }

    onRenderEnd();
}

void Renderer::bindShader(const ShaderPtr &shader) { shader->bind(this); }

void Renderer::unbindShader(const ShaderPtr &shader) { shader->unbind(this); }

void Renderer::bindVertexBuffer(const VertexBufferPtr &vertexBuffer) {
    vertexBuffer->bind(this);
}

void Renderer::unbindVertexBuffer(const VertexBufferPtr &vertexBuffer) {
    vertexBuffer->unbind(this);
}

void Renderer::bindIndexBuffer(const IndexBufferPtr &indexBuffer) {
    indexBuffer->bind(this);
}

void Renderer::unbindIndexBuffer(const IndexBufferPtr &indexBuffer) {
    indexBuffer->unbind(this);
}

void Renderer::bindTexture(const Texture2Ptr &texture, unsigned int slotId) {
    texture->bind(this, slotId);
}

void Renderer::bindTexture(const Texture3Ptr &texture, unsigned int slotId) {
    texture->bind(this, slotId);
}

const CameraPtr &Renderer::camera() const { return _camera; }

void Renderer::setCamera(const CameraPtr &camera) { _camera = camera; }

const RenderStates &Renderer::renderStates() const { return _renderStates; }

void Renderer::setRenderStates(const RenderStates &states) {
    _renderStates = states;

    onSetRenderStates(states);
}

const Vector4F &Renderer::backgroundColor() const { return _backgroundColor; }

void Renderer::setBackgroundColor(const Vector4F &color) {
    _backgroundColor = color;
}

}  // namespace gfx
}  // namespace jet

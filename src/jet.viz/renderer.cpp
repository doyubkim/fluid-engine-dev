// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/renderer.h>

using namespace jet;
using namespace viz;

Renderer::Renderer() { _camera = std::make_shared<Camera>(); }

Renderer::~Renderer() {}

void Renderer::render() {
    onRenderBegin();

    for (auto& renderable : _renderables) {
        renderable->render(this);
    }

    onRenderEnd();
}

void Renderer::resize(const Viewport& viewport) {
    _viewport = viewport;

    _camera->resize(viewport);

    onResize(viewport);
}

void Renderer::addRenderable(const RenderablePtr& renderable) {
    _renderables.push_back(renderable);
}

void Renderer::clearRenderables() { _renderables.clear(); }

void Renderer::bindShader(const ShaderPtr& shader) { shader->bind(this); }

void Renderer::unbindShader(const ShaderPtr& shader) { shader->unbind(this); }

void Renderer::bindVertexBuffer(const VertexBufferPtr& vertexBuffer) {
    vertexBuffer->bind(this);
}

void Renderer::unbindVertexBuffer(const VertexBufferPtr& vertexBuffer) {
    vertexBuffer->unbind(this);
}

void Renderer::bindIndexBuffer(const IndexBufferPtr& indexBuffer) {
    indexBuffer->bind(this);
}

void Renderer::unbindIndexBuffer(const IndexBufferPtr& indexBuffer) {
    indexBuffer->unbind(this);
}

void Renderer::bindTexture(const Texture2Ptr& texture, unsigned int slotId) {
    texture->bind(this, slotId);
}

void Renderer::bindTexture(const Texture3Ptr& texture, unsigned int slotId) {
    texture->bind(this, slotId);
}

const Viewport& Renderer::viewport() const { return _viewport; }

const CameraPtr& Renderer::camera() const { return _camera; }

void Renderer::setCamera(const CameraPtr& camera) {
    // Sync state
    camera->resize(_viewport);

    _camera = camera;
}

const RenderStates& Renderer::renderStates() const { return _renderStates; }

void Renderer::setRenderStates(const RenderStates& states) {
    _renderStates = states;

    onSetRenderStates(states);
}

const Color& Renderer::backgroundColor() const { return _backgroundColor; }

void Renderer::setBackgroundColor(const Color& color) {
    _backgroundColor = color;
}

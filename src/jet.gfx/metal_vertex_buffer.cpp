// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#ifdef JET_MACOSX

#include "mtlpp_wrappers.h"

#include <jet.gfx/metal_renderer.h>
#include <jet.gfx/metal_shader.h>
#include <jet.gfx/metal_vertex_buffer.h>
#include <jet.gfx/vertex.h>

namespace jet {
namespace gfx {

MetalVertexBuffer::MetalVertexBuffer(MetalPrivateDevice *device)
    : _device(device) {}

MetalVertexBuffer::~MetalVertexBuffer() { clear(); }

void MetalVertexBuffer::update(const float *data) {
    size_t vertexSizeInBytes = VertexHelper::getSizeInBytes(vertexFormat());
    memcpy(_buffer->value.GetContents(), data,
           vertexSizeInBytes * numberOfVertices());
}

MetalPrivateBuffer *MetalVertexBuffer::buffer() const { return _buffer.get(); }

void MetalVertexBuffer::onClear() {
    _buffer = std::unique_ptr<MetalPrivateBuffer>();
}

void MetalVertexBuffer::onResize(const ShaderPtr &shader, const float *data,
                                 size_t numberOfVertices) {
    size_t vertexSizeInBytes =
        VertexHelper::getSizeInBytes(shader->vertexFormat());
    _buffer = std::make_unique<MetalPrivateBuffer>(_device->value.NewBuffer(
        data, uint32_t(vertexSizeInBytes * numberOfVertices),
        mtlpp::ResourceOptions::CpuCacheModeDefaultCache));
}

void MetalVertexBuffer::onBind(Renderer *renderer) {
    const auto mtlRenderer = dynamic_cast<const MetalRenderer *>(renderer);
    JET_ASSERT(mtlRenderer != nullptr);

    mtlRenderer->renderCommandEncoder()->value.SetVertexBuffer(buffer()->value,
                                                               /* offset */ 0,
                                                               /* buffer0 */ 0);
}

void MetalVertexBuffer::onUnbind(Renderer *renderer) {
    UNUSED_VARIABLE(renderer);
}

}  // namespace gfx
}  // namespace jet

#endif  // JET_MACOSX

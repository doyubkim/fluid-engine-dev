// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include "mtlpp_wrappers.h"

#include <jet.gfx/metal_shader.h>
#include <jet.gfx/metal_vertex_buffer.h>
#include <jet.gfx/vertex.h>

namespace jet {
namespace gfx {

MetalVertexBuffer::MetalVertexBuffer(MetalPrivateDevice *device,
                                     const ShaderPtr &shader,
                                     const ConstArrayView1<float> &vertices) {
    size_t vertexSizeInBytes =
        VertexHelper::getSizeInBytes(shader->vertexFormat());
    _buffer = new MetalPrivateBuffer(device->value.NewBuffer(
        vertices.data(), uint32_t(vertexSizeInBytes * vertices.length()),
        mtlpp::ResourceOptions::CpuCacheModeDefaultCache));
}

MetalVertexBuffer::~MetalVertexBuffer() { clear(); }

void MetalVertexBuffer::update(const float *vertices) {}

MetalPrivateBuffer *MetalVertexBuffer::buffer() const { return _buffer; }

void MetalVertexBuffer::onClear() { delete _buffer; }

void MetalVertexBuffer::onResize(const ShaderPtr &shader, const float *vertices,
                                 size_t numberOfVertices) {}

void MetalVertexBuffer::onBind(Renderer *renderer) {
    UNUSED_VARIABLE(renderer);
}

void MetalVertexBuffer::onUnbind(Renderer *renderer) {
    UNUSED_VARIABLE(renderer);
}

}  // namespace gfx
}  // namespace jet

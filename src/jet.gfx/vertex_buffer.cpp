// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/vertex_buffer.h>

namespace jet {
namespace gfx {

VertexBuffer::VertexBuffer() {}

VertexBuffer::~VertexBuffer() {}

#ifdef JET_USE_CUDA
void VertexBuffer::updateWithCuda(const ConstArrayView1<float> &) {}

void *VertexBuffer::cudaMapResources() { return nullptr; }

void VertexBuffer::cudaUnmapResources() {}
#endif

void VertexBuffer::clear() {
    _numberOfVertices = 0;
    _vertexFormat = VertexFormat::Position3;
    _shader.reset();

    onClear();
}

void VertexBuffer::resize(const ShaderPtr &shader,
                          const ConstArrayView1<float> &vertices) {
    if (vertices.isEmpty()) {
        clear();
    } else if (_shader == shader && _vertexFormat == shader->vertexFormat() &&
               _numberOfVertices == vertices.length()) {
        update(vertices);
    } else {
        clear();

        _numberOfVertices = vertices.length();
        _vertexFormat = shader->vertexFormat();
        _shader = shader;

        onResize(shader, vertices);
    }
}

void VertexBuffer::bind(Renderer *renderer) { onBind(renderer); }

void VertexBuffer::unbind(Renderer *renderer) { onUnbind(renderer); }

size_t VertexBuffer::numberOfVertices() const { return _numberOfVertices; }

VertexFormat VertexBuffer::vertexFormat() const { return _vertexFormat; }

const ShaderPtr &VertexBuffer::shader() const { return _shader; }

}  // namespace gfx
}  // namespace jet

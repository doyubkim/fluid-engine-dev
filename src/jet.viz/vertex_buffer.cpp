// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/vertex_buffer.h>

using namespace jet;
using namespace viz;

VertexBuffer::VertexBuffer() {}

VertexBuffer::~VertexBuffer() {}

#ifdef JET_USE_CUDA
void VertexBuffer::updateWithCuda(const float*) {}

void* VertexBuffer::cudaMapResources() { return nullptr; }

void VertexBuffer::cudaUnmapResources() {}
#endif

void VertexBuffer::clear() {
    _numberOfVertices = 0;
    _vertexFormat = VertexFormat::Position3;
    _shader.reset();

    onClear();
}

void VertexBuffer::resize(const ShaderPtr& shader, const float* vertices,
                          size_t numberOfVertices) {
    if (numberOfVertices == 0) {
        clear();
    } else if (_shader == shader && _vertexFormat == shader->vertexFormat() &&
               _numberOfVertices == numberOfVertices) {
        update(vertices);
    } else {
        clear();

        _numberOfVertices = numberOfVertices;
        _vertexFormat = shader->vertexFormat();
        _shader = shader;

        onResize(shader, vertices, numberOfVertices);
    }
}

void VertexBuffer::bind(Renderer* renderer) { onBind(renderer); }

void VertexBuffer::unbind(Renderer* renderer) { onUnbind(renderer); }

size_t VertexBuffer::numberOfVertices() const { return _numberOfVertices; }

VertexFormat VertexBuffer::vertexFormat() const { return _vertexFormat; }

const ShaderPtr& VertexBuffer::shader() const { return _shader; }

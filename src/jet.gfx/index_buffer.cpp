// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/index_buffer.h>

namespace jet {
namespace gfx {

IndexBuffer::IndexBuffer() {}

IndexBuffer::~IndexBuffer() {}

void IndexBuffer::clear() {
    _numberOfIndices = 0;

    onClear();
}

void IndexBuffer::resize(const VertexBufferPtr& vertexBuffer,
                         const uint32_t* indices, size_t numberOfIndices) {
    if (numberOfIndices == 0) {
        clear();
    } else if (_numberOfIndices == numberOfIndices) {
        update(indices);
    } else {
        clear();

        _numberOfIndices = numberOfIndices;

        onResize(vertexBuffer, indices, numberOfIndices);
    }
}

void IndexBuffer::bind(Renderer* renderer) { onBind(renderer); }

void IndexBuffer::unbind(Renderer* renderer) { onUnbind(renderer); }

size_t IndexBuffer::numberOfIndices() const { return _numberOfIndices; }

}  // namespace gfx
}  // namespace jet

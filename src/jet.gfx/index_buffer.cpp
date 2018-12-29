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

void IndexBuffer::resize(const VertexBufferPtr &vertexBuffer,
                         const ConstArrayView1<uint32_t> &indices) {
    if (indices.isEmpty()) {
        clear();
    } else if (_numberOfIndices == indices.length()) {
        update(indices);
    } else {
        clear();

        _numberOfIndices = indices.length();

        onResize(vertexBuffer, indices);
    }
}

void IndexBuffer::bind(Renderer *renderer) { onBind(renderer); }

void IndexBuffer::unbind(Renderer *renderer) { onUnbind(renderer); }

size_t IndexBuffer::numberOfIndices() const { return _numberOfIndices; }

}  // namespace gfx
}  // namespace jet

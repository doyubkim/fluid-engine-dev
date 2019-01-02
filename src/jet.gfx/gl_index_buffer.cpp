// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_GL

#include <common.h>

#include <jet.gfx/gl_common.h>
#include <jet.gfx/gl_index_buffer.h>
#include <jet.gfx/gl_vertex_buffer.h>

namespace jet {
namespace gfx {

GLIndexBuffer::GLIndexBuffer() : _bufferId(0) {}

GLIndexBuffer::~GLIndexBuffer() { clear(); }

void GLIndexBuffer::update(const uint32_t* indices) {
    GLsizei sizeInBytes =
        static_cast<GLsizei>(sizeof(uint32_t) * numberOfIndices());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _bufferId);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeInBytes, indices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void GLIndexBuffer::onClear() {
    if (_bufferId > 0) {
        glDeleteBuffers(1, &_bufferId);
    }

    _bufferId = 0;
}

void GLIndexBuffer::onResize(const VertexBufferPtr& vertexBuffer,
                             const uint32_t* indices, size_t numberOfIndices) {
    const auto& glVertexBuffer =
        std::dynamic_pointer_cast<GLVertexBuffer>(vertexBuffer);
    JET_ASSERT(glVertexBuffer != nullptr);

    GLuint vertexArrayId = glVertexBuffer->vertexArrayId();
    glBindVertexArray(vertexArrayId);

    glGenBuffers(1, &_bufferId);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _bufferId);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * numberOfIndices,
                 indices, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

void GLIndexBuffer::onBind(Renderer* renderer) {
    UNUSED_VARIABLE(renderer);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _bufferId);
}

void GLIndexBuffer::onUnbind(Renderer* renderer) {
    UNUSED_VARIABLE(renderer);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL

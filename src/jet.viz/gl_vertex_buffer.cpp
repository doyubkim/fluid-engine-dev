// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_GL

#include <jet.viz/gl_shader.h>
#include <jet.viz/gl_vertex_buffer.h>

using namespace jet;
using namespace viz;

static GLuint glCreateBufferWithVertexCountStrideAndData(GLuint vertexCount,
                                                         GLuint strideInBytes,
                                                         const GLfloat* data) {
    GLsizeiptr sizeInBytes = strideInBytes * vertexCount;

    GLuint bufferId = 0;

    glGenBuffers(1, &bufferId);

    if (bufferId) {
        glBindBuffer(GL_ARRAY_BUFFER, bufferId);
        glBufferData(GL_ARRAY_BUFFER, sizeInBytes, data, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    return bufferId;
}

static const GLchar* getOffset(GLsizeiptr offset) {
    return ((const GLchar*)nullptr + offset);
}

static void glEnableAttribute(GLint numberOfFloats, const char* name,
                              GLsizei strideInBytes, const GLvoid* offset,
                              GLuint programId,
                              std::map<std::string, GLuint>& map) {
    glUseProgram(programId);

    GLint attribLoc = glGetAttribLocation(programId, name);

    if (attribLoc >= 0) {
        map[name] = attribLoc;

        glVertexAttribPointer(attribLoc, numberOfFloats, GL_FLOAT, GL_FALSE,
                              strideInBytes, offset);
        glEnableVertexAttribArray(attribLoc);
    }
}

GLVertexBuffer::GLVertexBuffer() {}

GLVertexBuffer::~GLVertexBuffer() { clear(); }

void GLVertexBuffer::update(const float* vertices) {
    GLsizei strideInBytes = static_cast<GLsizei>(
        VertexHelper::getSizeInBytes(shader()->vertexFormat()));
    GLsizeiptr sizeInBytes = strideInBytes * numberOfVertices();

    glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferId);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeInBytes, vertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

unsigned int GLVertexBuffer::vertexArrayId() const { return _vertexArrayId; }

void GLVertexBuffer::onClear() {
    if (_vertexBufferId > 0) {
        glDeleteBuffers(1, &_vertexBufferId);
    }

    if (_indexBufferId > 0) {
        glDeleteBuffers(1, &_indexBufferId);
    }

    if (_vertexArrayId > 0) {
        glDeleteVertexArrays(1, &_vertexArrayId);
    }

    _attributes.clear();

    _vertexArrayId = 0;
    _vertexBufferId = 0;
    _indexBufferId = 0;
}

void GLVertexBuffer::onResize(const ShaderPtr& shader, const float* vertices,
                              size_t numberOfVertices) {
    GLShader* glShader = dynamic_cast<GLShader*>(shader.get());
    assert(glShader != nullptr);

    VertexFormat vertexFormat = glShader->vertexFormat();

    GLuint programId = glShader->program();

    glGenVertexArrays(1, &_vertexArrayId);

    if (_vertexArrayId && programId) {
        glBindVertexArray(_vertexArrayId);

        GLsizei strideInBytes =
            static_cast<GLsizei>(VertexHelper::getSizeInBytes(vertexFormat));

        _vertexBufferId = glCreateBufferWithVertexCountStrideAndData(
            static_cast<GLuint>(numberOfVertices), strideInBytes, vertices);

        glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferId);

        GLsizeiptr offset = 0;

        if (static_cast<int>(vertexFormat & VertexFormat::Position3)) {
            GLint numberOfFloats = static_cast<GLint>(
                VertexHelper::getNumberOfFloats(VertexFormat::Position3));
            glEnableAttribute(numberOfFloats, "position", strideInBytes,
                              getOffset(offset), programId, _attributes);
            offset += numberOfFloats * sizeof(float);
        }

        if (static_cast<int>(vertexFormat & VertexFormat::Normal3)) {
            GLint numberOfFloats = static_cast<GLint>(
                VertexHelper::getNumberOfFloats(VertexFormat::Normal3));
            glEnableAttribute(numberOfFloats, "normal", strideInBytes,
                              getOffset(offset), programId, _attributes);
            offset += numberOfFloats * sizeof(float);
        }

        if (static_cast<int>(vertexFormat & VertexFormat::TexCoord2)) {
            GLint numberOfFloats = static_cast<GLint>(
                VertexHelper::getNumberOfFloats(VertexFormat::TexCoord2));
            glEnableAttribute(numberOfFloats, "texCoord2", strideInBytes,
                              getOffset(offset), programId, _attributes);
            offset += numberOfFloats * sizeof(float);
        }

        if (static_cast<int>(vertexFormat & VertexFormat::TexCoord3)) {
            GLint numberOfFloats = static_cast<GLint>(
                VertexHelper::getNumberOfFloats(VertexFormat::TexCoord3));
            glEnableAttribute(numberOfFloats, "texCoord3", strideInBytes,
                              getOffset(offset), programId, _attributes);
            offset += numberOfFloats * sizeof(float);
        }

        if (static_cast<int>(vertexFormat & VertexFormat::Color4)) {
            GLint numberOfFloats = static_cast<GLint>(
                VertexHelper::getNumberOfFloats(VertexFormat::Color4));
            glEnableAttribute(numberOfFloats, "color", strideInBytes,
                              getOffset(offset), programId, _attributes);
        }
    }

    glBindVertexArray(0);
    glUseProgram(0);
}

void GLVertexBuffer::onBind(Renderer* renderer) {
    UNUSED_VARIABLE(renderer);

    glBindVertexArray(_vertexArrayId);
}

void GLVertexBuffer::onUnbind(Renderer* renderer) {
    UNUSED_VARIABLE(renderer);

    glBindVertexArray(0);
}

#endif  // JET_USE_GL

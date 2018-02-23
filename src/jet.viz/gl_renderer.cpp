// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_GL

#include "gl_preset_shaders.h"

#include <jet.viz/gl_index_buffer.h>
#include <jet.viz/gl_renderer.h>
#include <jet.viz/gl_shader.h>
#include <jet.viz/gl_texture2.h>
#include <jet.viz/gl_texture3.h>
#include <jet.viz/gl_vertex_buffer.h>

#include <string>

using namespace jet;
using namespace viz;

static GLenum convertBlendFactor(RenderStates::BlendFactor blendFactor) {
    switch (blendFactor) {
        case RenderStates::BlendFactor::Zero:
            return GL_ZERO;
        case RenderStates::BlendFactor::One:
            return GL_ONE;
        case RenderStates::BlendFactor::SrcAlpha:
            return GL_SRC_ALPHA;
        case RenderStates::BlendFactor::OneMinusSrcAlpha:
            return GL_ONE_MINUS_SRC_ALPHA;
        case RenderStates::BlendFactor::SrcColor:
            return GL_SRC_COLOR;
        case RenderStates::BlendFactor::OneMinusSrcColor:
            return GL_ONE_MINUS_SRC_COLOR;
        case RenderStates::BlendFactor::DestAlpha:
            return GL_DST_ALPHA;
        case RenderStates::BlendFactor::OneMinusDestAlpha:
            return GL_ONE_MINUS_DST_ALPHA;
        case RenderStates::BlendFactor::DestColor:
            return GL_DST_COLOR;
        case RenderStates::BlendFactor::OneMinusDestColor:
            return GL_ONE_MINUS_DST_COLOR;
        default:
            assert(false);
            return GL_ZERO;
    }
}

static GLenum convertPrimitiveType(PrimitiveType type) {
    if (type == PrimitiveType::Points) {
        return GL_POINTS;
    } else if (type == PrimitiveType::Lines) {
        return GL_LINES;
    } else if (type == PrimitiveType::LineStrip) {
        return GL_LINE_STRIP;
    } else if (type == PrimitiveType::Triangles) {
        return GL_TRIANGLES;
    } else if (type == PrimitiveType::TriangleStrip) {
        return GL_TRIANGLE_STRIP;
    } else {
        assert(false);
        return 0;
    }
}

GLRenderer::GLRenderer() {
    // Init GL stuff here
    RenderStates defaultRenderStates;
    setRenderStates(defaultRenderStates);
}

GLRenderer::~GLRenderer() {}

VertexBufferPtr GLRenderer::createVertexBuffer(const ShaderPtr& shader,
                                               const float* vertices,
                                               size_t numberOfPoints) {
    GLVertexBuffer* vertexBuffer = new GLVertexBuffer();
    vertexBuffer->resize(shader, vertices, numberOfPoints);

    return VertexBufferPtr(vertexBuffer);
}

IndexBufferPtr GLRenderer::createIndexBuffer(
    const VertexBufferPtr& vertexBuffer, const uint32_t* indices,
    size_t numberOfIndices) {
    GLIndexBuffer* indexBuffer = new GLIndexBuffer();
    indexBuffer->resize(vertexBuffer, indices, numberOfIndices);

    return IndexBufferPtr(indexBuffer);
}

Texture2Ptr GLRenderer::createTexture2(const ConstArrayAccessor2<ByteColor>& data) {
    GLTexture2* texture2 = new GLTexture2();
    texture2->setTexture(data);

    return Texture2Ptr(texture2);
}

Texture2Ptr GLRenderer::createTexture2(const ConstArrayAccessor2<Color>& data) {
    GLTexture2* texture2 = new GLTexture2();
    texture2->setTexture(data);

    return Texture2Ptr(texture2);
}

Texture3Ptr GLRenderer::createTexture3(const ConstArrayAccessor3<ByteColor>& data) {
    GLTexture3* texture3 = new GLTexture3();
    texture3->setTexture(data);

    return Texture3Ptr(texture3);
}

Texture3Ptr GLRenderer::createTexture3(const ConstArrayAccessor3<Color>& data) {
    GLTexture3* texture3 = new GLTexture3();
    texture3->setTexture(data);

    return Texture3Ptr(texture3);
}

ShaderPtr GLRenderer::createPresetShader(const std::string& shaderName) const {
    RenderParameters params;

    if (shaderName == "simple_color") {
        return std::make_shared<GLShader>(params, VertexFormat::Position3Color4,
                                          kSimpleColorShaders[0],
                                          kSimpleColorShaders[1]);
    } else if (shaderName == "simple_texture2") {
        params.add("Multiplier", Vector4F(1, 1, 1, 1));

        return std::make_shared<GLShader>(
            params, VertexFormat::Position3TexCoord2, kSimpleTexture2Shader[0],
            kSimpleTexture2Shader[1]);
    } else if (shaderName == "simple_texture3") {
        params.add("Multiplier", Vector4F(1, 1, 1, 1));

        return std::make_shared<GLShader>(
            params, VertexFormat::Position3TexCoord3, kSimpleTexture3Shader[0],
            kSimpleTexture3Shader[1]);
    } else if (shaderName == "points") {
        glEnable(GL_PROGRAM_POINT_SIZE);

        params.add("Radius", 1.f);

        return std::make_shared<GLShader>(params, VertexFormat::Position3Color4,
                                          kPointsShaders[0], kPointsShaders[1]);
    } else if (shaderName == "point_sprite") {
        params.add("Radius", 1.f);

        return std::make_shared<GLShader>(
            params, VertexFormat::Position3Color4, kPointSpriteShaders[0],
            kPointSpriteShaders[1], kPointSpriteShaders[2]);
    }

    return ShaderPtr();
}

void GLRenderer::setPrimitiveType(PrimitiveType type) { _primitiveType = type; }

void GLRenderer::draw(size_t numberOfVertices) {
    if (numberOfVertices > 0) {
        unsigned int N = static_cast<unsigned int>(numberOfVertices);

        glDrawArrays(convertPrimitiveType(_primitiveType), 0, N);
    }
}

void GLRenderer::drawIndexed(size_t numberOfIndices) {
    if (numberOfIndices > 0) {
        unsigned int N = static_cast<unsigned int>(numberOfIndices);

        glDrawElements(convertPrimitiveType(_primitiveType), N, GL_UNSIGNED_INT,
                       0);
    }
}

void GLRenderer::onRenderBegin() {
    const auto& bg = backgroundColor();
    glClearColor(bg.r, bg.g, bg.b, bg.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void GLRenderer::onRenderEnd() {}

void GLRenderer::onResize(const Viewport& viewport) {
    glViewport(static_cast<GLint>(viewport.x), static_cast<GLint>(viewport.y),
               static_cast<GLsizei>(viewport.width),
               static_cast<GLsizei>(viewport.height));
}

void GLRenderer::onSetRenderStates(const RenderStates& states) {
    switch (states.cullMode) {
        case RenderStates::CullMode::None:
            glEnable(GL_CULL_FACE);
            glCullFace(GL_NONE);
            break;
        case RenderStates::CullMode::Front:
            glEnable(GL_CULL_FACE);
            glCullFace(GL_FRONT);
            break;
        case RenderStates::CullMode::Back:
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
            break;
    }

    if (states.isFrontFaceClockWise) {
        glFrontFace(GL_CW);
    } else {
        glFrontFace(GL_CCW);
    }

    if (states.isBlendEnabled) {
        glEnable(GL_BLEND);
        glBlendFunc(convertBlendFactor(states.sourceBlendFactor),
                    convertBlendFactor(states.destinationBlendFactor));
    } else {
        glDisable(GL_BLEND);
    }

    if (states.isDepthTestEnabled) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
}

#endif  // JET_USE_GL

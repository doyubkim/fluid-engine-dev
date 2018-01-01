// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_GL

#include <jet.viz/gl_texture3.h>

using namespace jet;
using namespace viz;

GLTexture3::GLTexture3() : GLTexture(GL_TEXTURE_3D) {}

GLTexture3::~GLTexture3() {}

void GLTexture3::update(const Color* data) {
    glBindTexture(GL_TEXTURE_3D, glTextureId());

    glTexSubImage3D(glTarget(), 0, 0, 0, 0, static_cast<GLsizei>(_size.x),
                    static_cast<GLsizei>(_size.y),
                    static_cast<GLsizei>(_size.z), GL_RGBA, GL_FLOAT, data);
}

void GLTexture3::update(const ByteColor* data) {
    glBindTexture(GL_TEXTURE_3D, glTextureId());

    glTexSubImage3D(glTarget(), 0, 0, 0, 0, static_cast<GLsizei>(_size.x),
                    static_cast<GLsizei>(_size.y),
                    static_cast<GLsizei>(_size.z), GL_RGBA, GL_UNSIGNED_BYTE,
                    data);
}

void GLTexture3::onClear() { clearGLTexture(); }

void GLTexture3::onResize(const Color* data, const Size3& size) {
    _size = size;

    createGLTexture();

    auto target = glTarget();
    const auto& param = glTextureParameters();

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param.minFilter);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param.magFilter);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, param.wrapS);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, param.wrapT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, param.wrapR);

    glTexImage3D(target, 0, GL_RGBA32F, static_cast<GLsizei>(size.x),
                 static_cast<GLsizei>(size.y), static_cast<GLsizei>(size.z), 0,
                 GL_RGBA, GL_FLOAT, data);
}

void GLTexture3::onResize(const ByteColor* data, const Size3& size) {
    _size = size;

    createGLTexture();

    auto target = glTarget();
    const auto& param = glTextureParameters();

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, param.minFilter);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, param.magFilter);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, param.wrapS);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, param.wrapT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, param.wrapR);

    glTexImage3D(target, 0, GL_RGBA8, static_cast<GLsizei>(size.x),
                 static_cast<GLsizei>(size.y), static_cast<GLsizei>(size.z), 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);
}

void GLTexture3::onBind(Renderer* renderer, unsigned int slotId) {
    UNUSED_VARIABLE(renderer);

    bindGLTexture(slotId);
}

void GLTexture3::onSamplingModeChanged(const TextureSamplingMode& mode) {
    auto param = glTextureParameters();

    if (mode == TextureSamplingMode::kNearest) {
        param.magFilter = param.minFilter = GL_NEAREST;
        setGLTextureParamters(param);
    } else if (mode == TextureSamplingMode::kLinear) {
        param.magFilter = param.minFilter = GL_LINEAR;
        setGLTextureParamters(param);
    }
}

#endif  // JET_USE_GL

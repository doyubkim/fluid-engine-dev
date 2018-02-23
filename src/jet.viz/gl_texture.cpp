// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#ifdef JET_USE_GL

#include <jet.viz/gl_texture.h>

using namespace jet;
using namespace viz;

GLTextureParameters::GLTextureParameters() {
    minFilter = GL_NEAREST;
    magFilter = GL_NEAREST;
    wrapS = GL_CLAMP_TO_EDGE;
    wrapT = GL_CLAMP_TO_EDGE;
    wrapR = GL_CLAMP_TO_EDGE;
}

GLTexture::GLTexture(GLenum target) : _target(target) {}

GLTexture::~GLTexture() { clearGLTexture(); }

const GLTextureParameters& GLTexture::glTextureParameters() const {
    return _param;
}

void GLTexture::setGLTextureParamters(const GLTextureParameters& params) {
    _param = params;

    glBindTexture(_target, _texId);
    glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, _param.minFilter);
    glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, _param.magFilter);
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, _param.wrapS);
    glTexParameteri(_target, GL_TEXTURE_WRAP_T, _param.wrapT);
    glTexParameteri(_target, GL_TEXTURE_WRAP_R, _param.wrapR);
}

void GLTexture::clearGLTexture() {
    if (_texId > 0) {
        glDeleteTextures(1, &_texId);
    }

    _texId = 0;
}

void GLTexture::createGLTexture() {
    glGenTextures(1, &_texId);
    glBindTexture(_target, _texId);
}

void GLTexture::bindGLTexture(GLenum slotId) {
    // OpenGL defines GL_TEXTURE0 to GLTEXTURE31 only.
    JET_ASSERT(slotId < 32);

    glActiveTexture(GL_TEXTURE0 + slotId);
    glBindTexture(_target, _texId);
}

GLuint GLTexture::glTextureId() const { return _texId; }

GLenum GLTexture::glTarget() const { return _target; }

#endif  // JET_USE_GL

// Copyright (c) 2017 Doyub Kim
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
    minFilter = GL_NEAREST;  // GL_LINEAR;
    magFilter = GL_NEAREST;  // GL_LINEAR;
    wrapS = GL_CLAMP_TO_EDGE;
    wrapT = GL_CLAMP_TO_EDGE;
    wrapR = GL_CLAMP_TO_EDGE;
}

GLTexture::GLTexture(unsigned int target) : _target(target) {}

GLTexture::~GLTexture() {}

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

void GLTexture::bindGLTexture(unsigned int slotId) {
    // OpenGL defines GL_TEXTURE0 to GLTEXTURE31 only.
    assert(slotId < 32);

    glActiveTexture(GL_TEXTURE0 + slotId);
    glBindTexture(_target, _texId);
}

const GLTextureParameters& GLTexture::glParameters() const { return _param; }

unsigned int GLTexture::glTextureId() const { return _texId; }

unsigned int GLTexture::glTarget() const { return _target; }

#endif  // JET_USE_GL

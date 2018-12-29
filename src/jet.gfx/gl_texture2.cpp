// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_GL

#include <common.h>

#include <jet.gfx/gl_common.h>
#include <jet.gfx/gl_texture2.h>

namespace jet {
namespace gfx {

GLTexture2::GLTexture2() : GLTexture(GL_TEXTURE_2D) {}

GLTexture2::~GLTexture2() {}

void GLTexture2::update(const ConstArrayView2<Vector4F> &data) {
    JET_ASSERT(data.width() == _size.x && data.height() == _size.y);

    glBindTexture(GL_TEXTURE_2D, glTextureId());

    glTexSubImage2D(glTarget(), 0, 0, 0, static_cast<GLsizei>(_size.x),
                    static_cast<GLsizei>(_size.y), GL_RGBA, GL_FLOAT,
                    data.data());
}

void GLTexture2::update(const ConstArrayView2<Vector4B> &data) {
    JET_ASSERT(data.width() == _size.x && data.height() == _size.y);

    glBindTexture(GL_TEXTURE_2D, glTextureId());

    glTexSubImage2D(glTarget(), 0, 0, 0, static_cast<GLsizei>(_size.x),
                    static_cast<GLsizei>(_size.y), GL_RGBA, GL_UNSIGNED_BYTE,
                    data.data());
}

void GLTexture2::onClear() { clearGLTexture(); }

void GLTexture2::onSetTexture(const ConstArrayView2<Vector4F> &data) {
    _size = data.size();

    createGLTexture();

    auto target = glTarget();
    const auto &param = glTextureParameters();

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, param.minFilter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, param.magFilter);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, param.wrapS);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, param.wrapT);

    glTexImage2D(target, 0, GL_RGBA32F, static_cast<GLsizei>(_size.x),
                 static_cast<GLsizei>(_size.y), 0, GL_RGBA, GL_FLOAT,
                 data.data());
}

void GLTexture2::onSetTexture(const ConstArrayView2<Vector4B> &data) {
    _size = data.size();

    createGLTexture();

    auto target = glTarget();
    const auto &param = glTextureParameters();

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, param.minFilter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, param.magFilter);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, param.wrapS);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, param.wrapT);

    glTexImage2D(target, 0, GL_RGBA8, static_cast<GLsizei>(_size.x),
                 static_cast<GLsizei>(_size.y), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 data.data());
}

void GLTexture2::onBind(Renderer *renderer, unsigned int slotId) {
    UNUSED_VARIABLE(renderer);

    bindGLTexture(slotId);
}

void GLTexture2::onSamplingModeChanged(const TextureSamplingMode &mode) {
    auto param = glTextureParameters();

    if (mode == TextureSamplingMode::kNearest) {
        param.magFilter = param.minFilter = GL_NEAREST;
        setGLTextureParamters(param);
    } else if (mode == TextureSamplingMode::kLinear) {
        param.magFilter = param.minFilter = GL_LINEAR;
        setGLTextureParamters(param);
    }
}

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL

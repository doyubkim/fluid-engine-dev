// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GL_TEXTURE2_H_
#define INCLUDE_JET_VIZ_GL_TEXTURE2_H_

#ifdef JET_USE_GL

#include <jet.viz/gl_texture.h>
#include <jet.viz/texture2.h>
#include <jet/macros.h>

namespace jet {
namespace viz {

class GLTexture2 : public Texture2, public GLTexture {
 public:
    GLTexture2();
    virtual ~GLTexture2();

    void update(const Color* const data) override;

    void update(const ByteColor* const data) override;

 protected:
    void onClear() override;

    void onResize(const Color* const data, const Size2& size) override;

    void onResize(const ByteColor* const data, const Size2& size) override;

    void onBind(Renderer* renderer, unsigned int slotId) override;

    void onSamplingModeChanged(const TextureSamplingMode& mode) override;

 private:
    Size2 _size;
};

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_TEXTURE2_H_

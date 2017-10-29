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

    virtual void update(const float* const data) override;

    virtual void update(const uint8_t* const data) override;

 protected:
    virtual void onClear() override;

    virtual void onResize(const float* const data, const Size2& size) override;

    virtual void onResize(const uint8_t* const data,
                          const Size2& size) override;

    virtual void onBind(Renderer* renderer, unsigned int slotId) override;

 private:
    Size2 _size;
};

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_TEXTURE2_H_

// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_GL_TEXTURE3_H_
#define INCLUDE_JET_VIZ_GL_TEXTURE3_H_

#ifdef JET_USE_GL

#include <jet.viz/gl_texture.h>
#include <jet.viz/texture3.h>
#include <jet/macros.h>

namespace jet {
namespace viz {

class GLTexture3 : public Texture3, public GLTexture {
 public:
    GLTexture3();
    virtual ~GLTexture3();

    virtual void update(const float* data) override;

 protected:
    virtual void onClear() override;

    virtual void onResize(const float* data, const Size3& size) override;

    virtual void onBind(Renderer* renderer, unsigned int slotId) override;

 private:
    Size3 _size;
};

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_TEXTURE3_H_

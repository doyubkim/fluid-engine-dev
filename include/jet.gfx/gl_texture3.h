// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_GL_TEXTURE3_H_
#define INCLUDE_JET_GFX_GL_TEXTURE3_H_

#ifdef JET_USE_GL

#include <jet.gfx/gl_texture.h>
#include <jet.gfx/texture3.h>
#include <jet/macros.h>

namespace jet {
namespace gfx {

//! 3-D OpenGL texture representation.
class GLTexture3 : public Texture3, public GLTexture {
 public:
    //! Constructs an empty texture.
    GLTexture3();

    //! Destructor.
    virtual ~GLTexture3();

    //! Updates current texture with given 32-bit color data.
    void update(const ConstArrayView3<Vector4F>& data) override;

    //! Updates current texture with given 8-bit color data.
    void update(const ConstArrayView3<Vector4B>& data) override;

 private:
    Vector3UZ _size;

    void onClear() override;

    void onSetTexture(const ConstArrayView3<Vector4F>& data) override;

    void onSetTexture(const ConstArrayView3<Vector4B>& data) override;

    void onBind(Renderer* renderer, unsigned int slotId) override;

    void onSamplingModeChanged(const TextureSamplingMode& mode) override;
};

}  // namespace gfx
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_GFX_GL_TEXTURE3_H_

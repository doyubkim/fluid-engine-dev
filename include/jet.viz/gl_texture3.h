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

//! 3-D OpenGL texture representation.
class GLTexture3 : public Texture3, public GLTexture {
 public:
    //! Constructs an empty texture.
    GLTexture3();

    //! Destructor.
    virtual ~GLTexture3();

    //! Updates current texture with given 32-bit color data.
    void update(const Color* data) override;

    //! Updates current texture with given 8-bit color data.
    void update(const ByteColor* data) override;

 private:
    Size3 _size;

    void onClear() override;

    void onResize(const ConstArrayAccessor3<Color>& data) override;

    void onResize(const ConstArrayAccessor3<ByteColor>& data) override;

    void onBind(Renderer* renderer, unsigned int slotId) override;

    void onSamplingModeChanged(const TextureSamplingMode& mode) override;
};

}  // namespace viz
}  // namespace jet

#endif  // JET_USE_GL

#endif  // INCLUDE_JET_VIZ_GL_TEXTURE3_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_TEXTURE2_H_
#define INCLUDE_JET_VIZ_TEXTURE2_H_

#include <jet.viz/color.h>
#include <jet.viz/texture.h>
#include <jet/array_view.h>
#include <jet/macros.h>
#include <jet/matrix.h>

#include <cstdint>
#include <memory>

namespace jet {
namespace viz {

class Renderer;

//! Abstract base class for 2-D textures.
class Texture2 {
 public:
    //! Default constructor.
    Texture2();

    //! Destructor.
    virtual ~Texture2();

    //! Updates current texture with given 32-bit color data.
    virtual void update(const Color* data) = 0;

    //! Updates current texture with given 8-bit color data.
    virtual void update(const ByteColor* data) = 0;

    //! Clears the contents.
    void clear();

    //! Sets the texture with given 32-bit color data and size.
    void setTexture(const ConstArrayView2<Color>& data);

    //! Sets the texture with given 8-bit color data and size.
    void setTexture(const ConstArrayView2<ByteColor>& data);

    //! Binds the texture to given renderer with slot ID.
    void bind(Renderer* renderer, unsigned int slotId);

    //! Returns the size of the texture.
    const Vector2UZ& size() const;

    //! Returns the sampling mode of the texture.
    const TextureSamplingMode& samplingMode() const;

    //! Sets the sampling mode of the texture.
    void setSamplingMode(const TextureSamplingMode& mode);

 protected:
    //! Called when clear() is invoked.
    virtual void onClear() = 0;

    //! Called when setTexture(...) is invoked.
    virtual void onSetTexture(const ConstArrayView2<Color>& data) = 0;

    //! Called when setTexture(...) is invoked.
    virtual void onSetTexture(const ConstArrayView2<ByteColor>& data) = 0;

    //! Called when bind(...) is invoked.
    virtual void onBind(Renderer* renderer, unsigned int slotId) = 0;

    //! Called when sampling mode has changed.
    virtual void onSamplingModeChanged(const TextureSamplingMode& mode) = 0;

 private:
    Vector2UZ _size;
    TextureSamplingMode _samplingMode = TextureSamplingMode::kNearest;
};

typedef std::shared_ptr<Texture2> Texture2Ptr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_TEXTURE2_H_

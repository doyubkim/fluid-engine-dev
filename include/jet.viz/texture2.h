// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_TEXTURE2_H_
#define INCLUDE_JET_VIZ_TEXTURE2_H_

#include <jet/macros.h>
#include <jet/size2.h>
#include <jet.viz/color.h>
#include <jet.viz/texture.h>

#include <cstdint>
#include <memory>

namespace jet {
namespace viz {

class Renderer;

class Texture2 {
 public:
    Texture2();
    virtual ~Texture2();

    virtual void update(const Color* const data) = 0;

    virtual void update(const ByteColor* const data) = 0;

    void clear();

    void resize(const Color* const data, const Size2& size);

    void resize(const ByteColor* const data, const Size2& size);

    void bind(Renderer* renderer, unsigned int slotId);

    const Size2& size() const;

    const TextureSamplingMode& samplingMode() const;

    void setSamplingMode(const TextureSamplingMode& mode);

 protected:
    TextureSamplingMode _samplingMode = TextureSamplingMode::kNearest;

    virtual void onClear() = 0;

    virtual void onResize(const Color* const data, const Size2& size) = 0;

    virtual void onResize(const ByteColor* const data, const Size2& size) = 0;

    virtual void onBind(Renderer* renderer, unsigned int slotId) = 0;

    virtual void onSamplingModeChanged(const TextureSamplingMode& mode) = 0;

 private:
    Size2 _size;
};

typedef std::shared_ptr<Texture2> Texture2Ptr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_TEXTURE2_H_

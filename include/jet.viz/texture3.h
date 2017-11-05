// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_TEXTURE3_H_
#define INCLUDE_JET_VIZ_TEXTURE3_H_

#include <jet/macros.h>
#include <jet/size3.h>
#include <jet.viz/texture.h>

#include <memory>

namespace jet { namespace viz {

class Renderer;

class Texture3 {
 public:
    Texture3();
    virtual ~Texture3();

    virtual void update(const float* data) = 0;

    void clear();

    void resize(const float* data, const Size3& size);

    void bind(Renderer* renderer, unsigned int slotId);

    const Size3& size() const;

    const TextureSamplingMode& samplingMode() const;

    void setSamplingMode(const TextureSamplingMode& mode);

 protected:
    virtual void onClear() = 0;

    virtual void onResize(const float* data, const Size3& size) = 0;

    virtual void onBind(Renderer* renderer, unsigned int slotId) = 0;

    virtual void onSamplingModeChanged(const TextureSamplingMode& mode) = 0;

 private:
    Size3 _size;
    TextureSamplingMode _samplingMode = TextureSamplingMode::kNearest;
};

typedef std::shared_ptr<Texture3> Texture3Ptr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_TEXTURE3_H_

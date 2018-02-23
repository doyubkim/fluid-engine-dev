// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_TEXTURE_H_
#define INCLUDE_JET_VIZ_TEXTURE_H_

namespace jet {
namespace viz {

class Renderer;

//! Texture sampling modes.
enum class TextureSamplingMode : uint8_t {
    //! Sample nearest pixel.
    kNearest = 0,

    //! Linear-interpolate nearby pixels.
    kLinear = 1
};

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_TEXTURE_H_

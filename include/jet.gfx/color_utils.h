// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_COLOR_H_
#define INCLUDE_JET_GFX_COLOR_H_

#include <jet/matrix.h>

namespace jet {
namespace gfx {

class ColorUtils {
 public:
    //!
    //! Makes color with jet colormap.
    //!
    //! \param value Input scalar value in [-1, 1] range.
    //! \return New color instance.
    //!
    static Vector4F makeJet(float value);
};

}  // namespace gfx
}  // namespace jet

#endif  // INCLUDE_JET_GFX_COLOR_H_

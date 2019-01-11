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
    static Vector4F makeWhite();

    static Vector4F makeGray();

    static Vector4F makeBlack();

    static Vector4F makeRed();

    static Vector4F makeGreen();

    static Vector4F makeBlue();

    static Vector4F makeCyan();

    static Vector4F makeMagenta();

    static Vector4F makeYellow();

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

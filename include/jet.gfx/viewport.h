// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_VIEWPORT_H_
#define INCLUDE_JET_GFX_VIEWPORT_H_

#include <jet/matrix.h>

namespace jet {
namespace gfx {

struct Viewport {
    double x;
    double y;
    double width;
    double height;

    Viewport();
    Viewport(double newX, double newY, double newWidth, double newHeight);

    double aspectRatio() const;

    Vector2D center() const;

    //! Returns true if equal to the other viewport.
    bool operator==(const Viewport& other) const;

    //! Returns true if not equal to the other viewport.
    bool operator!=(const Viewport& other) const;
};

}  // namespace gfx
}  // namespace jet

#endif  // INCLUDE_JET_GFX_VIEWPORT_H_

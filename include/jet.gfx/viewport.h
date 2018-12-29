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
    float x;
    float y;
    float width;
    float height;

    Viewport();
    Viewport(float newX, float newY, float newWidth, float newHeight);

    float aspectRatio() const;

    Vector2F center() const;

    //! Returns true if equal to the other viewport.
    bool operator==(const Viewport& other) const;

    //! Returns true if not equal to the other viewport.
    bool operator!=(const Viewport& other) const;
};

}  // namespace gfx
}  // namespace jet

#endif  // INCLUDE_JET_GFX_VIEWPORT_H_

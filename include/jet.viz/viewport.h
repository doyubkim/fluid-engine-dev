// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_VIEWPORT_H_
#define INCLUDE_JET_VIZ_VIEWPORT_H_

#include <jet/vector2.h>

namespace jet { namespace viz {

struct Viewport {
    double x;
    double y;
    double width;
    double height;

    Viewport();
    Viewport(double newX, double newY, double newWidth, double newHeight);

    double aspectRatio() const;

    Vector2D center() const;
};

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_VIEWPORT_H_

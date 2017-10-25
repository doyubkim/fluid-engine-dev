// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/viewport.h>

#include <cassert>

using namespace jet;
using namespace viz;

Viewport::Viewport() : x(0), y(0), width(1), height(1) {}

Viewport::Viewport(double newX, double newY, double newWidth, double newHeight)
    : x(newX), y(newY), width(newWidth), height(newHeight) {
    assert(width > 0.0 && height > 0.0);
}

double Viewport::aspectRatio() const { return width / height; }

Vector2D Viewport::center() const {
    return Vector2D(x + width / 2.0, y + height / 2.0);
}

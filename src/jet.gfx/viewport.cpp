// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/viewport.h>

#include <cassert>

using namespace jet;
using namespace gfx;

Viewport::Viewport() : x(0), y(0), width(256), height(256) {}

Viewport::Viewport(double newX, double newY, double newWidth, double newHeight)
    : x(newX), y(newY), width(newWidth), height(newHeight) {
    JET_ASSERT(width > 0.0 && height > 0.0);
}

double Viewport::aspectRatio() const { return width / height; }

Vector2D Viewport::center() const {
    return Vector2D(x + width / 2.0, y + height / 2.0);
}

bool Viewport::operator==(const Viewport& other) const {
    return x == other.x && y == other.y && width == other.width &&
           height == other.height;
}

bool Viewport::operator!=(const Viewport& other) const {
    return !(*this == other);
}

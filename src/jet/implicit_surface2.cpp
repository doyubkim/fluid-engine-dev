// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/implicit_surface2.h>

using namespace jet;

ImplicitSurface2::ImplicitSurface2(
    const Transform2& transform, bool isNormalFlipped)
: Surface2(transform, isNormalFlipped) {
}

ImplicitSurface2::ImplicitSurface2(const ImplicitSurface2& other) :
    Surface2(other) {
}

ImplicitSurface2::~ImplicitSurface2() {
}

double ImplicitSurface2::signedDistance(const Vector2D& otherPoint) const {
    return signedDistanceLocal(transform.toLocal(otherPoint));
}

double ImplicitSurface2::closestDistanceLocal(
    const Vector2D& otherPoint) const {
    return std::fabs(signedDistanceLocal(otherPoint));
}

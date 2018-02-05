// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/implicit_surface3.h>

using namespace jet;

ImplicitSurface3::ImplicitSurface3(
    const Transform3& transform_, bool isNormalFlipped_)
: Surface3(transform_, isNormalFlipped_) {
}

ImplicitSurface3::ImplicitSurface3(const ImplicitSurface3& other) :
    Surface3(other) {
}

ImplicitSurface3::~ImplicitSurface3() {
}

double ImplicitSurface3::signedDistance(const Vector3D& otherPoint) const {
    return signedDistanceLocal(transform.toLocal(otherPoint));
}

double ImplicitSurface3::closestDistanceLocal(
    const Vector3D& otherPoint) const {
    return std::fabs(signedDistanceLocal(otherPoint));
}

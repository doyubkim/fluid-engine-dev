// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/implicit_surface3.h>

using namespace jet;

ImplicitSurface3::ImplicitSurface3(bool isNormalFlipped_)
: Surface3(isNormalFlipped_) {
}

ImplicitSurface3::ImplicitSurface3(const ImplicitSurface3& other) :
    Surface3(other) {
}

ImplicitSurface3::~ImplicitSurface3() {
}

double ImplicitSurface3::closestDistance(const Vector3D& otherPoint) const {
    return std::fabs(signedDistance(otherPoint));
}

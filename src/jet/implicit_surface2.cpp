// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/implicit_surface2.h>

using namespace jet;

ImplicitSurface2::ImplicitSurface2() {
}

ImplicitSurface2::~ImplicitSurface2() {
}

double ImplicitSurface2::closestDistance(const Vector2D& otherPoint) const {
    return std::fabs(signedDistance(otherPoint));
}

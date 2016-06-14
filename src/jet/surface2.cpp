// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface2.h>

using namespace jet;

Surface2::Surface2() {
}

Surface2::Surface2(const Surface2& other) :
    isNormalFlipped(other.isNormalFlipped) {
}

Surface2::~Surface2() {
}

bool Surface2::intersects(const Ray2D& ray) const {
    SurfaceRayIntersection2 i = closestIntersection(ray);
    return i.isIntersecting;
}

double Surface2::closestDistance(const Vector2D& otherPoint) const {
    return otherPoint.distanceTo(closestPoint(otherPoint));
}

Vector2D Surface2::closestNormal(const Vector2D& otherPoint) const {
    Vector2D normal = actualClosestNormal(otherPoint);
    return (isNormalFlipped) ? -normal : normal;
}

SurfaceRayIntersection2 Surface2::closestIntersection(
    const Ray2D& ray) const {
    SurfaceRayIntersection2 intersection = actualClosestIntersection(ray);
    intersection.normal
        = (isNormalFlipped) ? -intersection.normal : intersection.normal;
    return intersection;
}

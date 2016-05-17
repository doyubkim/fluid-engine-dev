// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface2.h>

using namespace jet;

Surface2::Surface2() {
}

Surface2::~Surface2() {
}

bool Surface2::intersects(const Ray2D& ray) const {
    SurfaceRayIntersection2 i;
    getClosestIntersection(ray, &i);
    return i.isIntersecting;
}

double Surface2::closestDistance(const Vector2D& otherPoint) const {
    return otherPoint.distanceTo(closestPoint(otherPoint));
}

Vector2D Surface2::closestNormal(const Vector2D& otherPoint) const {
    Vector2D normal = actualClosestNormal(otherPoint);
    return (_isNormalFlipped) ? -normal : normal;
}

void Surface2::setIsNormalFlipped(bool isFlipped) {
    _isNormalFlipped = isFlipped;
}

bool Surface2::isNormalFlipped() const {
    return _isNormalFlipped;
}

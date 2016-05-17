// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface3.h>

using namespace jet;

Surface3::Surface3() {
}

Surface3::~Surface3() {
}

bool Surface3::intersects(const Ray3D& ray) const {
    SurfaceRayIntersection3 i;
    getClosestIntersection(ray, &i);
    return i.isIntersecting;
}

double Surface3::closestDistance(const Vector3D& otherPoint) const {
    return otherPoint.distanceTo(closestPoint(otherPoint));
}

Vector3D Surface3::closestNormal(const Vector3D& otherPoint) const {
    Vector3D normal = actualClosestNormal(otherPoint);
    return (_isNormalFlipped) ? -normal : normal;
}

void Surface3::setIsNormalFlipped(bool isFlipped) {
    _isNormalFlipped = isFlipped;
}

bool Surface3::isNormalFlipped() const {
    return _isNormalFlipped;
}

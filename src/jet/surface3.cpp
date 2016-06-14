// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface3.h>

using namespace jet;

Surface3::Surface3() {
}

Surface3::Surface3(const Surface3& other) :
    isNormalFlipped(other.isNormalFlipped) {
}

Surface3::~Surface3() {
}

bool Surface3::intersects(const Ray3D& ray) const {
    SurfaceRayIntersection3 i = closestIntersection(ray);
    return i.isIntersecting;
}

double Surface3::closestDistance(const Vector3D& otherPoint) const {
    return otherPoint.distanceTo(closestPoint(otherPoint));
}

Vector3D Surface3::closestNormal(const Vector3D& otherPoint) const {
    Vector3D normal = actualClosestNormal(otherPoint);
    return (isNormalFlipped) ? -normal : normal;
}

SurfaceRayIntersection3 Surface3::closestIntersection(
    const Ray3D& ray) const {
    SurfaceRayIntersection3 intersection = actualClosestIntersection(ray);
    intersection.normal
        = (isNormalFlipped) ? -intersection.normal : intersection.normal;
    return intersection;
}

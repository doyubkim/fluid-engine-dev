// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface3.h>
#include <algorithm>

using namespace jet;

Surface3::Surface3(const Transform3& transform_, bool isNormalFlipped_)
: transform(transform_)
, isNormalFlipped(isNormalFlipped_) {
}

Surface3::Surface3(const Surface3& other)
: transform(other.transform)
, isNormalFlipped(other.isNormalFlipped) {
}

Surface3::~Surface3() {
}

Vector3D Surface3::closestPoint(const Vector3D& otherPoint) const {
    return transform.toWorld(closestPointLocal(transform.toLocal(otherPoint)));
}

BoundingBox3D Surface3::boundingBox() const {
    return transform.toWorld(boundingBoxLocal());
}

bool Surface3::intersects(const Ray3D& ray) const {
    return intersectsLocal(transform.toLocal(ray));
}

double Surface3::closestDistance(const Vector3D& otherPoint) const {
    return closestDistanceLocal(transform.toLocal(otherPoint));
}

SurfaceRayIntersection3 Surface3::closestIntersection(const Ray3D& ray) const {
    auto result = closestIntersectionLocal(transform.toLocal(ray));
    result.point = transform.toWorld(result.point);
    result.normal = transform.toWorldDirection(result.normal);
    result.normal *= (isNormalFlipped) ? -1.0 : 1.0;
    return result;
}

Vector3D Surface3::closestNormal(const Vector3D& otherPoint) const {
    auto result = transform.toWorldDirection(
        closestNormalLocal(transform.toLocal(otherPoint)));
    result *= (isNormalFlipped) ? -1.0 : 1.0;
    return result;
}

bool Surface3::intersectsLocal(const Ray3D& rayLocal) const {
    auto result = closestIntersectionLocal(rayLocal);
    return result.isIntersecting;
}

double Surface3::closestDistanceLocal(const Vector3D& otherPointLocal) const {
    return otherPointLocal.distanceTo(closestPointLocal(otherPointLocal));
}

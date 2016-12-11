// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface_to_implicit3.h>

using namespace jet;

SurfaceToImplicit3::SurfaceToImplicit3(
    const Surface3Ptr& surface,
    bool isNormalFlipped_)
: ImplicitSurface3(isNormalFlipped_)
, _surface(surface) {
}

SurfaceToImplicit3::SurfaceToImplicit3(const SurfaceToImplicit3& other) :
    ImplicitSurface3(other),
    _surface(other._surface) {
}

Surface3Ptr SurfaceToImplicit3::surface() const {
    return _surface;
}

Vector3D SurfaceToImplicit3::closestPoint(
    const Vector3D& otherPoint) const {
    return _surface->closestPoint(otherPoint);
}

Vector3D SurfaceToImplicit3::actualClosestNormal(
    const Vector3D& otherPoint) const {
    return _surface->closestNormal(otherPoint);
}

double SurfaceToImplicit3::closestDistance(
    const Vector3D& otherPoint) const {
    return _surface->closestDistance(otherPoint);
}

bool SurfaceToImplicit3::intersects(const Ray3D& ray) const {
    return _surface->intersects(ray);
}

SurfaceRayIntersection3 SurfaceToImplicit3::actualClosestIntersection(
    const Ray3D& ray) const {
    return _surface->closestIntersection(ray);
}

BoundingBox3D SurfaceToImplicit3::boundingBox() const {
    return _surface->boundingBox();
}

double SurfaceToImplicit3::signedDistance(
    const Vector3D& otherPoint) const {
    Vector3D x = _surface->closestPoint(otherPoint);
    Vector3D n = _surface->closestNormal(otherPoint);
    n = (isNormalFlipped) ? -n : n;
    if (n.dot(otherPoint - x) < 0.0) {
        return -x.distanceTo(otherPoint);
    } else {
        return x.distanceTo(otherPoint);
    }
}

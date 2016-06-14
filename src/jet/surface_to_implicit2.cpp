// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface_to_implicit2.h>

using namespace jet;

SurfaceToImplicit2::SurfaceToImplicit2(
    const Surface2Ptr& surface) : _surface(surface) {
}

SurfaceToImplicit2::SurfaceToImplicit2(const SurfaceToImplicit2& other) :
    ImplicitSurface2(other),
    _surface(other._surface) {
}

Surface2Ptr SurfaceToImplicit2::surface() const {
    return _surface;
}

Vector2D SurfaceToImplicit2::closestPoint(
    const Vector2D& otherPoint) const {
    return _surface->closestPoint(otherPoint);
}

Vector2D SurfaceToImplicit2::actualClosestNormal(
    const Vector2D& otherPoint) const {
    return _surface->closestNormal(otherPoint);
}

double SurfaceToImplicit2::closestDistance(
    const Vector2D& otherPoint) const {
    return _surface->closestDistance(otherPoint);
}

bool SurfaceToImplicit2::intersects(const Ray2D& ray) const {
    return _surface->intersects(ray);
}

SurfaceRayIntersection2 SurfaceToImplicit2::actualClosestIntersection(
    const Ray2D& ray) const {
    return _surface->closestIntersection(ray);
}

BoundingBox2D SurfaceToImplicit2::boundingBox() const {
    return _surface->boundingBox();
}

double SurfaceToImplicit2::signedDistance(
    const Vector2D& otherPoint) const {
    Vector2D x = _surface->closestPoint(otherPoint);
    Vector2D n = _surface->closestNormal(otherPoint);
    n = (isNormalFlipped) ? -n : n;
    if (n.dot(otherPoint - x) < 0.0) {
        return -x.distanceTo(otherPoint);
    } else {
        return x.distanceTo(otherPoint);
    }
}

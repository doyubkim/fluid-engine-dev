// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface_to_implicit2.h>

using namespace jet;

SurfaceToImplicit2::SurfaceToImplicit2(
    const Surface2Ptr& surface,
    const Transform2& transform,
    bool isNormalFlipped)
: ImplicitSurface2(transform, isNormalFlipped)
, _surface(surface) {
}

SurfaceToImplicit2::SurfaceToImplicit2(const SurfaceToImplicit2& other) :
    ImplicitSurface2(other),
    _surface(other._surface) {
}

Surface2Ptr SurfaceToImplicit2::surface() const {
    return _surface;
}

Vector2D SurfaceToImplicit2::closestPointLocal(
    const Vector2D& otherPoint) const {
    return _surface->closestPoint(otherPoint);
}

Vector2D SurfaceToImplicit2::closestNormalLocal(
    const Vector2D& otherPoint) const {
    return _surface->closestNormal(otherPoint);
}

double SurfaceToImplicit2::closestDistanceLocal(
    const Vector2D& otherPoint) const {
    return _surface->closestDistance(otherPoint);
}

bool SurfaceToImplicit2::intersectsLocal(const Ray2D& ray) const {
    return _surface->intersects(ray);
}

SurfaceRayIntersection2 SurfaceToImplicit2::closestIntersectionLocal(
    const Ray2D& ray) const {
    return _surface->closestIntersection(ray);
}

BoundingBox2D SurfaceToImplicit2::boundingBoxLocal() const {
    return _surface->boundingBox();
}

double SurfaceToImplicit2::signedDistanceLocal(
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


SurfaceToImplicit2::Builder&
SurfaceToImplicit2::Builder::withSurface(const Surface2Ptr& surface) {
    _surface = surface;
    return *this;
}

SurfaceToImplicit2
SurfaceToImplicit2::Builder::build() const {
    return SurfaceToImplicit2(_surface, _transform, _isNormalFlipped);
}

SurfaceToImplicit2Ptr
SurfaceToImplicit2::Builder::makeShared() const {
    return std::shared_ptr<SurfaceToImplicit2>(
        new SurfaceToImplicit2(
            _surface,
            _transform,
            _isNormalFlipped),
        [] (SurfaceToImplicit2* obj) {
            delete obj;
        });
}

// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/surface_to_implicit2.h>
#include <pch.h>

using namespace jet;

SurfaceToImplicit2::SurfaceToImplicit2(const Surface2Ptr& surface,
                                       const Transform2& transform,
                                       bool isNormalFlipped)
    : ImplicitSurface2(transform, isNormalFlipped), _surface(surface) {}

SurfaceToImplicit2::SurfaceToImplicit2(const SurfaceToImplicit2& other)
    : ImplicitSurface2(other), _surface(other._surface) {}

bool SurfaceToImplicit2::isBounded() const { return _surface->isBounded(); }

void SurfaceToImplicit2::updateQueryEngine() { _surface->updateQueryEngine(); }

bool SurfaceToImplicit2::isValidGeometry() const {
    return _surface->isValidGeometry();
}

Surface2Ptr SurfaceToImplicit2::surface() const { return _surface; }

SurfaceToImplicit2::Builder SurfaceToImplicit2::builder() { return Builder(); }

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

bool SurfaceToImplicit2::isInsideLocal(const Vector2D& otherPoint) const {
    return _surface->isInside(otherPoint);
}

double SurfaceToImplicit2::signedDistanceLocal(
    const Vector2D& otherPoint) const {
    Vector2D x = _surface->closestPoint(otherPoint);
    bool inside = _surface->isInside(otherPoint);
    return (inside) ? -x.distanceTo(otherPoint) : x.distanceTo(otherPoint);
}

SurfaceToImplicit2::Builder& SurfaceToImplicit2::Builder::withSurface(
    const Surface2Ptr& surface) {
    _surface = surface;
    return *this;
}

SurfaceToImplicit2 SurfaceToImplicit2::Builder::build() const {
    return SurfaceToImplicit2(_surface, _transform, _isNormalFlipped);
}

SurfaceToImplicit2Ptr SurfaceToImplicit2::Builder::makeShared() const {
    return std::shared_ptr<SurfaceToImplicit2>(
        new SurfaceToImplicit2(_surface, _transform, _isNormalFlipped),
        [](SurfaceToImplicit2* obj) { delete obj; });
}

// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface_set2.h>

#include <algorithm>
#include <limits>
#include <vector>

using namespace jet;

SurfaceSet2::SurfaceSet2() {
}

SurfaceSet2::SurfaceSet2(
    const std::vector<Surface2Ptr>& others,
    const Transform2& transform,
    bool isNormalFlipped)
: Surface2(transform, isNormalFlipped)
, _surfaces(others) {
}

SurfaceSet2::SurfaceSet2(const SurfaceSet2& other)
: Surface2(other)
, _surfaces(other._surfaces) {
}

size_t SurfaceSet2::numberOfSurfaces() const {
    return _surfaces.size();
}

const Surface2Ptr& SurfaceSet2::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void SurfaceSet2::addSurface(const Surface2Ptr& surface) {
    _surfaces.push_back(surface);
}

Vector2D SurfaceSet2::closestPointLocal(const Vector2D& otherPoint) const {
    Vector2D result(
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max());

    double minimumDistance = std::numeric_limits<double>::max();

    for (const auto& surface : _surfaces) {
        Vector2D localClosestPoint = surface->closestPoint(otherPoint);
        double localDistance = surface->closestDistance(otherPoint);

        if (localDistance < minimumDistance) {
            result = localClosestPoint;
            minimumDistance = localDistance;
        }
    }

    return result;
}

Vector2D SurfaceSet2::closestNormalLocal(const Vector2D& otherPoint) const {
    Vector2D result(1, 0);

    double minimumDistance = std::numeric_limits<double>::max();

    for (const auto& surface : _surfaces) {
        Vector2D localClosestNormal = surface->closestNormal(otherPoint);
        double localDistance = surface->closestDistance(otherPoint);

        if (localDistance < minimumDistance) {
            result = localClosestNormal;
            minimumDistance = localDistance;
        }
    }

    return result;
}

double SurfaceSet2::closestDistanceLocal(const Vector2D& otherPoint) const {
    double minimumDistance = std::numeric_limits<double>::max();

    for (const auto& surface : _surfaces) {
        double localDistance = surface->closestDistance(otherPoint);

        minimumDistance = std::min(minimumDistance, localDistance);
    }

    return minimumDistance;
}

bool SurfaceSet2::intersectsLocal(const Ray2D& ray) const {
    for (const auto& surface : _surfaces) {
        if (surface->intersects(ray)) {
            return true;
        }
    }

    return false;
}

SurfaceRayIntersection2 SurfaceSet2::closestIntersectionLocal(
    const Ray2D& ray) const {
    SurfaceRayIntersection2 intersection;
    double tMin = kMaxD;

    for (const auto& surface : _surfaces) {
        SurfaceRayIntersection2 localResult;
        localResult = surface->closestIntersection(ray);

        if (localResult.isIntersecting && localResult.t < tMin) {
            intersection = localResult;
            tMin = localResult.t;
        }
    }

    return intersection;
}

BoundingBox2D SurfaceSet2::boundingBoxLocal() const {
    BoundingBox2D bbox;
    for (const auto& surface : _surfaces) {
        bbox.merge(surface->boundingBox());
    }

    return bbox;
}

SurfaceSet2::Builder SurfaceSet2::builder() {
    return Builder();
}


SurfaceSet2::Builder&
SurfaceSet2::Builder::withSurfaces(const std::vector<Surface2Ptr>& others) {
    _surfaces = others;
    return *this;
}

SurfaceSet2 SurfaceSet2::Builder::build() const {
    return SurfaceSet2(_surfaces, _transform, _isNormalFlipped);
}

SurfaceSet2Ptr SurfaceSet2::Builder::makeShared() const {
    return std::shared_ptr<SurfaceSet2>(
        new SurfaceSet2(_surfaces, _transform, _isNormalFlipped),
        [] (SurfaceSet2* obj) {
            delete obj;
        });
}

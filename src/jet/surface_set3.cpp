// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface_set3.h>

#include <algorithm>
#include <limits>
#include <vector>

using namespace jet;

SurfaceSet3::SurfaceSet3() {
}

SurfaceSet3::SurfaceSet3(
    const std::vector<Surface3Ptr>& others,
    bool isNormalFlipped)
: Surface3(isNormalFlipped)
, _surfaces(others) {
}

SurfaceSet3::SurfaceSet3(const SurfaceSet3& other) :
    Surface3(other),
    _surfaces(other._surfaces) {
}

size_t SurfaceSet3::numberOfSurfaces() const {
    return _surfaces.size();
}

const Surface3Ptr& SurfaceSet3::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void SurfaceSet3::addSurface(const Surface3Ptr& surface) {
    _surfaces.push_back(surface);
}

Vector3D SurfaceSet3::closestPoint(const Vector3D& otherPoint) const {
    Vector3D result(
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max());

    double minimumDistance = std::numeric_limits<double>::max();

    for (const auto& surface : _surfaces) {
        Vector3D localClosestPoint = surface->closestPoint(otherPoint);
        double localDistance = surface->closestDistance(otherPoint);

        if (localDistance < minimumDistance) {
            result = localClosestPoint;
            minimumDistance = localDistance;
        }
    }

    return result;
}

Vector3D SurfaceSet3::actualClosestNormal(const Vector3D& otherPoint) const {
    Vector3D result(1, 0, 0);

    double minimumDistance = std::numeric_limits<double>::max();

    for (const auto& surface : _surfaces) {
        Vector3D localClosestNormal = surface->closestNormal(otherPoint);
        double localDistance = surface->closestDistance(otherPoint);

        if (localDistance < minimumDistance) {
            result = localClosestNormal;
            minimumDistance = localDistance;
        }
    }

    return result;
}

double SurfaceSet3::closestDistance(const Vector3D& otherPoint) const {
    double minimumDistance = std::numeric_limits<double>::max();

    for (const auto& surface : _surfaces) {
        double localDistance = surface->closestDistance(otherPoint);

        minimumDistance = std::min(minimumDistance, localDistance);
    }

    return minimumDistance;
}

bool SurfaceSet3::intersects(const Ray3D& ray) const {
    for (const auto& surface : _surfaces) {
        if (surface->intersects(ray)) {
            return true;
        }
    }

    return false;
}

SurfaceRayIntersection3 SurfaceSet3::actualClosestIntersection(
    const Ray3D& ray) const {
    SurfaceRayIntersection3 intersection;
    double tMin = std::numeric_limits<double>::max();

    for (const auto& surface : _surfaces) {
        SurfaceRayIntersection3 localResult =
            surface->closestIntersection(ray);

        if (localResult.isIntersecting && localResult.t < tMin) {
            intersection = localResult;
            tMin = localResult.t;
        }
    }

    return intersection;
}

BoundingBox3D SurfaceSet3::boundingBox() const {
    BoundingBox3D bbox;
    for (const auto& surface : _surfaces) {
        bbox.merge(surface->boundingBox());
    }

    return bbox;
}

SurfaceSet3::Builder SurfaceSet3::builder() {
    return Builder();
}


SurfaceSet3::Builder&
SurfaceSet3::Builder::withIsNormalFlipped(bool isNormalFlipped) {
    _isNormalFlipped = isNormalFlipped;
    return *this;
}

SurfaceSet3::Builder&
SurfaceSet3::Builder::withSurfaces(const std::vector<Surface3Ptr>& others) {
    _surfaces = others;
    return *this;
}

SurfaceSet3 SurfaceSet3::Builder::build() const {
    return SurfaceSet3(_surfaces, _isNormalFlipped);
}

SurfaceSet3Ptr SurfaceSet3::Builder::makeShared() const {
    return std::shared_ptr<SurfaceSet3>(
        new SurfaceSet3(_surfaces, _isNormalFlipped),
        [] (SurfaceSet3* obj) {
            delete obj;
        });
}

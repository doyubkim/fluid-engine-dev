// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface_set2.h>

#include <algorithm>
#include <limits>

using namespace jet;

SurfaceSet2::SurfaceSet2() {
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

Vector2D SurfaceSet2::closestPoint(const Vector2D& otherPoint) const {
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

Vector2D SurfaceSet2::actualClosestNormal(const Vector2D& otherPoint) const {
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

double SurfaceSet2::closestDistance(const Vector2D& otherPoint) const {
    double minimumDistance = std::numeric_limits<double>::max();

    for (const auto& surface : _surfaces) {
        double localDistance = surface->closestDistance(otherPoint);

        minimumDistance = std::min(minimumDistance, localDistance);
    }

    return minimumDistance;
}

bool SurfaceSet2::intersects(const Ray2D& ray) const {
    for (const auto& surface : _surfaces) {
        if (surface->intersects(ray)) {
            return true;
        }
    }

    return false;
}

SurfaceRayIntersection2 SurfaceSet2::closestIntersection(
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

BoundingBox2D SurfaceSet2::boundingBox() const {
    BoundingBox2D bbox;
    for (const auto& surface : _surfaces) {
        bbox.merge(surface->boundingBox());
    }

    return bbox;
}

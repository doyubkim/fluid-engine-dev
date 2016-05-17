// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/surface_set3.h>

#include <algorithm>
#include <limits>

using namespace jet;

SurfaceSet3::SurfaceSet3() {
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

void SurfaceSet3::getClosestIntersection(
    const Ray3D& ray,
    SurfaceRayIntersection3* intersection) const {
    double tMin = std::numeric_limits<double>::max();

    for (const auto& surface : _surfaces) {
        SurfaceRayIntersection3 localResult;
        surface->getClosestIntersection(ray, &localResult);

        if (localResult.isIntersecting && localResult.t < tMin) {
            *intersection = localResult;
            tMin = localResult.t;
        }
    }
}

BoundingBox3D SurfaceSet3::boundingBox() const {
    BoundingBox3D bbox;
    for (const auto& surface : _surfaces) {
        bbox.merge(surface->boundingBox());
    }

    return bbox;
}

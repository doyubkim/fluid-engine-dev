// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/implicit_surface_set3.h>
#include <jet/surface_to_implicit3.h>

#include <algorithm>
#include <limits>

using namespace jet;

ImplicitSurfaceSet3::ImplicitSurfaceSet3() {
}

ImplicitSurfaceSet3::ImplicitSurfaceSet3(const ImplicitSurfaceSet3& other) :
    ImplicitSurface3(other),
    _surfaces(other._surfaces) {
}

size_t ImplicitSurfaceSet3::numberOfSurfaces() const {
    return _surfaces.size();
}

const ImplicitSurface3Ptr& ImplicitSurfaceSet3::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void ImplicitSurfaceSet3::addExplicitSurface(const Surface3Ptr& surface) {
    _surfaces.push_back(std::make_shared<SurfaceToImplicit3>(surface));
}

void ImplicitSurfaceSet3::addSurface(const ImplicitSurface3Ptr& surface) {
    _surfaces.push_back(surface);
}

Vector3D ImplicitSurfaceSet3::closestPoint(const Vector3D& otherPoint) const {
    Vector3D result(
        kMaxD,
        kMaxD,
        kMaxD);

    double minimumDistance = kMaxD;

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

double ImplicitSurfaceSet3::closestDistance(const Vector3D& otherPoint) const {
    double minimumDistance = kMaxD;

    for (const auto& surface : _surfaces) {
        double localDistance = surface->closestDistance(otherPoint);
        minimumDistance = std::min(localDistance, minimumDistance);
    }

    return minimumDistance;
}

Vector3D ImplicitSurfaceSet3::actualClosestNormal(
    const Vector3D& otherPoint) const {
    Vector3D result(1, 0, 0);

    double minimumDistance = kMaxD;

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

bool ImplicitSurfaceSet3::intersects(const Ray3D& ray) const {
    for (const auto& surface : _surfaces) {
        if (surface->intersects(ray)) {
            return true;
        }
    }

    return false;
}

SurfaceRayIntersection3 ImplicitSurfaceSet3::actualClosestIntersection(
    const Ray3D& ray) const {
    SurfaceRayIntersection3 intersection;
    double tMin = kMaxD;

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

BoundingBox3D ImplicitSurfaceSet3::boundingBox() const {
    BoundingBox3D bbox;
    for (const auto& surface : _surfaces) {
        bbox.merge(surface->boundingBox());
    }

    return bbox;
}

double ImplicitSurfaceSet3::signedDistance(const Vector3D& otherPoint) const {
    double sdf = kMaxD;
    for (const auto& surface : _surfaces) {
        sdf = std::min(sdf, surface->signedDistance(otherPoint));
    }

    return sdf;
}

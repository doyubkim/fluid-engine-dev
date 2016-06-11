// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/implicit_surface_set2.h>
#include <jet/surface_to_implicit2.h>

#include <algorithm>
#include <limits>

using namespace jet;

ImplicitSurfaceSet2::ImplicitSurfaceSet2() {
}

size_t ImplicitSurfaceSet2::numberOfSurfaces() const {
    return _surfaces.size();
}

const ImplicitSurface2Ptr& ImplicitSurfaceSet2::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void ImplicitSurfaceSet2::addSurface(const Surface2Ptr& surface) {
    _surfaces.push_back(std::make_shared<SurfaceToImplicit2>(surface));
}

void ImplicitSurfaceSet2::addImplicitSurface(
    const ImplicitSurface2Ptr& surface) {
    _surfaces.push_back(surface);
}

Vector2D ImplicitSurfaceSet2::closestPoint(const Vector2D& otherPoint) const {
    Vector2D result(
        kMaxD,
        kMaxD);

    double minimumDistance = kMaxD;

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

Vector2D ImplicitSurfaceSet2::actualClosestNormal(
    const Vector2D& otherPoint) const {
    Vector2D result(1, 0);

    double minimumDistance = kMaxD;

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

bool ImplicitSurfaceSet2::intersects(const Ray2D& ray) const {
    for (const auto& surface : _surfaces) {
        if (surface->intersects(ray)) {
            return true;
        }
    }

    return false;
}

SurfaceRayIntersection2 ImplicitSurfaceSet2::closestIntersection(
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

BoundingBox2D ImplicitSurfaceSet2::boundingBox() const {
    BoundingBox2D bbox;
    for (const auto& surface : _surfaces) {
        bbox.merge(surface->boundingBox());
    }

    return bbox;
}

double ImplicitSurfaceSet2::signedDistance(const Vector2D& otherPoint) const {
    double sdf = kMaxD;
    for (const auto& surface : _surfaces) {
        sdf = std::min(sdf, surface->signedDistance(otherPoint));
    }

    return sdf;
}

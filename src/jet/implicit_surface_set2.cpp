// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/implicit_surface_set2.h>
#include <jet/surface_to_implicit2.h>

#include <algorithm>
#include <limits>
#include <vector>

using namespace jet;

ImplicitSurfaceSet2::ImplicitSurfaceSet2() {
}

ImplicitSurfaceSet2::ImplicitSurfaceSet2(
    const std::vector<ImplicitSurface2Ptr>& surfaces)
: _surfaces(surfaces) {
}

ImplicitSurfaceSet2::ImplicitSurfaceSet2(
    const std::vector<Surface2Ptr>& surfaces) {
    for (const auto& surface : surfaces) {
        addExplicitSurface(surface);
    }
}

ImplicitSurfaceSet2::ImplicitSurfaceSet2(const ImplicitSurfaceSet2& other) :
    ImplicitSurface2(other),
    _surfaces(other._surfaces) {
}

size_t ImplicitSurfaceSet2::numberOfSurfaces() const {
    return _surfaces.size();
}

const ImplicitSurface2Ptr& ImplicitSurfaceSet2::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void ImplicitSurfaceSet2::addExplicitSurface(const Surface2Ptr& surface) {
    _surfaces.push_back(std::make_shared<SurfaceToImplicit2>(surface));
}

void ImplicitSurfaceSet2::addSurface(const ImplicitSurface2Ptr& surface) {
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

double ImplicitSurfaceSet2::closestDistance(const Vector2D& otherPoint) const {
    double minimumDistance = kMaxD;

    for (const auto& surface : _surfaces) {
        double localDistance = surface->closestDistance(otherPoint);
        minimumDistance = std::min(localDistance, minimumDistance);
    }

    return minimumDistance;
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

SurfaceRayIntersection2 ImplicitSurfaceSet2::actualClosestIntersection(
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

ImplicitSurfaceSet2::Builder ImplicitSurfaceSet2::builder() {
    return Builder();
}



ImplicitSurfaceSet2::Builder&
ImplicitSurfaceSet2::Builder::withSurfaces(
    const std::vector<ImplicitSurface2Ptr>& surfaces) {
    _surfaces = surfaces;
    return *this;
}

ImplicitSurfaceSet2::Builder&
ImplicitSurfaceSet2::Builder::withExplicitSurfaces(
    const std::vector<Surface2Ptr>& surfaces) {
    _surfaces.clear();
    for (const auto& surface : surfaces) {
        _surfaces.push_back(std::make_shared<SurfaceToImplicit2>(surface));
    }
    return *this;
}

ImplicitSurfaceSet2 ImplicitSurfaceSet2::Builder::build() const {
    return ImplicitSurfaceSet2(_surfaces);
}

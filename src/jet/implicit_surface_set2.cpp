// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/implicit_surface_set2.h>
#include <jet/surface_to_implicit2.h>

using namespace jet;

ImplicitSurfaceSet2::ImplicitSurfaceSet2() {}

ImplicitSurfaceSet2::ImplicitSurfaceSet2(
    const Array1<ImplicitSurface2Ptr>& surfaces,
    const Transform2& transform, bool isNormalFlipped)
    : ImplicitSurface2(transform, isNormalFlipped), _surfaces(surfaces) {
    for (auto surface : _surfaces) {
        if (!surface->isBounded()) {
            _unboundedSurfaces.append(surface);
        }
    }
    invalidateBvh();
}

ImplicitSurfaceSet2::ImplicitSurfaceSet2(
    const Array1<Surface2Ptr>& surfaces, const Transform2& transform,
    bool isNormalFlipped)
    : ImplicitSurface2(transform, isNormalFlipped) {
    for (const auto& surface : surfaces) {
        addExplicitSurface(surface);
    }
}

ImplicitSurfaceSet2::ImplicitSurfaceSet2(const ImplicitSurfaceSet2& other)
    : ImplicitSurface2(other),
      _surfaces(other._surfaces),
      _unboundedSurfaces(other._unboundedSurfaces) {}

void ImplicitSurfaceSet2::updateQueryEngine() { buildBvh(); }

bool ImplicitSurfaceSet2::isValidGeometry() const {
    // All surfaces should be valid.
    for (auto surface : _surfaces) {
        if (!surface->isValidGeometry()) {
            return false;
        }
    }

    // Empty set is not valid.
    return !_surfaces.isEmpty();
}

size_t ImplicitSurfaceSet2::numberOfSurfaces() const {
    return _surfaces.length();
}

const ImplicitSurface2Ptr& ImplicitSurfaceSet2::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void ImplicitSurfaceSet2::addExplicitSurface(const Surface2Ptr& surface) {
    addSurface(std::make_shared<SurfaceToImplicit2>(surface));
}

void ImplicitSurfaceSet2::addSurface(const ImplicitSurface2Ptr& surface) {
    _surfaces.append(surface);
    if (!surface->isBounded()) {
        _unboundedSurfaces.append(surface);
    }
    invalidateBvh();
}

Vector2D ImplicitSurfaceSet2::closestPointLocal(
    const Vector2D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface2Ptr& surface,
                                 const Vector2D& pt) {
        return surface->closestDistance(pt);
    };

    Vector2D result{kMaxD, kMaxD};
    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    if (queryResult.item != nullptr) {
        result = (*queryResult.item)->closestPoint(otherPoint);
    }

    double minDist = queryResult.distance;
    for (auto surface : _unboundedSurfaces) {
        auto pt = surface->closestPoint(otherPoint);
        double dist = pt.distanceTo(otherPoint);
        if (dist < minDist) {
            minDist = dist;
            result = surface->closestPoint(otherPoint);
        }
    }

    return result;
}

double ImplicitSurfaceSet2::closestDistanceLocal(
    const Vector2D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface2Ptr& surface,
                                 const Vector2D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);

    double minDist = queryResult.distance;
    for (auto surface : _unboundedSurfaces) {
        auto pt = surface->closestPoint(otherPoint);
        double dist = pt.distanceTo(otherPoint);
        if (dist < minDist) {
            minDist = dist;
        }
    }

    return minDist;
}

Vector2D ImplicitSurfaceSet2::closestNormalLocal(
    const Vector2D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface2Ptr& surface,
                                 const Vector2D& pt) {
        return surface->closestDistance(pt);
    };

    Vector2D result{1.0, 0.0};
    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    if (queryResult.item != nullptr) {
        result = (*queryResult.item)->closestNormal(otherPoint);
    }

    double minDist = queryResult.distance;
    for (auto surface : _unboundedSurfaces) {
        auto pt = surface->closestPoint(otherPoint);
        double dist = pt.distanceTo(otherPoint);
        if (dist < minDist) {
            minDist = dist;
            result = surface->closestNormal(otherPoint);
        }
    }

    return result;
}

bool ImplicitSurfaceSet2::intersectsLocal(const Ray2D& ray) const {
    buildBvh();

    const auto testFunc = [](const Surface2Ptr& surface, const Ray2D& ray) {
        return surface->intersects(ray);
    };

    bool result = _bvh.intersects(ray, testFunc);
    for (auto surface : _unboundedSurfaces) {
        result |= surface->intersects(ray);
    }

    return result;
}

SurfaceRayIntersection2 ImplicitSurfaceSet2::closestIntersectionLocal(
    const Ray2D& ray) const {
    buildBvh();

    const auto testFunc = [](const Surface2Ptr& surface, const Ray2D& ray) {
        SurfaceRayIntersection2 result = surface->closestIntersection(ray);
        return result.distance;
    };

    const auto queryResult = _bvh.closestIntersection(ray, testFunc);
    SurfaceRayIntersection2 result;
    result.distance = queryResult.distance;
    result.isIntersecting = queryResult.item != nullptr;
    if (queryResult.item != nullptr) {
        result.point = ray.pointAt(queryResult.distance);
        result.normal = (*queryResult.item)->closestNormal(result.point);
    }

    for (auto surface : _unboundedSurfaces) {
        SurfaceRayIntersection2 localResult = surface->closestIntersection(ray);
        if (localResult.distance < result.distance) {
            result = localResult;
        }
    }

    return result;
}

BoundingBox2D ImplicitSurfaceSet2::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
}

double ImplicitSurfaceSet2::signedDistanceLocal(
    const Vector2D& otherPoint) const {
    double sdf = kMaxD;
    for (const auto& surface : _surfaces) {
        sdf = std::min(sdf, surface->signedDistance(otherPoint));
    }

    return sdf;
}

void ImplicitSurfaceSet2::invalidateBvh() { _bvhInvalidated = true; }

void ImplicitSurfaceSet2::buildBvh() const {
    if (_bvhInvalidated) {
        Array1<BoundingBox2D> bounds;
        for (size_t i = 0; i < _surfaces.length(); ++i) {
            if (_surfaces[i]->isBounded()) {
                bounds.append(_surfaces[i]->boundingBox());
            }
        }
        _bvh.build(_surfaces, bounds);
        _bvhInvalidated = false;
    }
}

// ImplicitSurfaceSet2::Builder

ImplicitSurfaceSet2::Builder ImplicitSurfaceSet2::builder() {
    return Builder();
}

ImplicitSurfaceSet2::Builder& ImplicitSurfaceSet2::Builder::withSurfaces(
    const Array1<ImplicitSurface2Ptr>& surfaces) {
    _surfaces = surfaces;
    return *this;
}

ImplicitSurfaceSet2::Builder&
ImplicitSurfaceSet2::Builder::withExplicitSurfaces(
    const Array1<Surface2Ptr>& surfaces) {
    _surfaces.clear();
    for (const auto& surface : surfaces) {
        _surfaces.append(std::make_shared<SurfaceToImplicit2>(surface));
    }
    return *this;
}

ImplicitSurfaceSet2 ImplicitSurfaceSet2::Builder::build() const {
    return ImplicitSurfaceSet2(_surfaces, _transform, _isNormalFlipped);
}

ImplicitSurfaceSet2Ptr ImplicitSurfaceSet2::Builder::makeShared() const {
    return std::shared_ptr<ImplicitSurfaceSet2>(
        new ImplicitSurfaceSet2(_surfaces, _transform, _isNormalFlipped),
        [](ImplicitSurfaceSet2* obj) { delete obj; });
}

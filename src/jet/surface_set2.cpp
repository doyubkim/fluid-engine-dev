// Copyright (c) Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/surface_set2.h>

using namespace jet;

SurfaceSet2::SurfaceSet2() {}

SurfaceSet2::SurfaceSet2(const std::vector<Surface2Ptr>& others,
                         const Transform2& transform, bool isNormalFlipped)
    : Surface2(transform, isNormalFlipped), _surfaces(others) {
    for (auto surface : _surfaces) {
        if (!surface->isBounded()) {
            _unboundedSurfaces.push_back(surface);
        }
    }
    invalidateBvh();
}

SurfaceSet2::SurfaceSet2(const SurfaceSet2& other)
    : Surface2(other),
      _surfaces(other._surfaces),
      _unboundedSurfaces(other._unboundedSurfaces) {
    invalidateBvh();
}

void SurfaceSet2::updateQueryEngine() {
    invalidateBvh();
    buildBvh();
}

bool SurfaceSet2::isBounded() const {
    // All surfaces should be bounded.
    for (auto surface : _surfaces) {
        if (!surface->isBounded()) {
            return false;
        }
    }

    // Empty set is not bounded.
    return !_surfaces.empty();
}

bool SurfaceSet2::isValidGeometry() const {
    // All surfaces should be valid.
    for (auto surface : _surfaces) {
        if (!surface->isValidGeometry()) {
            return false;
        }
    }

    // Empty set is not valid.
    return !_surfaces.empty();
}

size_t SurfaceSet2::numberOfSurfaces() const { return _surfaces.size(); }

const Surface2Ptr& SurfaceSet2::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void SurfaceSet2::addSurface(const Surface2Ptr& surface) {
    _surfaces.push_back(surface);
    if (!surface->isBounded()) {
        _unboundedSurfaces.push_back(surface);
    }
    invalidateBvh();
}

Vector2D SurfaceSet2::closestPointLocal(const Vector2D& otherPoint) const {
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

Vector2D SurfaceSet2::closestNormalLocal(const Vector2D& otherPoint) const {
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

double SurfaceSet2::closestDistanceLocal(const Vector2D& otherPoint) const {
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

bool SurfaceSet2::intersectsLocal(const Ray2D& ray) const {
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

SurfaceRayIntersection2 SurfaceSet2::closestIntersectionLocal(
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

BoundingBox2D SurfaceSet2::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
}

bool SurfaceSet2::isInsideLocal(const Vector2D& otherPoint) const {
    for (auto surface : _surfaces) {
        if (surface->isInside(otherPoint)) {
            return true;
        }
    }

    return false;
}

void SurfaceSet2::invalidateBvh() { _bvhInvalidated = true; }

void SurfaceSet2::buildBvh() const {
    if (_bvhInvalidated) {
        std::vector<Surface2Ptr> surfs;
        std::vector<BoundingBox2D> bounds;
        for (size_t i = 0; i < _surfaces.size(); ++i) {
            if (_surfaces[i]->isBounded()) {
                surfs.push_back(_surfaces[i]);
                bounds.push_back(_surfaces[i]->boundingBox());
            }
        }
        _bvh.build(surfs, bounds);
        _bvhInvalidated = false;
    }
}

// SurfaceSet2::Builder

SurfaceSet2::Builder SurfaceSet2::builder() { return Builder(); }

SurfaceSet2::Builder& SurfaceSet2::Builder::withSurfaces(
    const std::vector<Surface2Ptr>& others) {
    _surfaces = others;
    return *this;
}

SurfaceSet2 SurfaceSet2::Builder::build() const {
    return SurfaceSet2(_surfaces, _transform, _isNormalFlipped);
}

SurfaceSet2Ptr SurfaceSet2::Builder::makeShared() const {
    return std::shared_ptr<SurfaceSet2>(
        new SurfaceSet2(_surfaces, _transform, _isNormalFlipped),
        [](SurfaceSet2* obj) { delete obj; });
}

// Copyright (c) Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/surface_set3.h>

using namespace jet;

SurfaceSet3::SurfaceSet3() {}

SurfaceSet3::SurfaceSet3(const std::vector<Surface3Ptr>& others,
                         const Transform3& transform, bool isNormalFlipped)
    : Surface3(transform, isNormalFlipped), _surfaces(others) {
    for (auto surface : _surfaces) {
        if (!surface->isBounded()) {
            _unboundedSurfaces.push_back(surface);
        }
    }
    invalidateBvh();
}

SurfaceSet3::SurfaceSet3(const SurfaceSet3& other)
    : Surface3(other),
      _surfaces(other._surfaces),
      _unboundedSurfaces(other._unboundedSurfaces) {
    invalidateBvh();
}

void SurfaceSet3::updateQueryEngine() {
    invalidateBvh();
    buildBvh();
}

bool SurfaceSet3::isBounded() const {
    // All surfaces should be bounded.
    for (auto surface : _surfaces) {
        if (!surface->isBounded()) {
            return false;
        }
    }

    // Empty set is not bounded.
    return !_surfaces.empty();
}

bool SurfaceSet3::isValidGeometry() const {
    // All surfaces should be valid.
    for (auto surface : _surfaces) {
        if (!surface->isValidGeometry()) {
            return false;
        }
    }

    // Empty set is not valid.
    return !_surfaces.empty();
}

size_t SurfaceSet3::numberOfSurfaces() const { return _surfaces.size(); }

const Surface3Ptr& SurfaceSet3::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void SurfaceSet3::addSurface(const Surface3Ptr& surface) {
    _surfaces.push_back(surface);
    if (!surface->isBounded()) {
        _unboundedSurfaces.push_back(surface);
    }
    invalidateBvh();
}

Vector3D SurfaceSet3::closestPointLocal(const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
        return surface->closestDistance(pt);
    };

    Vector3D result{kMaxD, kMaxD, kMaxD};
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

Vector3D SurfaceSet3::closestNormalLocal(const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
        return surface->closestDistance(pt);
    };

    Vector3D result{1.0, 0.0, 0.0};
    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    if (queryResult.item != nullptr) {
        return (*queryResult.item)->closestNormal(otherPoint);
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

double SurfaceSet3::closestDistanceLocal(const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
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

bool SurfaceSet3::intersectsLocal(const Ray3D& ray) const {
    buildBvh();

    const auto testFunc = [](const Surface3Ptr& surface, const Ray3D& ray) {
        return surface->intersects(ray);
    };

    bool result = _bvh.intersects(ray, testFunc);
    for (auto surface : _unboundedSurfaces) {
        result |= surface->intersects(ray);
    }

    return result;
}

SurfaceRayIntersection3 SurfaceSet3::closestIntersectionLocal(
    const Ray3D& ray) const {
    buildBvh();

    const auto testFunc = [](const Surface3Ptr& surface, const Ray3D& ray) {
        SurfaceRayIntersection3 result = surface->closestIntersection(ray);
        return result.distance;
    };

    const auto queryResult = _bvh.closestIntersection(ray, testFunc);
    SurfaceRayIntersection3 result;
    result.distance = queryResult.distance;
    result.isIntersecting = queryResult.item != nullptr;
    if (queryResult.item != nullptr) {
        result.point = ray.pointAt(queryResult.distance);
        result.normal = (*queryResult.item)->closestNormal(result.point);
    }

    for (auto surface : _unboundedSurfaces) {
        SurfaceRayIntersection3 localResult = surface->closestIntersection(ray);
        if (localResult.distance < result.distance) {
            result = localResult;
        }
    }

    return result;
}

BoundingBox3D SurfaceSet3::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
}

bool SurfaceSet3::isInsideLocal(const Vector3D& otherPoint) const {
    for (auto surface : _surfaces) {
        if (surface->isInside(otherPoint)) {
            return true;
        }
    }

    return false;
}

void SurfaceSet3::invalidateBvh() { _bvhInvalidated = true; }

void SurfaceSet3::buildBvh() const {
    if (_bvhInvalidated) {
        std::vector<Surface3Ptr> surfs;
        std::vector<BoundingBox3D> bounds;
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

// SurfaceSet3::Builder

SurfaceSet3::Builder SurfaceSet3::builder() { return Builder(); }

SurfaceSet3::Builder& SurfaceSet3::Builder::withSurfaces(
    const std::vector<Surface3Ptr>& others) {
    _surfaces = others;
    return *this;
}

SurfaceSet3 SurfaceSet3::Builder::build() const {
    return SurfaceSet3(_surfaces, _transform, _isNormalFlipped);
}

SurfaceSet3Ptr SurfaceSet3::Builder::makeShared() const {
    return std::shared_ptr<SurfaceSet3>(
        new SurfaceSet3(_surfaces, _transform, _isNormalFlipped),
        [](SurfaceSet3* obj) { delete obj; });
}

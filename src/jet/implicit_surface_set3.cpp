// Copyright (c) Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/implicit_surface_set3.h>
#include <jet/surface_to_implicit3.h>

using namespace jet;

ImplicitSurfaceSet3::ImplicitSurfaceSet3() {}

ImplicitSurfaceSet3::ImplicitSurfaceSet3(
    const std::vector<ImplicitSurface3Ptr>& surfaces,
    const Transform3& transform, bool isNormalFlipped)
    : ImplicitSurface3(transform, isNormalFlipped), _surfaces(surfaces) {
    for (auto surface : _surfaces) {
        if (!surface->isBounded()) {
            _unboundedSurfaces.push_back(surface);
        }
    }
    invalidateBvh();
}

ImplicitSurfaceSet3::ImplicitSurfaceSet3(
    const std::vector<Surface3Ptr>& surfaces, const Transform3& transform,
    bool isNormalFlipped)
    : ImplicitSurface3(transform, isNormalFlipped) {
    for (const auto& surface : surfaces) {
        addExplicitSurface(surface);
    }
}

ImplicitSurfaceSet3::ImplicitSurfaceSet3(const ImplicitSurfaceSet3& other)
    : ImplicitSurface3(other),
      _surfaces(other._surfaces),
      _unboundedSurfaces(other._unboundedSurfaces) {}

void ImplicitSurfaceSet3::updateQueryEngine() {
    invalidateBvh();
    buildBvh();
}

bool ImplicitSurfaceSet3::isBounded() const {
    // All surfaces should be bounded.
    for (auto surface : _surfaces) {
        if (!surface->isBounded()) {
            return false;
        }
    }

    // Empty set is not bounded.
    return !_surfaces.empty();
}

bool ImplicitSurfaceSet3::isValidGeometry() const {
    // All surfaces should be valid.
    for (auto surface : _surfaces) {
        if (!surface->isValidGeometry()) {
            return false;
        }
    }

    // Empty set is not valid.
    return !_surfaces.empty();
}

size_t ImplicitSurfaceSet3::numberOfSurfaces() const {
    return _surfaces.size();
}

const ImplicitSurface3Ptr& ImplicitSurfaceSet3::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void ImplicitSurfaceSet3::addExplicitSurface(const Surface3Ptr& surface) {
    addSurface(std::make_shared<SurfaceToImplicit3>(surface));
}

void ImplicitSurfaceSet3::addSurface(const ImplicitSurface3Ptr& surface) {
    _surfaces.push_back(surface);
    if (!surface->isBounded()) {
        _unboundedSurfaces.push_back(surface);
    }
    invalidateBvh();
}

Vector3D ImplicitSurfaceSet3::closestPointLocal(
    const Vector3D& otherPoint) const {
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

double ImplicitSurfaceSet3::closestDistanceLocal(
    const Vector3D& otherPoint) const {
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

Vector3D ImplicitSurfaceSet3::closestNormalLocal(
    const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
        return surface->closestDistance(pt);
    };

    Vector3D result{1.0, 0.0, 0.0};
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

bool ImplicitSurfaceSet3::intersectsLocal(const Ray3D& ray) const {
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

SurfaceRayIntersection3 ImplicitSurfaceSet3::closestIntersectionLocal(
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

BoundingBox3D ImplicitSurfaceSet3::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
}

bool ImplicitSurfaceSet3::isInsideLocal(const Vector3D& otherPoint) const {
    for (auto surface : _surfaces) {
        if (surface->isInside(otherPoint)) {
            return true;
        }
    }

    return false;
}

double ImplicitSurfaceSet3::signedDistanceLocal(
    const Vector3D& otherPoint) const {
    double sdf = kMaxD;
    for (const auto& surface : _surfaces) {
        sdf = std::min(sdf, surface->signedDistance(otherPoint));
    }

    return sdf;
}

void ImplicitSurfaceSet3::invalidateBvh() { _bvhInvalidated = true; }

void ImplicitSurfaceSet3::buildBvh() const {
    if (_bvhInvalidated) {
        std::vector<ImplicitSurface3Ptr> surfs;
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

// ImplicitSurfaceSet3::Builder

ImplicitSurfaceSet3::Builder ImplicitSurfaceSet3::builder() {
    return Builder();
}

ImplicitSurfaceSet3::Builder& ImplicitSurfaceSet3::Builder::withSurfaces(
    const std::vector<ImplicitSurface3Ptr>& surfaces) {
    _surfaces = surfaces;
    return *this;
}

ImplicitSurfaceSet3::Builder&
ImplicitSurfaceSet3::Builder::withExplicitSurfaces(
    const std::vector<Surface3Ptr>& surfaces) {
    _surfaces.clear();
    for (const auto& surface : surfaces) {
        _surfaces.push_back(std::make_shared<SurfaceToImplicit3>(surface));
    }
    return *this;
}

ImplicitSurfaceSet3 ImplicitSurfaceSet3::Builder::build() const {
    return ImplicitSurfaceSet3(_surfaces, _transform, _isNormalFlipped);
}

ImplicitSurfaceSet3Ptr ImplicitSurfaceSet3::Builder::makeShared() const {
    return std::shared_ptr<ImplicitSurfaceSet3>(
        new ImplicitSurfaceSet3(_surfaces, _transform, _isNormalFlipped),
        [](ImplicitSurfaceSet3* obj) { delete obj; });
}

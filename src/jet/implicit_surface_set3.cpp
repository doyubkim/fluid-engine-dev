// Copyright (c) 2017 Doyub Kim
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
    : ImplicitSurface3(transform, isNormalFlipped), _surfaces(surfaces) {}

ImplicitSurfaceSet3::ImplicitSurfaceSet3(
    const std::vector<Surface3Ptr>& surfaces, const Transform3& transform,
    bool isNormalFlipped)
    : ImplicitSurface3(transform, isNormalFlipped) {
    for (const auto& surface : surfaces) {
        addExplicitSurface(surface);
    }
}

ImplicitSurfaceSet3::ImplicitSurfaceSet3(const ImplicitSurfaceSet3& other)
    : ImplicitSurface3(other), _surfaces(other._surfaces) {}

void ImplicitSurfaceSet3::updateQueryEngine() { buildBvh(); }

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
    invalidateBvh();
}

Vector3D ImplicitSurfaceSet3::closestPointLocal(
    const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return (*queryResult.item)->closestPoint(otherPoint);
}

double ImplicitSurfaceSet3::closestDistanceLocal(
    const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return queryResult.distance;
}

Vector3D ImplicitSurfaceSet3::closestNormalLocal(
    const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return (*queryResult.item)->closestNormal(otherPoint);
}

bool ImplicitSurfaceSet3::intersectsLocal(const Ray3D& ray) const {
    buildBvh();

    const auto testFunc = [](const Surface3Ptr& surface, const Ray3D& ray) {
        return surface->intersects(ray);
    };

    return _bvh.intersects(ray, testFunc);
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
    return result;
}

BoundingBox3D ImplicitSurfaceSet3::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
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
        std::vector<BoundingBox3D> bounds(_surfaces.size());
        for (size_t i = 0; i < _surfaces.size(); ++i) {
            bounds[i] = _surfaces[i]->boundingBox();
        }
        _bvh.build(_surfaces, bounds);
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

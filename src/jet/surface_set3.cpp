// Copyright (c) 2017 Doyub Kim
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
    invalidateBvh();
}

SurfaceSet3::SurfaceSet3(const SurfaceSet3& other)
    : Surface3(other), _surfaces(other._surfaces) {
    invalidateBvh();
}

void SurfaceSet3::updateQueryEngine() { buildBvh(); }

size_t SurfaceSet3::numberOfSurfaces() const { return _surfaces.size(); }

const Surface3Ptr& SurfaceSet3::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void SurfaceSet3::addSurface(const Surface3Ptr& surface) {
    _surfaces.push_back(surface);
    invalidateBvh();
}

Vector3D SurfaceSet3::closestPointLocal(const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return (*queryResult.item)->closestPoint(otherPoint);
}

Vector3D SurfaceSet3::closestNormalLocal(const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return (*queryResult.item)->closestNormal(otherPoint);
}

double SurfaceSet3::closestDistanceLocal(const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface3Ptr& surface,
                                 const Vector3D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return queryResult.distance;
}

bool SurfaceSet3::intersectsLocal(const Ray3D& ray) const {
    buildBvh();

    const auto testFunc = [](const Surface3Ptr& surface, const Ray3D& ray) {
        return surface->intersects(ray);
    };

    return _bvh.intersects(ray, testFunc);
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
    return result;
}

BoundingBox3D SurfaceSet3::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
}

void SurfaceSet3::invalidateBvh() { _bvhInvalidated = true; }

void SurfaceSet3::buildBvh() const {
    if (_bvhInvalidated) {
        std::vector<BoundingBox3D> bounds(_surfaces.size());
        for (size_t i = 0; i < _surfaces.size(); ++i) {
            bounds[i] = _surfaces[i]->boundingBox();
        }
        _bvh.build(_surfaces, bounds);
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

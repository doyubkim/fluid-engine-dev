// Copyright (c) 2017 Doyub Kim
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
    invalidateBvh();
}

SurfaceSet2::SurfaceSet2(const SurfaceSet2& other)
    : Surface2(other), _surfaces(other._surfaces) {
    invalidateBvh();
}

void SurfaceSet2::updateQueryEngine() { buildBvh(); }

size_t SurfaceSet2::numberOfSurfaces() const { return _surfaces.size(); }

const Surface2Ptr& SurfaceSet2::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void SurfaceSet2::addSurface(const Surface2Ptr& surface) {
    _surfaces.push_back(surface);
    invalidateBvh();
}

Vector2D SurfaceSet2::closestPointLocal(const Vector2D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface2Ptr& surface,
                                 const Vector2D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return (*queryResult.item)->closestPoint(otherPoint);
}

Vector2D SurfaceSet2::closestNormalLocal(const Vector2D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface2Ptr& surface,
                                 const Vector2D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return (*queryResult.item)->closestNormal(otherPoint);
}

double SurfaceSet2::closestDistanceLocal(const Vector2D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface2Ptr& surface,
                                 const Vector2D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return queryResult.distance;
}

bool SurfaceSet2::intersectsLocal(const Ray2D& ray) const {
    buildBvh();

    const auto testFunc = [](const Surface2Ptr& surface, const Ray2D& ray) {
        return surface->intersects(ray);
    };

    return _bvh.intersects(ray, testFunc);
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
    return result;
}

BoundingBox2D SurfaceSet2::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
}

void SurfaceSet2::invalidateBvh() { _bvhInvalidated = true; }

void SurfaceSet2::buildBvh() const {
    if (_bvhInvalidated) {
        std::vector<BoundingBox2D> bounds(_surfaces.size());
        for (size_t i = 0; i < _surfaces.size(); ++i) {
            bounds[i] = _surfaces[i]->boundingBox();
        }
        _bvh.build(_surfaces, bounds);
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

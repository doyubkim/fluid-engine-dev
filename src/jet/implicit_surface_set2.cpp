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
    const std::vector<ImplicitSurface2Ptr>& surfaces,
    const Transform2& transform, bool isNormalFlipped)
    : ImplicitSurface2(transform, isNormalFlipped), _surfaces(surfaces) {}

ImplicitSurfaceSet2::ImplicitSurfaceSet2(
    const std::vector<Surface2Ptr>& surfaces, const Transform2& transform,
    bool isNormalFlipped)
    : ImplicitSurface2(transform, isNormalFlipped) {
    for (const auto& surface : surfaces) {
        addExplicitSurface(surface);
    }
}

ImplicitSurfaceSet2::ImplicitSurfaceSet2(const ImplicitSurfaceSet2& other)
    : ImplicitSurface2(other), _surfaces(other._surfaces) {}

void ImplicitSurfaceSet2::updateQueryEngine() { buildBvh(); }

size_t ImplicitSurfaceSet2::numberOfSurfaces() const {
    return _surfaces.size();
}

const ImplicitSurface2Ptr& ImplicitSurfaceSet2::surfaceAt(size_t i) const {
    return _surfaces[i];
}

void ImplicitSurfaceSet2::addExplicitSurface(const Surface2Ptr& surface) {
    addSurface(std::make_shared<SurfaceToImplicit2>(surface));
}

void ImplicitSurfaceSet2::addSurface(const ImplicitSurface2Ptr& surface) {
    _surfaces.push_back(surface);
    invalidateBvh();
}

Vector2D ImplicitSurfaceSet2::closestPointLocal(
    const Vector2D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface2Ptr& surface,
                                 const Vector2D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return (*queryResult.item)->closestPoint(otherPoint);
}

double ImplicitSurfaceSet2::closestDistanceLocal(
    const Vector2D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface2Ptr& surface,
                                 const Vector2D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return queryResult.distance;
}

Vector2D ImplicitSurfaceSet2::closestNormalLocal(
    const Vector2D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const Surface2Ptr& surface,
                                 const Vector2D& pt) {
        return surface->closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return (*queryResult.item)->closestNormal(otherPoint);
}

bool ImplicitSurfaceSet2::intersectsLocal(const Ray2D& ray) const {
    buildBvh();

    const auto testFunc = [](const Surface2Ptr& surface, const Ray2D& ray) {
        return surface->intersects(ray);
    };

    return _bvh.intersects(ray, testFunc);
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
        std::vector<BoundingBox2D> bounds(_surfaces.size());
        for (size_t i = 0; i < _surfaces.size(); ++i) {
            bounds[i] = _surfaces[i]->boundingBox();
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
    return ImplicitSurfaceSet2(_surfaces, _transform, _isNormalFlipped);
}

ImplicitSurfaceSet2Ptr ImplicitSurfaceSet2::Builder::makeShared() const {
    return std::shared_ptr<ImplicitSurfaceSet2>(
        new ImplicitSurfaceSet2(_surfaces, _transform, _isNormalFlipped),
        [](ImplicitSurfaceSet2* obj) { delete obj; });
}

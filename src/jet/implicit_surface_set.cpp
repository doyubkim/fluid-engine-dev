// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/implicit_surface_set.h>
#include <jet/surface_to_implicit.h>

namespace jet {

template <size_t N>
ImplicitSurfaceSet<N>::ImplicitSurfaceSet() {}

template <size_t N>
ImplicitSurfaceSet<N>::ImplicitSurfaceSet(
    ConstArrayView1<std::shared_ptr<ImplicitSurface<N>>> surfaces,
    const Transform<N> &transform, bool isNormalFlipped)
    : ImplicitSurface<N>(transform, isNormalFlipped), _surfaces(surfaces) {
    for (auto surface : _surfaces) {
        if (!surface->isBounded()) {
            _unboundedSurfaces.append(surface);
        }
    }
    invalidateBvh();
}

template <size_t N>
ImplicitSurfaceSet<N>::ImplicitSurfaceSet(
    ConstArrayView1<std::shared_ptr<Surface<N>>> surfaces,
    const Transform<N> &transform, bool isNormalFlipped)
    : ImplicitSurface<N>(transform, isNormalFlipped) {
    for (const auto &surface : surfaces) {
        addExplicitSurface(surface);
    }
}

template <size_t N>
ImplicitSurfaceSet<N>::ImplicitSurfaceSet(const ImplicitSurfaceSet &other)
    : ImplicitSurface<N>(other),
      _surfaces(other._surfaces),
      _unboundedSurfaces(other._unboundedSurfaces) {}

template <size_t N>
void ImplicitSurfaceSet<N>::updateQueryEngine() {
    buildBvh();
}

template <size_t N>
bool ImplicitSurfaceSet<N>::isValidGeometry() const {
    // All surfaces should be valid.
    for (auto surface : _surfaces) {
        if (!surface->isValidGeometry()) {
            return false;
        }
    }

    // Empty set is not valid.
    return !_surfaces.isEmpty();
}

template <size_t N>
size_t ImplicitSurfaceSet<N>::numberOfSurfaces() const {
    return _surfaces.length();
}

template <size_t N>
const std::shared_ptr<ImplicitSurface<N>> &ImplicitSurfaceSet<N>::surfaceAt(
    size_t i) const {
    return _surfaces[i];
}

template <size_t N>
void ImplicitSurfaceSet<N>::addExplicitSurface(
    const std::shared_ptr<Surface<N>> &surface) {
    addSurface(std::make_shared<SurfaceToImplicit<N>>(surface));
}

template <size_t N>
void ImplicitSurfaceSet<N>::addSurface(
    const std::shared_ptr<ImplicitSurface<N>> &surface) {
    _surfaces.append(surface);
    if (!surface->isBounded()) {
        _unboundedSurfaces.append(surface);
    }
    invalidateBvh();
}

template <size_t N>
Vector<double, N> ImplicitSurfaceSet<N>::closestPointLocal(
    const Vector<double, N> &otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const std::shared_ptr<Surface<N>> &surface,
                                 const Vector<double, N> &pt) {
        return surface->closestDistance(pt);
    };

    Vector<double, N> result = Vector<double, N>::makeConstant(kMaxD);
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

template <size_t N>
double ImplicitSurfaceSet<N>::closestDistanceLocal(
    const Vector<double, N> &otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const std::shared_ptr<Surface<N>> &surface,
                                 const Vector<double, N> &pt) {
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

template <size_t N>
Vector<double, N> ImplicitSurfaceSet<N>::closestNormalLocal(
    const Vector<double, N> &otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const std::shared_ptr<Surface<N>> &surface,
                                 const Vector<double, N> &pt) {
        return surface->closestDistance(pt);
    };

    Vector<double, N> result = Vector<double, N>::makeUnitX();
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

template <size_t N>
bool ImplicitSurfaceSet<N>::intersectsLocal(const Ray<double, N> &ray) const {
    buildBvh();

    const auto testFunc = [](const std::shared_ptr<Surface<N>> &surface,
                             const Ray<double, N> &ray) {
        return surface->intersects(ray);
    };

    bool result = _bvh.intersects(ray, testFunc);
    for (auto surface : _unboundedSurfaces) {
        result |= surface->intersects(ray);
    }

    return result;
}

template <size_t N>
SurfaceRayIntersection<N> ImplicitSurfaceSet<N>::closestIntersectionLocal(
    const Ray<double, N> &ray) const {
    buildBvh();

    const auto testFunc = [](const std::shared_ptr<Surface<N>> &surface,
                             const Ray<double, N> &ray) {
        SurfaceRayIntersection<N> result = surface->closestIntersection(ray);
        return result.distance;
    };

    const auto queryResult = _bvh.closestIntersection(ray, testFunc);
    SurfaceRayIntersection<N> result;
    result.distance = queryResult.distance;
    result.isIntersecting = queryResult.item != nullptr;
    if (queryResult.item != nullptr) {
        result.point = ray.pointAt(queryResult.distance);
        result.normal = (*queryResult.item)->closestNormal(result.point);
    }

    for (auto surface : _unboundedSurfaces) {
        SurfaceRayIntersection<N> localResult =
            surface->closestIntersection(ray);
        if (localResult.distance < result.distance) {
            result = localResult;
        }
    }

    return result;
}

template <size_t N>
BoundingBox<double, N> ImplicitSurfaceSet<N>::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
}

template <size_t N>
double ImplicitSurfaceSet<N>::signedDistanceLocal(
    const Vector<double, N> &otherPoint) const {
    double sdf = kMaxD;
    for (const auto &surface : _surfaces) {
        sdf = std::min(sdf, surface->signedDistance(otherPoint));
    }

    return sdf;
}

template <size_t N>
void ImplicitSurfaceSet<N>::invalidateBvh() {
    _bvhInvalidated = true;
}

template <size_t N>
void ImplicitSurfaceSet<N>::buildBvh() const {
    if (_bvhInvalidated) {
        Array1<BoundingBox<double, N>> bounds;
        for (size_t i = 0; i < _surfaces.length(); ++i) {
            if (_surfaces[i]->isBounded()) {
                bounds.append(_surfaces[i]->boundingBox());
            }
        }
        _bvh.build(_surfaces, bounds);
        _bvhInvalidated = false;
    }
}

// ImplicitSurfaceSet<N>::Builder

template <size_t N>
typename ImplicitSurfaceSet<N>::Builder ImplicitSurfaceSet<N>::builder() {
    return Builder();
}

template <size_t N>
typename ImplicitSurfaceSet<N>::Builder &
ImplicitSurfaceSet<N>::Builder::withSurfaces(
    const ConstArrayView1<std::shared_ptr<ImplicitSurface<N>>> &surfaces) {
    _surfaces = surfaces;
    return *this;
}

template <size_t N>
typename ImplicitSurfaceSet<N>::Builder &
ImplicitSurfaceSet<N>::Builder::withExplicitSurfaces(
    const ConstArrayView1<std::shared_ptr<Surface<N>>> &surfaces) {
    _surfaces.clear();
    for (const auto &surface : surfaces) {
        _surfaces.append(std::make_shared<SurfaceToImplicit<N>>(surface));
    }
    return *this;
}

template <size_t N>
ImplicitSurfaceSet<N> ImplicitSurfaceSet<N>::Builder::build() const {
    return ImplicitSurfaceSet(_surfaces, _transform, _isNormalFlipped);
}

template <size_t N>
std::shared_ptr<ImplicitSurfaceSet<N>>
ImplicitSurfaceSet<N>::Builder::makeShared() const {
    return std::shared_ptr<ImplicitSurfaceSet>(
        new ImplicitSurfaceSet(_surfaces, _transform, _isNormalFlipped),
        [](ImplicitSurfaceSet *obj) { delete obj; });
}

template class ImplicitSurfaceSet<2>;

template class ImplicitSurfaceSet<3>;

}  // namespace jet

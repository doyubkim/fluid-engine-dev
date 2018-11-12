// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/surface_set.h>

namespace jet {

template <size_t N>
SurfaceSet<N>::SurfaceSet() {}

template <size_t N>
SurfaceSet<N>::SurfaceSet(const Array1<std::shared_ptr<Surface<N>>> &others,
                          const Transform<N> &transform, bool isNormalFlipped)
    : Surface<N>(transform, isNormalFlipped), _surfaces(others) {
    for (auto surface : _surfaces) {
        if (!surface->isBounded()) {
            _unboundedSurfaces.append(surface);
        }
    }
    invalidateBvh();
}

template <size_t N>
SurfaceSet<N>::SurfaceSet(const SurfaceSet &other)
    : Surface<N>(other),
      _surfaces(other._surfaces),
      _unboundedSurfaces(other._unboundedSurfaces) {
    invalidateBvh();
}

template <size_t N>
void SurfaceSet<N>::updateQueryEngine() {
    buildBvh();
}

template <size_t N>
bool SurfaceSet<N>::isValidGeometry() const {
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
size_t SurfaceSet<N>::numberOfSurfaces() const {
    return _surfaces.length();
}

template <size_t N>
const std::shared_ptr<Surface<N>> &SurfaceSet<N>::surfaceAt(size_t i) const {
    return _surfaces[i];
}

template <size_t N>
void SurfaceSet<N>::addSurface(const std::shared_ptr<Surface<N>> &surface) {
    _surfaces.append(surface);
    if (!surface->isBounded()) {
        _unboundedSurfaces.append(surface);
    }
    invalidateBvh();
}

template <size_t N>
Vector<double, N> SurfaceSet<N>::closestPointLocal(
    const Vector<double, N> &otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const std::shared_ptr<Surface<N>> &surface,
                                 const Vector<double, N> &pt) {
        return surface->closestDistance(pt);
    };

    Vector<double, N> result{kMaxD, kMaxD};
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
Vector<double, N> SurfaceSet<N>::closestNormalLocal(
    const Vector<double, N> &otherPoint) const {
    buildBvh();

    const auto distanceFunc = [](const std::shared_ptr<Surface<N>> &surface,
                                 const Vector<double, N> &pt) {
        return surface->closestDistance(pt);
    };

    Vector<double, N> result{1.0, 0.0};
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
double SurfaceSet<N>::closestDistanceLocal(
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
bool SurfaceSet<N>::intersectsLocal(const Ray<double, N> &ray) const {
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
SurfaceRayIntersection<N> SurfaceSet<N>::closestIntersectionLocal(
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
BoundingBox<double, N> SurfaceSet<N>::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
}

template <size_t N>
void SurfaceSet<N>::invalidateBvh() {
    _bvhInvalidated = true;
}

template <size_t N>
void SurfaceSet<N>::buildBvh() const {
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

// SurfaceSet<N>::Builder

template <size_t N>
typename SurfaceSet<N>::Builder SurfaceSet<N>::builder() {
    return Builder();
}

template <size_t N>
typename SurfaceSet<N>::Builder &SurfaceSet<N>::Builder::withSurfaces(
    const Array1<std::shared_ptr<Surface<N>>> &others) {
    _surfaces = others;
    return *this;
}

template <size_t N>
SurfaceSet<N> SurfaceSet<N>::Builder::build() const {
    return SurfaceSet(_surfaces, _transform, _isNormalFlipped);
}

template <size_t N>
std::shared_ptr<SurfaceSet<N>> SurfaceSet<N>::Builder::makeShared() const {
    return std::shared_ptr<SurfaceSet>(
        new SurfaceSet(_surfaces, _transform, _isNormalFlipped),
        [](SurfaceSet *obj) { delete obj; });
}

template class SurfaceSet<2>;

template class SurfaceSet<3>;

}  // namespace jet

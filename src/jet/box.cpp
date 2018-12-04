// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/box.h>
#include <jet/plane.h>

namespace jet {

template <size_t N>
Box<N>::Box(const Transform<N> &transform, bool isNormalFlipped)
    : Surface<N>(transform, isNormalFlipped) {}

template <size_t N>
Box<N>::Box(const Vector<double, N> &lowerCorner,
            const Vector<double, N> &upperCorner, const Transform<N> &transform,
            bool isNormalFlipped)
    : Box(BoundingBox<double, N>(lowerCorner, upperCorner), transform,
          isNormalFlipped) {}

template <size_t N>
Box<N>::Box(const BoundingBox<double, N> &boundingBox,
            const Transform<N> &transform, bool isNormalFlipped)
    : Surface<N>(transform, isNormalFlipped), bound(boundingBox) {}

template <size_t N>
Box<N>::Box(const Box &other) : Surface<N>(other), bound(other.bound) {}

template <size_t N>
Vector<double, N> Box<N>::closestPointLocal(
    const Vector<double, N> &otherPoint) const {
    if (bound.contains(otherPoint)) {
        Plane<N> planes[2 * N];
        for (size_t i = 0; i < N; ++i) {
            Vector<double, N> normal;
            normal[i] = 1.0;
            planes[i] = Plane<N>(normal, bound.upperCorner);
            planes[i + N] = Plane<N>(-normal, bound.lowerCorner);
        }

        Vector<double, N> result = planes[0].closestPoint(otherPoint);
        double distanceSquared = result.distanceSquaredTo(otherPoint);

        for (size_t i = 1; i < 2 * N; ++i) {
            Vector<double, N> localResult = planes[i].closestPoint(otherPoint);
            double localDistanceSquared =
                localResult.distanceSquaredTo(otherPoint);

            if (localDistanceSquared < distanceSquared) {
                result = localResult;
                distanceSquared = localDistanceSquared;
            }
        }

        return result;
    } else {
        return clamp(otherPoint, bound.lowerCorner, bound.upperCorner);
    }
}

template <size_t N>
Vector<double, N> Box<N>::closestNormalLocal(
    const Vector<double, N> &otherPoint) const {
    Plane<N> planes[2 * N];
    for (size_t i = 0; i < N; ++i) {
        Vector<double, N> normal;
        normal[i] = 1.0;
        planes[i] = Plane<N>(normal, bound.upperCorner);
        planes[i + N] = Plane<N>(-normal, bound.lowerCorner);
    }

    if (bound.contains(otherPoint)) {
        Vector<double, N> closestNormal = planes[0].normal;
        Vector<double, N> closestPoint = planes[0].closestPoint(otherPoint);
        double minDistanceSquared = (closestPoint - otherPoint).lengthSquared();

        for (size_t i = 1; i < 2 * N; ++i) {
            Vector<double, N> localClosestPoint =
                planes[i].closestPoint(otherPoint);
            double localDistanceSquared =
                (localClosestPoint - otherPoint).lengthSquared();

            if (localDistanceSquared < minDistanceSquared) {
                closestNormal = planes[i].normal;
                minDistanceSquared = localDistanceSquared;
            }
        }

        return closestNormal;
    } else {
        Vector<double, N> closestPoint =
            clamp(otherPoint, bound.lowerCorner, bound.upperCorner);
        Vector<double, N> closestPointToInputPoint = otherPoint - closestPoint;
        Vector<double, N> closestNormal = planes[0].normal;
        double maxCosineAngle = closestNormal.dot(closestPointToInputPoint);

        for (size_t i = 1; i < 2 * N; ++i) {
            double cosineAngle = planes[i].normal.dot(closestPointToInputPoint);

            if (cosineAngle > maxCosineAngle) {
                closestNormal = planes[i].normal;
                maxCosineAngle = cosineAngle;
            }
        }

        return closestNormal;
    }
}

template <size_t N>
bool Box<N>::intersectsLocal(const Ray<double, N> &ray) const {
    return bound.intersects(ray);
}

template <size_t N>
SurfaceRayIntersection<N> Box<N>::closestIntersectionLocal(
    const Ray<double, N> &ray) const {
    SurfaceRayIntersection<N> intersection;
    BoundingBoxRayIntersectionD bbRayIntersection =
        bound.closestIntersection(ray);
    intersection.isIntersecting = bbRayIntersection.isIntersecting;
    if (intersection.isIntersecting) {
        intersection.distance = bbRayIntersection.tNear;
        intersection.point = ray.pointAt(bbRayIntersection.tNear);
        intersection.normal = Box<N>::closestNormal(intersection.point);
    }
    return intersection;
}

template <size_t N>
BoundingBox<double, N> Box<N>::boundingBoxLocal() const {
    return bound;
}

template <size_t N>
typename Box<N>::Builder Box<N>::builder() {
    return Builder();
}

template <size_t N>
typename Box<N>::Builder &Box<N>::Builder::withLowerCorner(
    const Vector<double, N> &pt) {
    _lowerCorner = pt;
    return *this;
}

template <size_t N>
typename Box<N>::Builder &Box<N>::Builder::withUpperCorner(
    const Vector<double, N> &pt) {
    _upperCorner = pt;
    return *this;
}

template <size_t N>
typename Box<N>::Builder &Box<N>::Builder::withBoundingBox(
    const BoundingBox<double, N> &bbox) {
    _lowerCorner = bbox.lowerCorner;
    _upperCorner = bbox.upperCorner;
    return *this;
}

template <size_t N>
Box<N> Box<N>::Builder::build() const {
    return Box<N>(_lowerCorner, _upperCorner, _transform, _isNormalFlipped);
}

template <size_t N>
std::shared_ptr<Box<N>> Box<N>::Builder::makeShared() const {
    return std::shared_ptr<Box<N>>(
        new Box<N>(_lowerCorner, _upperCorner, _transform, _isNormalFlipped),
        [](Box<N> *obj) { delete obj; });
}

template class Box<2>;

template class Box<3>;

}  // namespace jet

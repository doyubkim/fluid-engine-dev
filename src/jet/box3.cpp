// Copyright (c) 2020 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/box3.h>
#include <jet/plane3.h>

using namespace jet;

Box3::Box3(const Transform3& transform, bool isNormalFlipped)
    : Surface3(transform, isNormalFlipped) {}

Box3::Box3(const Vector3D& lowerCorner, const Vector3D& upperCorner,
           const Transform3& transform, bool isNormalFlipped)
    : Box3(BoundingBox3D(lowerCorner, upperCorner), transform,
           isNormalFlipped) {}

Box3::Box3(const BoundingBox3D& boundingBox, const Transform3& transform,
           bool isNormalFlipped)
    : Surface3(transform, isNormalFlipped), bound(boundingBox) {}

Box3::Box3(const Box3& other) : Surface3(other), bound(other.bound) {}

Vector3D Box3::closestPointLocal(const Vector3D& otherPoint) const {
    if (bound.contains(otherPoint)) {
        Plane3 planes[6] = {Plane3(Vector3D(1, 0, 0), bound.upperCorner),
                            Plane3(Vector3D(0, 1, 0), bound.upperCorner),
                            Plane3(Vector3D(0, 0, 1), bound.upperCorner),
                            Plane3(Vector3D(-1, 0, 0), bound.lowerCorner),
                            Plane3(Vector3D(0, -1, 0), bound.lowerCorner),
                            Plane3(Vector3D(0, 0, -1), bound.lowerCorner)};

        Vector3D result = planes[0].closestPoint(otherPoint);
        double distanceSquared = result.distanceSquaredTo(otherPoint);

        for (int i = 1; i < 6; ++i) {
            Vector3D localResult = planes[i].closestPoint(otherPoint);
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

Vector3D Box3::closestNormalLocal(const Vector3D& otherPoint) const {
    Plane3 planes[6] = {Plane3(Vector3D(1, 0, 0), bound.upperCorner),
                        Plane3(Vector3D(0, 1, 0), bound.upperCorner),
                        Plane3(Vector3D(0, 0, 1), bound.upperCorner),
                        Plane3(Vector3D(-1, 0, 0), bound.lowerCorner),
                        Plane3(Vector3D(0, -1, 0), bound.lowerCorner),
                        Plane3(Vector3D(0, 0, -1), bound.lowerCorner)};

    if (bound.contains(otherPoint)) {
        Vector3D closestNormal = planes[0].normal;
        Vector3D closestPoint = planes[0].closestPoint(otherPoint);
        double minDistanceSquared = (closestPoint - otherPoint).lengthSquared();

        for (int i = 1; i < 6; ++i) {
            Vector3D localClosestPoint = planes[i].closestPoint(otherPoint);
            double localDistanceSquared =
                (localClosestPoint - otherPoint).lengthSquared();

            if (localDistanceSquared < minDistanceSquared) {
                closestNormal = planes[i].normal;
                minDistanceSquared = localDistanceSquared;
            }
        }

        return closestNormal;
    } else {
        Vector3D closestPoint =
            clamp(otherPoint, bound.lowerCorner, bound.upperCorner);
        Vector3D closestPointToInputPoint = otherPoint - closestPoint;
        Vector3D closestNormal = planes[0].normal;
        double maxCosineAngle = closestNormal.dot(closestPointToInputPoint);

        for (int i = 1; i < 6; ++i) {
            double cosineAngle = planes[i].normal.dot(closestPointToInputPoint);

            if (cosineAngle > maxCosineAngle) {
                closestNormal = planes[i].normal;
                maxCosineAngle = cosineAngle;
            }
        }

        return closestNormal;
    }
}

bool Box3::intersectsLocal(const Ray3D& ray) const {
    return bound.intersects(ray);
}

SurfaceRayIntersection3 Box3::closestIntersectionLocal(const Ray3D& ray) const {
    SurfaceRayIntersection3 intersection;
    BoundingBoxRayIntersection3D bbRayIntersection =
        bound.closestIntersection(ray);
    intersection.isIntersecting = bbRayIntersection.isIntersecting;
    if (intersection.isIntersecting) {
        intersection.distance = bbRayIntersection.tNear;
        intersection.point = ray.pointAt(bbRayIntersection.tNear);
        intersection.normal = closestNormalLocal(intersection.point);
    }

    return intersection;
}

BoundingBox3D Box3::boundingBoxLocal() const { return bound; }

Box3::Builder Box3::builder() { return Builder(); }

Box3::Builder& Box3::Builder::withLowerCorner(const Vector3D& pt) {
    _lowerCorner = pt;
    return *this;
}

Box3::Builder& Box3::Builder::withUpperCorner(const Vector3D& pt) {
    _upperCorner = pt;
    return *this;
}

Box3::Builder& Box3::Builder::withBoundingBox(const BoundingBox3D& bbox) {
    _lowerCorner = bbox.lowerCorner;
    _upperCorner = bbox.upperCorner;
    return *this;
}

Box3 Box3::Builder::build() const {
    return Box3(_lowerCorner, _upperCorner, _transform, _isNormalFlipped);
}

Box3Ptr Box3::Builder::makeShared() const {
    return std::shared_ptr<Box3>(
        new Box3(_lowerCorner, _upperCorner, _transform, _isNormalFlipped),
        [](Box3* obj) { delete obj; });
}

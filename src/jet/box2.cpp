// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/box2.h>
#include <jet/plane2.h>

using namespace jet;

Box2::Box2(bool isNormalFlipped_) : Surface2(isNormalFlipped_) {
}

Box2::Box2(
    const Vector2D& lowerCorner,
    const Vector2D& upperCorner,
    bool isNormalFlipped_) :
    Box2(BoundingBox2D(lowerCorner, upperCorner), isNormalFlipped_) {
}

Box2::Box2(const BoundingBox2D& boundingBox, bool isNormalFlipped_)
: Surface2(isNormalFlipped_)
, bound(boundingBox) {
}

Box2::Box2(const Box2& other) : Surface2(other), bound(other.bound) {
}

Vector2D Box2::closestPoint(const Vector2D& otherPoint) const {
    if (bound.contains(otherPoint)) {
        Plane2 planes[4] = {
            Plane2(Vector2D(1, 0), bound.upperCorner),
            Plane2(Vector2D(0, 1), bound.upperCorner),
            Plane2(Vector2D(-1, 0), bound.lowerCorner),
            Plane2(Vector2D(0, -1), bound.lowerCorner)
        };

        Vector2D result = planes[0].closestPoint(otherPoint);
        double distanceSquared = result.distanceSquaredTo(otherPoint);

        for (int i = 1; i < 4; ++i) {
            Vector2D localResult = planes[i].closestPoint(otherPoint);
            double localDistanceSquared
                = localResult.distanceSquaredTo(otherPoint);

            if (localDistanceSquared < distanceSquared) {
                result = localResult;
                distanceSquared = localDistanceSquared;
            }
        }

        return result;
    } else {
        return clamp(
            otherPoint,
            bound.lowerCorner,
            bound.upperCorner);
    }
}

Vector2D Box2::actualClosestNormal(const Vector2D& otherPoint) const {
    Plane2 planes[4] = {
        Plane2(Vector2D(1, 0), bound.upperCorner),
        Plane2(Vector2D(0, 1), bound.upperCorner),
        Plane2(Vector2D(-1, 0), bound.lowerCorner),
        Plane2(Vector2D(0, -1), bound.lowerCorner)
    };

    if (bound.contains(otherPoint)) {
        Vector2D closestNormal = planes[0].normal;
        Vector2D closestPoint = planes[0].closestPoint(otherPoint);
        double minDistanceSquared = (closestPoint - otherPoint).lengthSquared();

        for (int i = 1; i < 4; ++i) {
            Vector2D localClosestPoint = planes[i].closestPoint(otherPoint);
            double localDistanceSquared
                = (localClosestPoint - otherPoint).lengthSquared();

            if (localDistanceSquared < minDistanceSquared) {
                closestNormal = planes[i].normal;
                minDistanceSquared = localDistanceSquared;
            }
        }

        return closestNormal;
    } else {
        Vector2D closestPoint = clamp(
            otherPoint,
            bound.lowerCorner,
            bound.upperCorner);
        Vector2D closestPointToInputPoint = otherPoint - closestPoint;
        Vector2D closestNormal = planes[0].normal;
        double maxCosineAngle = closestNormal.dot(closestPointToInputPoint);

        for (int i = 1; i < 4; ++i) {
            double cosineAngle
                = planes[i].normal.dot(closestPointToInputPoint);

            if (cosineAngle > maxCosineAngle) {
                closestNormal = planes[i].normal;
                maxCosineAngle = cosineAngle;
            }
        }

        return closestNormal;
    }
}

double Box2::closestDistance(const Vector2D& otherPoint) const {
    return Box2::closestPoint(otherPoint).distanceTo(otherPoint);
}

bool Box2::intersects(const Ray2D& ray) const {
    return bound.intersects(ray);
}

SurfaceRayIntersection2 Box2::actualClosestIntersection(
    const Ray2D& ray) const {
    SurfaceRayIntersection2 intersection;
    BoundingBoxRayIntersection2D bbRayIntersection;
    bound.getClosestIntersection(ray, &bbRayIntersection);
    intersection.isIntersecting = bbRayIntersection.isIntersecting;
    if (intersection.isIntersecting) {
        intersection.t = bbRayIntersection.tNear;
        intersection.point = ray.pointAt(bbRayIntersection.tNear);
        intersection.normal = Box2::closestNormal(intersection.point);
    }
    return intersection;
}

BoundingBox2D Box2::boundingBox() const {
    return bound;
}

Box2::Builder Box2::builder() {
    return Builder();
}

Box2::Builder& Box2::Builder::withIsNormalFlipped(bool isNormalFlipped) {
    _isNormalFlipped = isNormalFlipped;
    return *this;
}

Box2::Builder& Box2::Builder::withLowerCorner(const Vector2D& pt) {
    _lowerCorner = pt;
    return *this;
}

Box2::Builder& Box2::Builder::withUpperCorner(const Vector2D& pt) {
    _upperCorner = pt;
    return *this;
}

Box2::Builder& Box2::Builder::withBoundingBox(const BoundingBox2D& bbox) {
    _lowerCorner = bbox.lowerCorner;
    _upperCorner = bbox.upperCorner;
    return *this;
}

Box2 Box2::Builder::build() const {
    return Box2(_lowerCorner, _upperCorner, _isNormalFlipped);
}

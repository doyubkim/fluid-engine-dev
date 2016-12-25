// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/plane2.h>

#include <limits>

using namespace jet;

Plane2::Plane2(const Transform2& transform_, bool isNormalFlipped_)
: Surface2(transform_, isNormalFlipped_) {
}

Plane2::Plane2(
    const Vector2D& normal_,
    const Vector2D& point_,
    const Transform2& transform_,
    bool isNormalFlipped_)
: Surface2(transform_, isNormalFlipped_)
, normal(normal_)
, point(point_) {
}

Plane2::Plane2(const Plane2& other) :
    Surface2(other),
    normal(other.normal),
    point(other.point) {
}

Vector2D Plane2::closestPointLocal(const Vector2D& otherPoint) const {
    Vector2D r = otherPoint - point;
    return r - normal.dot(r) * normal + point;
}

Vector2D Plane2::closestNormalLocal(const Vector2D& otherPoint) const {
    UNUSED_VARIABLE(otherPoint);
    return normal;
}

bool Plane2::intersectsLocal(const Ray2D& ray) const {
    return std::fabs(ray.direction.dot(normal)) > 0;
}

SurfaceRayIntersection2 Plane2::closestIntersectionLocal(
    const Ray2D& ray) const {
    SurfaceRayIntersection2 intersection;
    double dDotN = ray.direction.dot(normal);

    // Check if not parallel
    if (std::fabs(dDotN) > 0) {
        double t = normal.dot(point - ray.origin) / dDotN;
        if (t >= 0.0) {
            intersection.isIntersecting = true;
            intersection.t = t;
            intersection.point = ray.pointAt(t);
            intersection.normal = normal;
        }
    }

    return intersection;
}

BoundingBox2D Plane2::boundingBoxLocal() const {
    if (std::fabs(normal.dot(Vector2D(1, 0)) - 1.0) < kEpsilonD) {
        return BoundingBox2D(
            point - Vector2D(0, kMaxD),
            point + Vector2D(0, kMaxD));
    } else if (std::fabs(normal.dot(Vector2D(0, 1)) - 1.0) < kEpsilonD) {
        return BoundingBox2D(
            point - Vector2D(kMaxD, 0),
            point + Vector2D(kMaxD, 0));
    } else {
        return BoundingBox2D(
            Vector2D(kMaxD, kMaxD),
            Vector2D(kMaxD, kMaxD));
    }
}

Plane2::Builder Plane2::builder() {
    return Builder();
}

Plane2::Builder& Plane2::Builder::withNormal(const Vector2D& normal) {
    _normal = normal;
    return *this;
}

Plane2::Builder& Plane2::Builder::withPoint(const Vector2D& point) {
    _point = point;
    return *this;
}

Plane2 Plane2::Builder::build() const {
    return Plane2(_normal, _point, _transform, _isNormalFlipped);
}

Plane2Ptr Plane2::Builder::makeShared() const {
    return std::shared_ptr<Plane2>(
        new Plane2(
            _normal,
            _point,
            _transform,
            _isNormalFlipped),
        [] (Plane2* obj) {
            delete obj;
        });
}

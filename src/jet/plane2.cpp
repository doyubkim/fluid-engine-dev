// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/plane2.h>

#include <limits>

using namespace jet;

Plane2::Plane2() {
}

Plane2::Plane2(const Vector2D& normal_, const Vector2D& point_) :
    normal(normal_),
    point(point_) {
}

Plane2::Plane2(const Plane2& other) :
    normal(other.normal),
    point(other.point) {
}

Vector2D Plane2::closestPoint(const Vector2D& otherPoint) const {
    Vector2D r = otherPoint - point;
    return r - normal.dot(r) * normal + point;
}

double Plane2::closestDistance(const Vector2D& otherPoint) const {
    return (otherPoint - closestPoint(otherPoint)).length();
}

Vector2D Plane2::actualClosestNormal(const Vector2D& otherPoint) const {
    UNUSED_VARIABLE(otherPoint);
    return normal;
}

bool Plane2::intersects(const Ray2D& ray) const {
    return std::fabs(ray.direction.dot(normal)) > 0;
}

SurfaceRayIntersection2 Plane2::closestIntersection(const Ray2D& ray) const {
    SurfaceRayIntersection2 intersection;
    double dDotN = ray.direction.dot(normal);

    if (std::fabs(dDotN) > 0) {
        double t = normal.dot(point - ray.origin) / dDotN;

        intersection.isIntersecting = true;
        intersection.t = t;
        intersection.point = ray.pointAt(t);
        intersection.normal = normal;
    } else {
        intersection.isIntersecting = false;
    }

    return intersection;
}

BoundingBox2D Plane2::boundingBox() const {
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

// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/sphere2.h>

#include <limits>

using namespace jet;

Sphere2::Sphere2(const Vector2D& center, double radius) :
    _center(center),
    _radius(radius) {
}

Vector2D Sphere2::closestPoint(const Vector2D& otherPoint) const {
    return _radius * closestNormal(otherPoint) + _center;
}

Vector2D Sphere2::actualClosestNormal(const Vector2D& otherPoint) const {
    if (_center.isSimilar(otherPoint)) {
        return Vector2D(1, 0);
    } else {
        return (otherPoint - _center).normalized();
    }
}

void Sphere2::getClosestIntersection(
    const Ray2D& ray,
    SurfaceRayIntersection2* intersection) const {
    Vector2D r = ray.origin - _center;
    double b = ray.direction.dot(r);
    double c = r.lengthSquared() - square(_radius);
    double d = b * b - c;

    if (d > 0.) {
        d = std::sqrt(d);
        double tMin = -b - d;
        double tMax = -b + d;

        if (tMin < 0.0) {
            tMin = tMax;
            tMax = std::numeric_limits<double>::max();
        }

        if (tMin < 0.0) {
            intersection->isIntersecting = false;
        } else {
            intersection->isIntersecting = true;
            intersection->t = tMin;
            intersection->point = ray.origin + tMin * ray.direction;
            intersection->normal = (intersection->point - _center).normalized();
        }
    } else {
        intersection->isIntersecting = false;
    }
}

BoundingBox2D Sphere2::boundingBox() const {
    Vector2D r(_radius, _radius);
    return BoundingBox2D(_center - r, _center + r);
}

const Vector2D& Sphere2::center() const {
    return _center;
}

void Sphere2::setCenter(const Vector2D& newCenter) {
    _center = newCenter;
}

double Sphere2::radius() const {
    return _radius;
}

void Sphere2::setRadius(double newRadius) {
    _radius = newRadius;
}

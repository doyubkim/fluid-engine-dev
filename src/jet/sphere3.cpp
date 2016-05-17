// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/sphere3.h>

#include <limits>

using namespace jet;

Sphere3::Sphere3(const Vector3D& center, double radius) :
    _center(center),
    _radius(radius) {
}

Vector3D Sphere3::closestPoint(const Vector3D& otherPoint) const {
    return _radius * closestNormal(otherPoint) + _center;
}

Vector3D Sphere3::actualClosestNormal(const Vector3D& otherPoint) const {
    if (_center.isSimilar(otherPoint)) {
        return Vector3D(1, 0, 0);
    } else {
        return (otherPoint - _center).normalized();
    }
}

void Sphere3::getClosestIntersection(
    const Ray3D& ray,
    SurfaceRayIntersection3* intersection) const {
    Vector3D r = ray.origin - _center;
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

BoundingBox3D Sphere3::boundingBox() const {
    Vector3D r(_radius, _radius, _radius);
    return BoundingBox3D(_center - r, _center + r);
}

const Vector3D& Sphere3::center() const {
    return _center;
}

void Sphere3::setCenter(const Vector3D& newCenter) {
    _center = newCenter;
}

double Sphere3::radius() const {
    return _radius;
}

void Sphere3::setRadius(double newRadius) {
    _radius = newRadius;
}

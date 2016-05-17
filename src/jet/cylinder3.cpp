// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/box2.h>
#include <jet/cylinder3.h>
#include <jet/plane3.h>

using namespace jet;

Cylinder3::Cylinder3() {
}

Cylinder3::Cylinder3(const Vector3D& center_, double radius_, double height_)
    : _center(center_), _radius(radius_), _height(height_) {
}

Cylinder3::Cylinder3(const Cylinder3& other)
    : _center(other._center),
    _radius(other._radius),
    _height(other._height) {
}

Vector3D Cylinder3::closestPoint(const Vector3D& otherPoint) const {
    Vector3D r = otherPoint - _center;
    Vector2D rr(std::sqrt(r.x * r.x + r.z * r.z), r.y);
    Box2 box(
        Vector2D(-_radius, -0.5 * _height),
        Vector2D(_radius, 0.5 * _height));

    Vector2D cp = box.closestPoint(rr);
    if (rr.lengthSquared() > 0.0) {
        return cp.length() / rr.length() * r + _center;
    }

    return Vector3D();
}

double Cylinder3::closestDistance(const Vector3D& otherPoint) const {
    Vector3D r = otherPoint - _center;
    Vector2D rr(std::sqrt(r.x * r.x + r.z * r.z), r.y);
    Box2 box(
        Vector2D(-_radius, -0.5 * _height),
        Vector2D(_radius, 0.5 * _height));

    return box.closestDistance(rr);
}

Vector3D Cylinder3::actualClosestNormal(const Vector3D& otherPoint) const {
    Vector3D r = otherPoint - _center;
    Vector2D rr(std::sqrt(r.x * r.x + r.z * r.z), r.y);
    Box2 box(
        Vector2D(-_radius, -0.5 * _height),
        Vector2D(_radius, 0.5 * _height));

    Vector2D cn = box.actualClosestNormal(rr);
    if (cn.y > 0) {
        return Vector3D(0, 1, 0);
    } else if (cn.y < 0) {
        return Vector3D(0, -1, 0);
    } else {
        return Vector3D(r.x, 0, r.z).normalized();
    }
}

bool Cylinder3::intersects(const Ray3D& ray) const {
    // Calculate intersection with infinite cylinder
    // (dx^2 + dz^2)t^2 + 2(ox.dx + oz.dz)t + ox^2 + oz^2 - r^2 = 0
    Vector3D d = ray.direction;
    d.y = 0.0;
    Vector3D o = ray.origin;
    o.y = 0.0;
    double A = d.lengthSquared();
    double B = d.dot(o);
    double C = o.lengthSquared() - square(_radius);

    if (A < kEpsilonD || B*B - A*C < 0.0) {
        return false;
    }

    double t1 = (-B + std::sqrt(B*B - A*C)) / A;
    double t2 = (-B - std::sqrt(B*B - A*C)) / A;
    double tCylinder = t2;

    if (t2 < 0.0) {
        tCylinder = t1;
    }

    Vector3D pointOnCylinder = ray.pointAt(tCylinder);

    if (pointOnCylinder.y >= _center.y - 0.5 * _height
        || pointOnCylinder.y <= _center.y + 0.5 * _height) {
        return true;
    }

    BoundingBox3D bbox = boundingBox();
    Plane3 upperPlane(Vector3D(0,  1, 0), bbox.upperCorner);
    Plane3 lowerPlane(Vector3D(0, -1, 0), bbox.lowerCorner);

    SurfaceRayIntersection3 upperIntersection;
    upperPlane.getClosestIntersection(ray, &upperIntersection);

    SurfaceRayIntersection3 lowerIntersection;
    lowerPlane.getClosestIntersection(ray, &lowerIntersection);

    if (upperIntersection.isIntersecting) {
        Vector3D r = upperIntersection.point - _center;
        r.y = 0.0;
        if (r.lengthSquared() <= square(_radius)) {
            return true;
        }
    }

    if (lowerIntersection.isIntersecting) {
        Vector3D r = lowerIntersection.point - _center;
        r.y = 0.0;
        if (r.lengthSquared() <= square(_radius)) {
            return true;
        }
    }

    return false;
}

void Cylinder3::getClosestIntersection(
    const Ray3D& ray,
    SurfaceRayIntersection3* intersection) const {
    // Calculate intersection with infinite cylinder
    // (dx^2 + dz^2)t^2 + 2(ox.dx + oz.dz)t + ox^2 + oz^2 - r^2 = 0
    Vector3D d = ray.direction;
    d.y = 0.0;
    Vector3D o = ray.origin;
    o.y = 0.0;
    double A = d.lengthSquared();
    double B = d.dot(o);
    double C = o.lengthSquared() - square(_radius);

    intersection->isIntersecting = false;

    if (A < kEpsilonD || B*B - A*C < 0.0) {
        return;
    }

    double t1 = (-B + std::sqrt(B*B - A*C)) / A;
    double t2 = (-B - std::sqrt(B*B - A*C)) / A;
    double tCylinder = t2;

    if (t2 < 0.0) {
        tCylinder = t1;
    }

    Vector3D pointOnCylinder = ray.pointAt(tCylinder);

    if (pointOnCylinder.y >= _center.y - 0.5 * _height
        || pointOnCylinder.y <= _center.y + 0.5 * _height) {
        intersection->isIntersecting = true;
        intersection->t = tCylinder;
        intersection->point = pointOnCylinder;
        intersection->normal = pointOnCylinder - _center;
        intersection->normal.y = 0.0;
        intersection->normal.normalize();
    }

    BoundingBox3D bbox = boundingBox();
    Plane3 upperPlane(Vector3D(0,  1, 0), bbox.upperCorner);
    Plane3 lowerPlane(Vector3D(0, -1, 0), bbox.lowerCorner);

    SurfaceRayIntersection3 upperIntersection;
    upperPlane.getClosestIntersection(ray, &upperIntersection);

    SurfaceRayIntersection3 lowerIntersection;
    lowerPlane.getClosestIntersection(ray, &lowerIntersection);

    if (upperIntersection.isIntersecting) {
        Vector3D r = upperIntersection.point - _center;
        r.y = 0.0;
        if (r.lengthSquared() > square(_radius)) {
            upperIntersection.isIntersecting = false;
        } else if (upperIntersection.t < intersection->t) {
            *intersection = upperIntersection;
        }
    }

    if (lowerIntersection.isIntersecting) {
        Vector3D r = lowerIntersection.point - _center;
        r.y = 0.0;
        if (r.lengthSquared() > square(_radius)) {
            lowerIntersection.isIntersecting = false;
        } else if (lowerIntersection.t < intersection->t) {
            *intersection = lowerIntersection;
        }
    }
}

BoundingBox3D Cylinder3::boundingBox() const {
    return BoundingBox3D(
        _center - Vector3D(_radius, 0.5 * _height, _radius),
        _center + Vector3D(_radius, 0.5 * _height, _radius));
}

const Vector3D& Cylinder3::center() const {
    return _center;
}

void Cylinder3::setCenter(const Vector3D& newCenter) {
    _center = newCenter;
}

double Cylinder3::radius() const {
    return _radius;
}

void Cylinder3::setRadius(double newRadius) {
    _radius = newRadius;
}

double Cylinder3::height() const {
    return _height;
}

void Cylinder3::setHeight(double newHeight) {
    _height = newHeight;
}

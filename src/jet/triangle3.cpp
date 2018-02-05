// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/triangle3.h>

using namespace jet;

inline Vector3D closestPointOnLine(const Vector3D& v0, const Vector3D& v1,
                                   const Vector3D& pt) {
    const double lenSquared = (v1 - v0).lengthSquared();
    if (lenSquared < std::numeric_limits<double>::epsilon()) {
        return v0;
    }

    const double t = (pt - v0).dot(v1 - v0) / lenSquared;
    if (t < 0.0) {
        return v0;
    } else if (t > 1.0) {
        return v1;
    }

    return v0 + t * (v1 - v0);
}

inline Vector3D closestNormalOnLine(const Vector3D& v0, const Vector3D& v1,
                                    const Vector3D& n0, const Vector3D& n1,
                                    const Vector3D& pt) {
    const double lenSquared = (v1 - v0).lengthSquared();
    if (lenSquared < std::numeric_limits<double>::epsilon()) {
        return n0;
    }

    const double t = (pt - v0).dot(v1 - v0) / lenSquared;
    if (t < 0.0) {
        return n0;
    } else if (t > 1.0) {
        return n1;
    }

    return (n0 + t * (n1 - n0)).normalized();
}

Triangle3::Triangle3(const Transform3& transform_, bool isNormalFlipped_)
        : Surface3(transform_, isNormalFlipped_) {}

Triangle3::Triangle3(const std::array<Vector3D, 3>& newPoints,
                     const std::array<Vector3D, 3>& newNormals,
                     const std::array<Vector2D, 3>& newUvs,
                     const Transform3& transform_, bool isNormalFlipped_)
        : Surface3(transform_, isNormalFlipped_),
          points(newPoints),
          normals(newNormals),
          uvs(newUvs) {}

Triangle3::Triangle3(const Triangle3& other)
        : Surface3(other),
          points(other.points),
          normals(other.normals),
          uvs(other.uvs) {}

Vector3D Triangle3::closestPointLocal(const Vector3D& otherPoint) const {
    Vector3D n = faceNormal();
    double nd = n.dot(n);
    double d = n.dot(points[0]);
    double t = (d - n.dot(otherPoint)) / nd;

    Vector3D q = t * n + otherPoint;

    Vector3D q01 = (points[1] - points[0]).cross(q - points[0]);
    if (n.dot(q01) < 0) {
        return closestPointOnLine(points[0], points[1], q);
    }

    Vector3D q12 = (points[2] - points[1]).cross(q - points[1]);
    if (n.dot(q12) < 0) {
        return closestPointOnLine(points[1], points[2], q);
    }

    Vector3D q02 = (points[0] - points[2]).cross(q - points[2]);
    if (n.dot(q02) < 0) {
        return closestPointOnLine(points[0], points[2], q);
    }

    double a = area();
    double b0 = 0.5 * q12.length() / a;
    double b1 = 0.5 * q02.length() / a;
    double b2 = 0.5 * q01.length() / a;

    return b0 * points[0] + b1 * points[1] + b2 * points[2];
}

Vector3D Triangle3::closestNormalLocal(const Vector3D& otherPoint) const {
    Vector3D n = faceNormal();
    double nd = n.dot(n);
    double d = n.dot(points[0]);
    double t = (d - n.dot(otherPoint)) / nd;

    Vector3D q = t * n + otherPoint;

    Vector3D q01 = (points[1] - points[0]).cross(q - points[0]);
    if (n.dot(q01) < 0) {
        return closestNormalOnLine(points[0], points[1], normals[0], normals[1],
                                   q);
    }

    Vector3D q12 = (points[2] - points[1]).cross(q - points[1]);
    if (n.dot(q12) < 0) {
        return closestNormalOnLine(points[1], points[2], normals[1], normals[2],
                                   q);
    }

    Vector3D q02 = (points[0] - points[2]).cross(q - points[2]);
    if (n.dot(q02) < 0) {
        return closestNormalOnLine(points[0], points[2], normals[0], normals[2],
                                   q);
    }

    double a = area();
    double b0 = 0.5 * q12.length() / a;
    double b1 = 0.5 * q02.length() / a;
    double b2 = 0.5 * q01.length() / a;

    return (b0 * normals[0] + b1 * normals[1] + b2 * normals[2]).normalized();
}

bool Triangle3::intersectsLocal(const Ray3D& ray) const {
    Vector3D n = faceNormal();
    double nd = n.dot(ray.direction);

    if (nd < std::numeric_limits<double>::epsilon()) {
        return false;
    }

    double d = n.dot(points[0]);
    double t = (d - n.dot(ray.origin)) / nd;

    if (t < 0.0) {
        return false;
    }

    Vector3D q = ray.pointAt(t);

    Vector3D q01 = (points[1] - points[0]).cross(q - points[0]);
    if (n.dot(q01) <= 0.0) {
        return false;
    }

    Vector3D q12 = (points[2] - points[1]).cross(q - points[1]);
    if (n.dot(q12) <= 0.0) {
        return false;
    }

    Vector3D q02 = (points[0] - points[2]).cross(q - points[2]);
    if (n.dot(q02) <= 0.0) {
        return false;
    }

    return true;
}

SurfaceRayIntersection3 Triangle3::closestIntersectionLocal(
        const Ray3D& ray) const {
    SurfaceRayIntersection3 intersection;
    Vector3D n = faceNormal();
    double nd = n.dot(ray.direction);

    if (nd < std::numeric_limits<double>::epsilon()) {
        intersection.isIntersecting = false;
        return intersection;
    }

    double d = n.dot(points[0]);
    double t = (d - n.dot(ray.origin)) / nd;

    if (t < 0.0) {
        intersection.isIntersecting = false;
        return intersection;
    }

    Vector3D q = ray.pointAt(t);

    Vector3D q01 = (points[1] - points[0]).cross(q - points[0]);
    if (n.dot(q01) <= 0.0) {
        intersection.isIntersecting = false;
        return intersection;
    }

    Vector3D q12 = (points[2] - points[1]).cross(q - points[1]);
    if (n.dot(q12) <= 0.0) {
        intersection.isIntersecting = false;
        return intersection;
    }

    Vector3D q02 = (points[0] - points[2]).cross(q - points[2]);
    if (n.dot(q02) <= 0.0) {
        intersection.isIntersecting = false;
        return intersection;
    }

    double a = area();
    double b0 = 0.5 * q12.length() / a;
    double b1 = 0.5 * q02.length() / a;
    double b2 = 0.5 * q01.length() / a;

    Vector3D normal = b0 * normals[0] + b1 * normals[1] + b2 * normals[2];

    intersection.isIntersecting = true;
    intersection.distance = t;
    intersection.point = q;
    intersection.normal = normal.normalized();

    return intersection;
}

BoundingBox3D Triangle3::boundingBoxLocal() const {
    BoundingBox3D box(points[0], points[1]);
    box.merge(points[2]);
    return box;
}

double Triangle3::area() const {
    return 0.5 * (points[1] - points[0]).cross(points[2] - points[0]).length();
}

void Triangle3::getBarycentricCoords(const Vector3D& pt, double* b0, double* b1,
                                     double* b2) const {
    Vector3D q01 = (points[1] - points[0]).cross(pt - points[0]);
    Vector3D q12 = (points[2] - points[1]).cross(pt - points[1]);
    Vector3D q02 = (points[0] - points[2]).cross(pt - points[2]);

    double a = area();
    *b0 = 0.5 * q12.length() / a;
    *b1 = 0.5 * q02.length() / a;
    *b2 = 0.5 * q01.length() / a;
}

Vector3D Triangle3::faceNormal() const {
    Vector3D ret = (points[1] - points[0]).cross(points[2] - points[0]);
    return ret.normalized();
}

void Triangle3::setNormalsToFaceNormal() {
    normals[0] = normals[1] = normals[2] = faceNormal();
}

Triangle3::Builder Triangle3::builder() { return Builder(); }

Triangle3::Builder& Triangle3::Builder::withPoints(
        const std::array<Vector3D, 3>& points) {
    _points = points;
    return *this;
}

Triangle3::Builder& Triangle3::Builder::withNormals(
        const std::array<Vector3D, 3>& normals) {
    _normals = normals;
    return *this;
}

Triangle3::Builder& Triangle3::Builder::withUvs(
        const std::array<Vector2D, 3>& uvs) {
    _uvs = uvs;
    return *this;
}

Triangle3 Triangle3::Builder::build() const {
    return Triangle3(_points, _normals, _uvs, _transform, _isNormalFlipped);
}

Triangle3Ptr Triangle3::Builder::makeShared() const {
    return std::shared_ptr<Triangle3>(
            new Triangle3(_points, _normals, _uvs, _transform, _isNormalFlipped),
            [](Triangle3* obj) { delete obj; });
}

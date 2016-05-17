// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_TRIANGLE3_H_
#define INCLUDE_JET_TRIANGLE3_H_

#include <jet/surface3.h>

namespace jet {

class Triangle3 final : public Surface3 {
 public:
    std::array<Vector3D, 3> points;
    std::array<Vector3D, 3> normals;
    std::array<Vector2D, 3> uvs;

    Triangle3();

    Triangle3(
        const std::array<Vector3D, 3>& newPoints,
        const std::array<Vector3D, 3>& newNormals,
        const std::array<Vector2D, 3>& newUvs);

    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    void getClosestIntersection(
        const Ray3D& ray,
        SurfaceRayIntersection3* intersection) const override;

    bool intersects(const Ray3D& ray) const override;

    BoundingBox3D boundingBox() const override;

    double area() const;

    void getBarycentricCoords(
        const Vector3D& pt,
        double* b0,
        double* b1,
        double* b2) const;

    void sample(double u1, double u2, Vector3D* pt, Vector3D* n) const;

    Vector3D faceNormal() const;

    void setNormalsToFaceNormal();
};

typedef std::shared_ptr<Triangle3> Triangle3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_TRIANGLE3_H_

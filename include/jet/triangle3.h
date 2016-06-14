// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_TRIANGLE3_H_
#define INCLUDE_JET_TRIANGLE3_H_

#include <jet/surface3.h>

namespace jet {

//!
//! \brief 3-D triangle geometry.
//!
//! This class represents 3-D triangle geometry which extends Surface3 by
//! overriding surface-related queries.
//!
class Triangle3 final : public Surface3 {
 public:
    //! Three points.
    std::array<Vector3D, 3> points;

    //! Three normals.
    std::array<Vector3D, 3> normals;

    //! Three UV coordinates.
    std::array<Vector2D, 3> uvs;

    //! Constructs an empty triangle.
    Triangle3();

    //! Constructs a triangle with given \p points, \p normals, and \p uvs.
    Triangle3(
        const std::array<Vector3D, 3>& points,
        const std::array<Vector3D, 3>& normals,
        const std::array<Vector2D, 3>& uvs);

    //! Copy constructor
    Triangle3(const Triangle3& other);

    //! Returns the closest point from the given point \p otherPoint to the
    //! surface.
    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    //! Returns true if the given \p ray intersects with this triangle object.
    bool intersects(const Ray3D& ray) const override;

    //! Returns the bounding box of this triangle object.
    BoundingBox3D boundingBox() const override;

    //! Returns the area of this triangle.
    double area() const;

    //! Returns barycentric coordinates for the given point \p pt.
    void getBarycentricCoords(
        const Vector3D& pt,
        double* b0,
        double* b1,
        double* b2) const;

    //! Returns the face normal of the triangle.
    Vector3D faceNormal() const;

    //! Set Triangle3::normals to the face normal.
    void setNormalsToFaceNormal();

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;
};

typedef std::shared_ptr<Triangle3> Triangle3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_TRIANGLE3_H_

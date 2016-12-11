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
    class Builder;

    //! Three points.
    std::array<Vector3D, 3> points;

    //! Three normals.
    std::array<Vector3D, 3> normals;

    //! Three UV coordinates.
    std::array<Vector2D, 3> uvs;

    //! Constructs an empty triangle.
    explicit Triangle3(bool isNormalFlipped = false);

    //! Constructs a triangle with given \p points, \p normals, and \p uvs.
    Triangle3(
        const std::array<Vector3D, 3>& points,
        const std::array<Vector3D, 3>& normals,
        const std::array<Vector2D, 3>& uvs,
        bool isNormalFlipped = false);

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

    //! Returns builder fox Triangle3.
    static Builder builder();

 protected:
    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    //! Note, the book has different name and interface. This function used to
    //! be getClosestIntersection, but now it is simply
    //! actualClosestIntersection. Also, the book's function do not return
    //! SurfaceRayIntersection3 instance, but rather takes a pointer to existing
    //! SurfaceRayIntersection3 instance and modify its contents.
    SurfaceRayIntersection3 actualClosestIntersection(
        const Ray3D& ray) const override;
};

//! Shared pointer for the Triangle3 type.
typedef std::shared_ptr<Triangle3> Triangle3Ptr;


//!
//! \brief Front-end to create Triangle3 objects step by step.
//!
class Triangle3::Builder final {
 public:
    //! Returns builder with normal direction.
    Builder& withIsNormalFlipped(bool isNormalFlipped);

    //! Returns builder with points.
    Builder& withPoints(const std::array<Vector3D, 3>& points);

    //! Returns builder with normals.
    Builder& withNormals(const std::array<Vector3D, 3>& normals);

    //! Returns builder with uvs.
    Builder& withUvs(const std::array<Vector2D, 3>& uvs);

    //! Builds Triangle3.
    Triangle3 build() const;

    //! Builds shared pointer of Triangle3 instance.
    Triangle3Ptr makeShared() const {
        return std::make_shared<Triangle3>(
            _points,
            _normals,
            _uvs,
            _isNormalFlipped);
    }

 private:
    bool _isNormalFlipped = false;
    std::array<Vector3D, 3> _points;
    std::array<Vector3D, 3> _normals;
    std::array<Vector2D, 3> _uvs;
};

}  // namespace jet

#endif  // INCLUDE_JET_TRIANGLE3_H_

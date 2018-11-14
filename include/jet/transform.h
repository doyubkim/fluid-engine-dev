// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_TRANSFORM_H_
#define INCLUDE_JET_TRANSFORM_H_

#include <jet/bounding_box.h>
#include <jet/matrix.h>
#include <jet/quaternion.h>
#include <jet/ray.h>

namespace jet {

template <size_t N>
class Orientation {};

template <>
class Orientation<2> {
 public:
    Orientation();
    Orientation(double angleInRadian);

    double rotation() const;
    void setRotation(double angleInRadian);

    //! Rotates a point in world coordinate to the local frame.
    Vector2D toLocal(const Vector2D& pointInWorld) const;

    //! Rotates a point in local space to the world coordinate.
    Vector2D toWorld(const Vector2D& pointInLocal) const;

 private:
    double _angle = 0.0;
    double _cosAngle = 1.0;
    double _sinAngle = 0.0;
};

template <>
class Orientation<3> {
 public:
    Orientation();
    Orientation(const QuaternionD& quat);

    const QuaternionD& rotation() const;
    void setRotation(const QuaternionD& quat);

    //! Rotates a point in world coordinate to the local frame.
    Vector3D toLocal(const Vector3D& pointInWorld) const;

    //! Rotates a point in local space to the world coordinate.
    Vector3D toWorld(const Vector3D& pointInLocal) const;

 private:
    QuaternionD _quat;
    Matrix3x3D _rotationMat3 = Matrix3x3D::makeIdentity();
    Matrix3x3D _inverseRotationMat3 = Matrix3x3D::makeIdentity();
};

using Orientation2 = Orientation<2>;
using Orientation3 = Orientation<3>;

//!
//! \brief Represents N-D rigid body transform.
//!
template <size_t N>
class Transform {
 public:
    //! Constructs identity transform.
    Transform();

    //! Constructs a transform with translation and orientation.
    Transform(const Vector<double, N>& translation,
              const Orientation<N>& orientation);

    //! Returns the translation.
    const Vector<double, N>& translation() const;

    //! Sets the traslation.
    void setTranslation(const Vector<double, N>& translation);

    //! Returns the orientation.
    const Orientation<N>& orientation() const;

    //! Sets the orientation.
    void setOrientation(const Orientation<N>& orientation);

    //! Transforms a point in world coordinate to the local frame.
    Vector<double, N> toLocal(const Vector<double, N>& pointInWorld) const;

    //! Transforms a direction in world coordinate to the local frame.
    Vector<double, N> toLocalDirection(
        const Vector<double, N>& dirInWorld) const;

    //! Transforms a ray in world coordinate to the local frame.
    Ray<double, N> toLocal(const Ray<double, N>& rayInWorld) const;

    //! Transforms a bounding box in world coordinate to the local frame.
    BoundingBox<double, N> toLocal(
        const BoundingBox<double, N>& bboxInWorld) const;

    //! Transforms a point in local space to the world coordinate.
    Vector<double, N> toWorld(const Vector<double, N>& pointInLocal) const;

    //! Transforms a direction in local space to the world coordinate.
    Vector<double, N> toWorldDirection(
        const Vector<double, N>& dirInLocal) const;

    //! Transforms a ray in local space to the world coordinate.
    Ray<double, N> toWorld(const Ray<double, N>& rayInLocal) const;

    //! Transforms a bounding box in local space to the world coordinate.
    BoundingBox<double, N> toWorld(
        const BoundingBox<double, N>& bboxInLocal) const;

 private:
    Vector<double, N> _translation;
    Orientation<N> _orientation;
};

using Transform2 = Transform<2>;
using Transform3 = Transform<3>;

}  // namespace jet

#endif  // INCLUDE_JET_TRANSFORM_H_

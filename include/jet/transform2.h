// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_TRANSFORM2_H_
#define INCLUDE_JET_TRANSFORM2_H_

#include <jet/bounding_box2.h>
#include <jet/ray2.h>
#include <jet/vector2.h>

namespace jet {

//!
//! \brief Represents 2-D rigid body transform.
//!
class Transform2 {
 public:
    //! Constructs identity transform.
    Transform2();

    //! Constructs a transform with translation and orientation.
    Transform2(const Vector2D& translation, double orientation);

    //! Returns the translation.
    const Vector2D& translation() const;

    //! Sets the traslation.
    void setTranslation(const Vector2D& translation);

    //! Returns the orientation in radians.
    double orientation() const;

    //! Sets the orientation in radians.
    void setOrientation(double orientation);

    //! Transforms a point in world coordinate to the local frame.
    Vector2D toLocal(const Vector2D& pointInWorld) const;

    //! Transforms a direction in world coordinate to the local frame.
    Vector2D toLocalDirection(const Vector2D& dirInWorld) const;

    //! Transforms a ray in world coordinate to the local frame.
    Ray2D toLocal(const Ray2D& rayInWorld) const;

    //! Transforms a bounding box in world coordinate to the local frame.
    BoundingBox2D toLocal(const BoundingBox2D& bboxInWorld) const;

    //! Transforms a point in local space to the world coordinate.
    Vector2D toWorld(const Vector2D& pointInLocal) const;

    //! Transforms a direction in local space to the world coordinate.
    Vector2D toWorldDirection(const Vector2D& dirInLocal) const;

    //! Transforms a ray in local space to the world coordinate.
    Ray2D toWorld(const Ray2D& rayInLocal) const;

    //! Transforms a bounding box in local space to the world coordinate.
    BoundingBox2D toWorld(const BoundingBox2D& bboxInLocal) const;

 private:
    Vector2D _translation;
    double _orientation = 0.0;
    double _cosAngle = 1.0;
    double _sinAngle = 0.0;
};

}  // namespace jet

#include "detail/transform2-inl.h"

#endif  // INCLUDE_JET_TRANSFORM2_H_

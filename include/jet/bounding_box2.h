// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_BOUNDING_BOX2_H_
#define INCLUDE_JET_BOUNDING_BOX2_H_

#include <jet/bounding_box.h>
#include <jet/ray2.h>
#include <jet/vector2.h>
#include <limits>

namespace jet {

//!
//! \brief      2-D box-ray intersection result.
//!
//! \tparam     T     The value type.
//!
template <typename T>
struct BoundingBoxRayIntersection2 {
    //! True if the box and ray intersects.
    bool isIntersecting = false;

    //! Distance to the first intersection point.
    T tNear = std::numeric_limits<T>::max();

    //! Distance to the second (and the last) intersection point.
    T tFar = std::numeric_limits<T>::max();
};

//!
//! \brief 2-D axis-aligned bounding box class.
//!
//! \tparam T - Real number type.
//! \tparam N - Dimension.
//!
template <typename T>
class BoundingBox<T, 2> {
 public:
    //! Lower corner of the bounding box.
    Vector2<T> lowerCorner;

    //! Upper corner of the bounding box.
    Vector2<T> upperCorner;

    //! Default constructor.
    BoundingBox();

    //! Constructs a box that tightly covers two points.
    BoundingBox(const Vector2<T>& point1, const Vector2<T>& point2);

    //! Constructs a box with other box instance.
    BoundingBox(const BoundingBox& other);

    //! Returns width of the box.
    T width() const;

    //! Returns height of the box.
    T height() const;

    //! Returns length of the box in given axis.
    T length(size_t axis);

    //! Returns true of this box and other box overlaps.
    bool overlaps(const BoundingBox& other) const;

    //! Returns true if the input point is inside of this box.
    bool contains(const Vector2<T>& point) const;

    //! Returns true if the input ray is intersecting with this box.
    bool intersects(const Ray2<T>& ray) const;

    //! Returns intersection.isIntersecting = true if the input ray is
    //! intersecting with this box. If interesects, intersection.tNear is
    //! assigned with distant to the closest intersecting point, and
    //! intersection.tFar with furthest.
    BoundingBoxRayIntersection2<T> closestIntersection(
        const Ray2<T>& ray) const;

    //! Returns the mid-point of this box.
    Vector2<T> midPoint() const;

    //! Returns diagonal length of this box.
    T diagonalLength() const;

    //! Returns squared diagonal length of this box.
    T diagonalLengthSquared() const;

    //! Resets this box to initial state (min=infinite, max=-infinite).
    void reset();

    //! Merges this and other point.
    void merge(const Vector2<T>& point);

    //! Merges this and other box.
    void merge(const BoundingBox& other);

    //! Expands this box by given delta to all direction.
    //! If the width of the box was x, expand(y) will result a box with
    //! x+y+y width.
    void expand(T delta);

    //! Returns corner position. Index starts from x-first order.
    Vector2<T> corner(size_t idx) const;

    //! Returns the clamped point.
    Vector2<T> clamp(const Vector2<T>& pt) const;

    //! Returns true if the box is empty.
    bool isEmpty() const;
};

//! Type alias for 2-D BoundingBox.
template <typename T>
using BoundingBox2 = BoundingBox<T, 2>;

//! Float-type 2-D BoundingBox.
typedef BoundingBox2<float> BoundingBox2F;

//! Double-type 2-D BoundingBox.
typedef BoundingBox2<double> BoundingBox2D;

//! Float-type 2-D box-ray intersection result.
typedef BoundingBoxRayIntersection2<float> BoundingBoxRayIntersection2F;

//! Double-type 2-D box-ray intersection result.
typedef BoundingBoxRayIntersection2<double> BoundingBoxRayIntersection2D;

}  // namespace jet

#include "detail/bounding_box2-inl.h"

#endif  // INCLUDE_JET_BOUNDING_BOX2_H_

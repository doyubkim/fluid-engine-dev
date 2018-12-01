// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_BOUNDING_BOX_H_
#define INCLUDE_JET_BOUNDING_BOX_H_

#include <jet/bounding_box.h>
#include <jet/matrix.h>
#include <jet/ray.h>

#include <limits>

namespace jet {

//!
//! \brief  Box-ray intersection result.
//!
//! \tparam T   The value type.
//!
template <typename T>
struct BoundingBoxRayIntersection {
    //! True if the box and ray intersects.
    bool isIntersecting = false;

    //! Distance to the first intersection point.
    T tNear = std::numeric_limits<T>::max();

    //! Distance to the second (and the last) intersection point.
    T tFar = std::numeric_limits<T>::max();
};

//!
//! \brief  N-D axis-aligned bounding box class.
//!
//! \tparam T   Real number type.
//! \tparam N   Dimension.
//!
template <typename T, size_t N>
class BoundingBox {
 public:
    static_assert(N > 0, "Dimension should be greater than 0");
    static_assert(
        std::is_floating_point<T>::value,
        "BoundingBox only can be instantiated with floating point types");

    using VectorType = Vector<T, N>;
    using RayType = Ray<T, N>;

    //! Lower corner of the bounding box.
    VectorType lowerCorner;

    //! Upper corner of the bounding box.
    VectorType upperCorner;

    //! Default constructor.
    BoundingBox();

    //! Constructs a box that tightly covers two points.
    BoundingBox(const VectorType& point1, const VectorType& point2);

    //! Constructs a box with other box instance.
    BoundingBox(const BoundingBox& other);

    //! Returns the size of the box.
    VectorType size() const;

    //! Returns width of the box.
    T width() const;

    //! Returns height of the box.
    template <typename U = T>
    std::enable_if_t<(N > 1), U> height() const;

    //! Returns depth of the box.
    template <typename U = T>
    std::enable_if_t<(N > 2), U> depth() const;

    //! Returns length of the box in given axis.
    T length(size_t axis);

    //! Returns true of this box and other box overlaps.
    bool overlaps(const BoundingBox& other) const;

    //! Returns true if the input vector is inside of this box.
    bool contains(const VectorType& point) const;

    //! Returns true if the input ray is intersecting with this box.
    bool intersects(const RayType& ray) const;

    //! Returns intersection.isIntersecting = true if the input ray is
    //! intersecting with this box. If interesects, intersection.tNear is
    //! assigned with distant to the closest intersecting point, and
    //! intersection.tFar with furthest.
    BoundingBoxRayIntersection<T> closestIntersection(const RayType& ray) const;

    //! Returns the mid-point of this box.
    VectorType midPoint() const;

    //! Returns diagonal length of this box.
    T diagonalLength() const;

    //! Returns squared diagonal length of this box.
    T diagonalLengthSquared() const;

    //! Resets this box to initial state (min=infinite, max=-infinite).
    void reset();

    //! Merges this and other point.
    void merge(const VectorType& point);

    //! Merges this and other box.
    void merge(const BoundingBox& other);

    //! Expands this box by given delta to all direction.
    //! If the width of the box was x, expand(y) will result a box with
    //! x+y+y width.
    void expand(T delta);

    //! Returns corner position. Index starts from x-first order.
    VectorType corner(size_t idx) const;

    //! Returns the clamped point.
    VectorType clamp(const VectorType& point) const;

    //! Returns true if the box is empty.
    bool isEmpty() const;

    //! Returns box with different value type.
    template <typename U>
    BoundingBox<U, N> castTo() const;
};

template <typename T>
using BoundingBox2 = BoundingBox<T, 2>;

template <typename T>
using BoundingBox3 = BoundingBox<T, 3>;

using BoundingBox2F = BoundingBox2<float>;

using BoundingBox2D = BoundingBox2<double>;

using BoundingBox3F = BoundingBox3<float>;

using BoundingBox3D = BoundingBox3<double>;

using BoundingBoxRayIntersectionF = BoundingBoxRayIntersection<float>;

using BoundingBoxRayIntersectionD = BoundingBoxRayIntersection<double>;

}  // namespace jet

#include "detail/bounding_box-inl.h"

#endif  // INCLUDE_JET_BOUNDING_BOX_H_

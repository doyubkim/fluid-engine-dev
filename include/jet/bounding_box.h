// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_BOUNDING_BOX_H_
#define INCLUDE_JET_BOUNDING_BOX_H_

#include <jet/vector.h>

namespace jet {

//!
//! \brief Generic N-D axis-aligned bounding box class.
//!
//! \tparam T - Real number type.
//! \tparam N - Dimension.
//!
template <typename T, size_t N>
class BoundingBox {
 public:
    static_assert(
        N > 0, "Size of static-sized box should be greater than zero.");

    typedef Vector<T, N> VectorType;

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


    //! Returns true of this box and other box overlaps.
    bool overlaps(const BoundingBox& other) const;

    //! Returns true if the input point is inside of this box.
    bool contains(const VectorType& point) const;

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

    //! Merges this and other boxes.
    void merge(const BoundingBox& other);

    //! Expands this box by given delta to all direction.
    //! If the width of the box was x, expand(y) will result a box with
    //! x+y+y width.
    void expand(T delta);
};

}  // namespace jet

#include "detail/bounding_box-inl.h"

#endif  // INCLUDE_JET_BOUNDING_BOX_H_

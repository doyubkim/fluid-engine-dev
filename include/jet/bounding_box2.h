// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_BOUNDING_BOX2_H_
#define INCLUDE_JET_BOUNDING_BOX2_H_

#include <jet/bounding_box.h>
#include <jet/vector2.h>
#include <jet/ray2.h>
#include <limits>

namespace jet {

template <typename T>
struct BoundingBoxRayIntersection2 {
    bool isIntersecting = false;
    T tNear = std::numeric_limits<T>::max();
    T tFar = std::numeric_limits<T>::max();
};

//!
//! \brief 2-D axis-aligned bounding box class.
//! \tparam T - Real number type.
//! \tparam N - Dimension.
//!
template <typename T>
class BoundingBox<T, 2> {
 public:
    Vector2<T> lowerCorner;
    Vector2<T> upperCorner;

    //! Default constructor.
    BoundingBox();

    //! Constructs a box that tightly covers two points.
    explicit BoundingBox(const Vector2<T>& point1, const Vector2<T>& point2);

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
    void getClosestIntersection(
        const Ray2<T>& ray,
        BoundingBoxRayIntersection2<T>* intersection) const;

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

    //! Merges this and other boxes.
    void merge(const BoundingBox& other);

    //! Expands this box by given delta to all direction.
    //! If the width of the box was x, expand(y) will result a box with
    //! x+y+y width.
    void expand(T delta);

    //! Returns corner position. Index starts from x-first order.
    Vector2<T> corner(size_t idx) const;
};

template <typename T> using BoundingBox2 = BoundingBox<T, 2>;

typedef BoundingBox2<float> BoundingBox2F;
typedef BoundingBox2<double> BoundingBox2D;
typedef BoundingBoxRayIntersection2<float> BoundingBoxRayIntersection2F;
typedef BoundingBoxRayIntersection2<double> BoundingBoxRayIntersection2D;

}  // namespace jet

#include "detail/bounding_box2-inl.h"

#endif  // INCLUDE_JET_BOUNDING_BOX2_H_

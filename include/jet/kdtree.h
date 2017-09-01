// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_KDTREE_H
#define INCLUDE_JET_KDTREE_H

#include <jet/vector2.h>
#include <jet/vector3.h>
#include <jet/vector4.h>
#include <jet/bounding_box2.h>
#include <jet/bounding_box3.h>

#include <vector>

namespace jet {

//! Generic k-d tree structure.
template <typename T, size_t K>
class KdTree {
 public:
    typedef Vector<T, K> Point;
    typedef BoundingBox<T, K> BBox;

    //! Constructs an empty kD-tree instance.
    KdTree();

    //! Builds internal acceleration structure for given points list.
    void build(const ConstArrayAccessor1<Point>& points);

    //!
    //! Invokes the callback function for each nearby point around the origin
    //! within given radius.
    //!
    //! \param[in]  origin   The origin position.
    //! \param[in]  radius   The search radius.
    //! \param[in]  callback The callback function.
    //!
    void forEachNearbyPoint(
        const Point& origin, T radius,
        const std::function<void(size_t, const Point&)>& callback) const;

    //!
    //! Returns true if there are any nearby points for given origin within
    //! radius.
    //!
    //! \param[in]  origin The origin.
    //! \param[in]  radius The radius.
    //!
    //! \return     True if has nearby point, false otherwise.
    //!
    bool hasNearbyPoint(const Point& origin, T radius) const;

    //! Returns index of the nearest point.
    size_t nearestPoint(const Point& origin) const;

 private:
    struct Node {
        size_t flags = 0;
        size_t child = kMaxSize;
        size_t item = kMaxSize;
        Point point;

        Node();
        void initLeaf(size_t it, const Point& pt);
        void initInternal(size_t axis, size_t it, size_t c, const Point& pt);
        bool isLeaf() const;
    };

    std::vector<Point> _points;
    std::vector<Node> _nodes;

    size_t build(size_t nodeIndex, size_t* itemIndices, size_t nItems,
                 size_t currentDepth);
};

}  // namespace jet

#include "detail/kdtree-inl.h"

#endif  // INCLUDE_JET_KDTREE_H

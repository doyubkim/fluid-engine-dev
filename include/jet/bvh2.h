// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_BVH2_H_
#define INCLUDE_JET_BVH2_H_

#include <jet/intersection_query_engine2.h>
#include <jet/nearest_neighbor_query_engine2.h>

#include <vector>

namespace jet {

//!
//! \brief Bounding Volume Hierarchy (BVH) in 2D
//!
//! This class implements the classic bounding volume hierarchy structure in 2D.
//! It implements IntersectionQueryEngine2 in order to support box/ray
//! intersection tests. Also, NearestNeighborQueryEngine2 is implemented to
//! provide nearest neighbor query.
//!
template <typename T>
class Bvh2 final : public IntersectionQueryEngine2<T>,
                   public NearestNeighborQueryEngine2<T> {
 public:
    using ContainerType = std::vector<T>;
    using Iterator = typename ContainerType::iterator;
    using ConstIterator = typename ContainerType::const_iterator;

    //! Default constructor.
    Bvh2();

    //! Builds bounding volume hierarchy.
    void build(const std::vector<T>& items,
               const std::vector<BoundingBox2D>& itemsBounds);

    //! Clears all the contents of this instance.
    void clear();

    //! Returns the nearest neighbor for given point and distance measure
    //! function.
    NearestNeighborQueryResult2<T> nearest(
        const Vector2D& pt,
        const NearestNeighborDistanceFunc2<T>& distanceFunc) const override;

    //! Returns true if given \p box intersects with any of the stored items.
    bool intersects(const BoundingBox2D& box,
                    const BoxIntersectionTestFunc2<T>& testFunc) const override;

    //! Returns true if given \p ray intersects with any of the stored items.
    bool intersects(const Ray2D& ray,
                    const RayIntersectionTestFunc2<T>& testFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const BoundingBox2D& box, const BoxIntersectionTestFunc2<T>& testFunc,
        const IntersectionVisitorFunc2<T>& visitorFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const Ray2D& ray, const RayIntersectionTestFunc2<T>& testFunc,
        const IntersectionVisitorFunc2<T>& visitorFunc) const override;

    //! Returns the closest intersection for given \p ray.
    ClosestIntersectionQueryResult2<T> closestIntersection(
        const Ray2D& ray,
        const GetRayIntersectionFunc2<T>& testFunc) const override;

    //! Returns bounding box of every items.
    const BoundingBox2D& boundingBox() const;

    //! Returns the begin iterator of the item.
    Iterator begin();

    //! Returns the end iterator of the item.
    Iterator end();

    //! Returns the immutable begin iterator of the item.
    ConstIterator begin() const;

    //! Returns the immutable end iterator of the item.
    ConstIterator end() const;

    //! Returns the number of items.
    size_t numberOfItems() const;

    //! Returns the item at \p i.
    const T& item(size_t i) const;

    //! Returns the number of nodes.
    size_t numberOfNodes() const;

    //! Returns the children indices of \p i-th node.
    std::pair<size_t, size_t> children(size_t i) const;

    //! Returns true if \p i-th node is a leaf node.
    bool isLeaf(size_t i) const;

    //! Returns bounding box of \p i-th node.
    const BoundingBox2D& nodeBound(size_t i) const;

    //! Returns item of \p i-th node.
    Iterator itemOfNode(size_t i);

    //! Returns item of \p i-th node.
    ConstIterator itemOfNode(size_t i) const;

 private:
    struct Node {
        char flags;
        union {
            size_t child;
            size_t item;
        };
        BoundingBox2D bound;

        Node();
        void initLeaf(size_t it, const BoundingBox2D& b);
        void initInternal(uint8_t axis, size_t c, const BoundingBox2D& b);
        bool isLeaf() const;
    };

    BoundingBox2D _bound;
    ContainerType _items;
    std::vector<BoundingBox2D> _itemBounds;
    std::vector<Node> _nodes;

    size_t build(size_t nodeIndex, size_t* itemIndices, size_t nItems,
                 size_t currentDepth);

    size_t qsplit(size_t* itemIndices, size_t numItems, double pivot,
                  uint8_t axis);
};
}  // namespace jet

#include "detail/bvh2-inl.h"

#endif  // INCLUDE_JET_BVH2_H_

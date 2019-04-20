// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_BVH3_H_
#define INCLUDE_JET_BVH3_H_

#include <jet/intersection_query_engine3.h>
#include <jet/nearest_neighbor_query_engine3.h>

#include <vector>

namespace jet {

//!
//! \brief Bounding Volume Hierarchy (BVH) in 3D
//!
//! This class implements the classic bounding volume hierarchy structure in 3D.
//! It implements IntersectionQueryEngine3 in order to support box/ray
//! intersection tests. Also, NearestNeighborQueryEngine3 is implemented to
//! provide nearest neighbor query.
//!
template <typename T>
class Bvh3 final : public IntersectionQueryEngine3<T>,
                   public NearestNeighborQueryEngine3<T> {
 public:
    using ContainerType = std::vector<T>;
    using Iterator = typename ContainerType::iterator;
    using ConstIterator = typename ContainerType::const_iterator;

    //! Default constructor.
    Bvh3();

    //! Builds bounding volume hierarchy.
    void build(const std::vector<T>& items,
               const std::vector<BoundingBox3D>& itemsBounds);

    //! Clears all the contents of this instance.
    void clear();

    //! Returns the nearest neighbor for given point and distance measure
    //! function.
    NearestNeighborQueryResult3<T> nearest(
        const Vector3D& pt,
        const NearestNeighborDistanceFunc3<T>& distanceFunc) const override;

    //! Returns true if given \p box intersects with any of the stored items.
    bool intersects(const BoundingBox3D& box,
                    const BoxIntersectionTestFunc3<T>& testFunc) const override;

    //! Returns true if given \p ray intersects with any of the stored items.
    bool intersects(const Ray3D& ray,
                    const RayIntersectionTestFunc3<T>& testFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const BoundingBox3D& box, const BoxIntersectionTestFunc3<T>& testFunc,
        const IntersectionVisitorFunc3<T>& visitorFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const Ray3D& ray, const RayIntersectionTestFunc3<T>& testFunc,
        const IntersectionVisitorFunc3<T>& visitorFunc) const override;

    //! Returns the closest intersection for given \p ray.
    ClosestIntersectionQueryResult3<T> closestIntersection(
        const Ray3D& ray,
        const GetRayIntersectionFunc3<T>& testFunc) const override;

    //! Returns bounding box of every items.
    const BoundingBox3D& boundingBox() const;

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
    const BoundingBox3D& nodeBound(size_t i) const;

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
        BoundingBox3D bound;

        Node();
        void initLeaf(size_t it, const BoundingBox3D& b);
        void initInternal(uint8_t axis, size_t c, const BoundingBox3D& b);
        bool isLeaf() const;
    };

    BoundingBox3D _bound;
    ContainerType _items;
    std::vector<BoundingBox3D> _itemBounds;
    std::vector<Node> _nodes;

    size_t build(size_t nodeIndex, size_t* itemIndices, size_t nItems,
                 size_t currentDepth);

    size_t qsplit(size_t* itemIndices, size_t numItems, double pivot,
                  uint8_t axis);
};
}  // namespace jet

#include "detail/bvh3-inl.h"

#endif  // INCLUDE_JET_BVH3_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_BVH_H_
#define INCLUDE_JET_BVH_H_

#include <jet/array.h>
#include <jet/array_view.h>
#include <jet/intersection_query_engine.h>
#include <jet/nearest_neighbor_query_engine.h>

namespace jet {

//!
//! \brief Bounding Volume Hierarchy (BVH) in N-D
//!
//! This class implements the classic bounding volume hierarchy structure in
//! N-D. It implements IntersectionQueryEngine in order to support box/ray
//! intersection tests. Also, NearestNeighborQueryEngine is implemented to
//! provide nearest neighbor query.
//!
template <typename T, size_t N>
class Bvh final : public IntersectionQueryEngine<T, N>,
                  public NearestNeighborQueryEngine<T, N> {
 public:
    using ContainerType = Array1<T>;
    using iterator = typename ContainerType::iterator;
    using const_iterator = typename ContainerType::const_iterator;

    using TraveralVisitorFunc = std::function<void(size_t)>;

    template <typename Data>
    using TraveralVisitorReduceDataFunc =
        std::function<void(size_t, const Data&)>;

    template <typename Data>
    using TraveralLeafReduceDataFunc = std::function<Data(size_t)>;

    //! Default constructor.
    Bvh();

    //! Builds bounding volume hierarchy.
    void build(const ConstArrayView1<T>& items,
               const ConstArrayView1<BoundingBox<double, N>>& itemsBounds);

    //! Clears all the contents of this instance.
    void clear();

    //! Returns the nearest neighbor for given point and distance measure
    //! function.
    NearestNeighborQueryResult<T, N> nearest(
        const Vector<double, N>& pt,
        const NearestNeighborDistanceFunc<T, N>& distanceFunc) const override;

    //! Returns true if given \p box intersects with any of the stored items.
    bool intersects(
        const BoundingBox<double, N>& box,
        const BoxIntersectionTestFunc<T, N>& testFunc) const override;

    //! Returns true if given \p ray intersects with any of the stored items.
    bool intersects(
        const Ray<double, N>& ray,
        const RayIntersectionTestFunc<T, N>& testFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const BoundingBox<double, N>& box,
        const BoxIntersectionTestFunc<T, N>& testFunc,
        const IntersectionVisitorFunc<T>& visitorFunc) const override;

    //! Invokes \p visitorFunc for every intersecting items.
    void forEachIntersectingItem(
        const Ray<double, N>& ray,
        const RayIntersectionTestFunc<T, N>& testFunc,
        const IntersectionVisitorFunc<T>& visitorFunc) const override;

    //! Returns the closest intersection for given \p ray.
    ClosestIntersectionQueryResult<T, N> closestIntersection(
        const Ray<double, N>& ray,
        const GetRayIntersectionFunc<T, N>& testFunc) const override;

    void preOrderTraversal(const TraveralVisitorFunc& visitorFunc) const;

    void postOrderTraversal(const TraveralVisitorFunc& visitorFunc) const;

    template <typename ReduceData>
    void postOrderTraversal(
        const TraveralVisitorReduceDataFunc<ReduceData>& visitorFunc,
        const TraveralLeafReduceDataFunc<ReduceData>& leafFunc,
        const ReduceData& initData = ReduceData{}) const;

    //! Returns bounding box of every items.
    const BoundingBox<double, N>& boundingBox() const;

    //! Returns the begin iterator of the item.
    iterator begin();

    //! Returns the end iterator of the item.
    iterator end();

    //! Returns the immutable begin iterator of the item.
    const_iterator begin() const;

    //! Returns the immutable end iterator of the item.
    const_iterator end() const;

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
    const BoundingBox<double, N>& nodeBound(size_t i) const;

    //! Returns item of \p i-th node.
    iterator itemOfNode(size_t i);

    //! Returns item of \p i-th node.
    const_iterator itemOfNode(size_t i) const;

 private:
    struct Node {
        char flags;
        union {
            size_t child;
            size_t item;
        };
        BoundingBox<double, N> bound;

        Node();
        void initLeaf(size_t it, const BoundingBox<double, N>& b);
        void initInternal(uint8_t axis, size_t c,
                          const BoundingBox<double, N>& b);
        bool isLeaf() const;
    };

    BoundingBox<double, N> _bound;
    ContainerType _items;
    Array1<BoundingBox<double, N>> _itemBounds;
    Array1<Node> _nodes;

    size_t build(size_t nodeIndex, size_t* itemIndices, size_t nItems,
                 size_t currentDepth);

    size_t qsplit(size_t* itemIndices, size_t numItems, double pivot,
                  uint8_t axis);

    void preOrderTraversal(const Node* node,
                           const TraveralVisitorFunc& visitorFunc) const;

    void postOrderTraversal(const Node* node,
                            const TraveralVisitorFunc& visitorFunc) const;

    template <typename ReduceData>
    ReduceData postOrderTraversal(
        const Node* node,
        const TraveralVisitorReduceDataFunc<ReduceData>& visitorFunc,
        const TraveralLeafReduceDataFunc<ReduceData>& leafFunc,
        const ReduceData& initData) const;
};

//! 2-D BVH type.
template <typename T>
using Bvh2 = Bvh<T, 2>;

//! 3-D BVH type.
template <typename T>
using Bvh3 = Bvh<T, 3>;

}  // namespace jet

#include "detail/bvh-inl.h"

#endif  // INCLUDE_JET_BVH_H_

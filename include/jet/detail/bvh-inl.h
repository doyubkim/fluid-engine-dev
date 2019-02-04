// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_BVH_INL_H_
#define INCLUDE_JET_DETAIL_BVH_INL_H_

#include <jet/bvh.h>
#include <jet/constants.h>
#include <jet/math_utils.h>

#include <numeric>

namespace jet {

template <typename T, size_t N>
Bvh<T, N>::Node::Node() : flags(0) {
    child = kMaxSize;
}

template <typename T, size_t N>
void Bvh<T, N>::Node::initLeaf(size_t it, const BoundingBox<double, N>& b) {
    flags = static_cast<char>(N);
    item = it;
    bound = b;
}

template <typename T, size_t N>
void Bvh<T, N>::Node::initInternal(uint8_t axis, size_t c,
                                   const BoundingBox<double, N>& b) {
    flags = axis;
    child = c;
    bound = b;
}

template <typename T, size_t N>
bool Bvh<T, N>::Node::isLeaf() const {
    return flags == static_cast<char>(N);
}

//

template <typename T, size_t N>
Bvh<T, N>::Bvh() {}

template <typename T, size_t N>
void Bvh<T, N>::build(
    const ConstArrayView1<T>& items,
    const ConstArrayView1<BoundingBox<double, N>>& itemsBounds) {
    _items = items;
    _itemBounds = itemsBounds;

    if (_items.isEmpty()) {
        return;
    }

    _nodes.clear();

    for (size_t i = 0; i < _items.length(); ++i) {
        _bound.merge(_itemBounds[i]);
    }

    Array1<size_t> itemIndices(_items.length());
    std::iota(std::begin(itemIndices), std::end(itemIndices), 0);

    build(0, itemIndices.data(), _items.length(), 0);
}

template <typename T, size_t N>
void Bvh<T, N>::clear() {
    _bound = BoundingBox<double, N>();
    _items.clear();
    _itemBounds.clear();
    _nodes.clear();
}

template <typename T, size_t N>
inline NearestNeighborQueryResult<T, N> Bvh<T, N>::nearest(
    const Vector<double, N>& pt,
    const NearestNeighborDistanceFunc<T, N>& distanceFunc) const {
    NearestNeighborQueryResult<T, N> best;
    best.distance = kMaxD;
    best.item = nullptr;

    // Prepare to traverse BVH
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // Traverse BVH nodes
    const Node* node = _nodes.data();
    while (node != nullptr) {
        if (node->isLeaf()) {
            double dist = distanceFunc(_items[node->item], pt);
            if (dist < best.distance) {
                best.distance = dist;
                best.item = &_items[node->item];
            }

            // Grab next node to process from todo stack
            if (todoPos > 0) {
                // Dequeue
                --todoPos;
                node = todo[todoPos];
            } else {
                break;
            }
        } else {
            const double bestDistSqr = best.distance * best.distance;

            const Node* left = node + 1;
            const Node* right = &_nodes[node->child];

            // If pt is inside the box, then the closestLeft and Right will be
            // identical to pt. This will make distMinLeftSqr and
            // distMinRightSqr zero, meaning that such a box will have higher
            // priority.
            Vector<double, N> closestLeft = left->bound.clamp(pt);
            Vector<double, N> closestRight = right->bound.clamp(pt);

            double distMinLeftSqr = closestLeft.distanceSquaredTo(pt);
            double distMinRightSqr = closestRight.distanceSquaredTo(pt);

            bool shouldVisitLeft = distMinLeftSqr < bestDistSqr;
            bool shouldVisitRight = distMinRightSqr < bestDistSqr;

            const Node* firstChild;
            const Node* secondChild;
            if (shouldVisitLeft && shouldVisitRight) {
                if (distMinLeftSqr < distMinRightSqr) {
                    firstChild = left;
                    secondChild = right;
                } else {
                    firstChild = right;
                    secondChild = left;
                }

                // Enqueue secondChild in todo stack
                todo[todoPos] = secondChild;
                ++todoPos;
                node = firstChild;
            } else if (shouldVisitLeft) {
                node = left;
            } else if (shouldVisitRight) {
                node = right;
            } else {
                if (todoPos > 0) {
                    // Dequeue
                    --todoPos;
                    node = todo[todoPos];
                } else {
                    break;
                }
            }
        }
    }

    return best;
}

template <typename T, size_t N>
inline bool Bvh<T, N>::intersects(
    const BoundingBox<double, N>& box,
    const BoxIntersectionTestFunc<T, N>& testFunc) const {
    if (!_bound.overlaps(box)) {
        return false;
    }

    // prepare to traverse BVH for box
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse BVH nodes for box
    const Node* node = _nodes.data();

    while (node != nullptr) {
        if (node->isLeaf()) {
            if (testFunc(_items[node->item], box)) {
                return true;
            }

            // grab next node to process from todo stack
            if (todoPos > 0) {
                // Dequeue
                --todoPos;
                node = todo[todoPos];
            } else {
                break;
            }
        } else {
            // get node children pointers for box
            const Node* firstChild = node + 1;
            const Node* secondChild = (Node*)&_nodes[node->child];

            // advance to next child node, possibly enqueue other child
            if (!firstChild->bound.overlaps(box)) {
                node = secondChild;
            } else if (!secondChild->bound.overlaps(box)) {
                node = firstChild;
            } else {
                // enqueue secondChild in todo stack
                todo[todoPos] = secondChild;
                ++todoPos;
                node = firstChild;
            }
        }
    }

    return false;
}

template <typename T, size_t N>
inline bool Bvh<T, N>::intersects(
    const Ray<double, N>& ray,
    const RayIntersectionTestFunc<T, N>& testFunc) const {
    if (!_bound.intersects(ray)) {
        return false;
    }

    // prepare to traverse BVH for ray
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse BVH nodes for ray
    const Node* node = _nodes.data();

    while (node != nullptr) {
        if (node->isLeaf()) {
            if (testFunc(_items[node->item], ray)) {
                return true;
            }

            // grab next node to process from todo stack
            if (todoPos > 0) {
                // Dequeue
                --todoPos;
                node = todo[todoPos];
            } else {
                break;
            }
        } else {
            // get node children pointers for ray
            const Node* firstChild;
            const Node* secondChild;
            if (ray.direction[node->flags] > 0.0) {
                firstChild = node + 1;
                secondChild = (Node*)&_nodes[node->child];
            } else {
                firstChild = (Node*)&_nodes[node->child];
                secondChild = node + 1;
            }

            // advance to next child node, possibly enqueue other child
            if (!firstChild->bound.intersects(ray)) {
                node = secondChild;
            } else if (!secondChild->bound.intersects(ray)) {
                node = firstChild;
            } else {
                // enqueue secondChild in todo stack
                todo[todoPos] = secondChild;
                ++todoPos;
                node = firstChild;
            }
        }
    }

    return false;
}

template <typename T, size_t N>
inline void Bvh<T, N>::forEachIntersectingItem(
    const BoundingBox<double, N>& box,
    const BoxIntersectionTestFunc<T, N>& testFunc,
    const IntersectionVisitorFunc<T>& visitorFunc) const {
    if (!_bound.overlaps(box)) {
        return;
    }

    // prepare to traverse BVH for box
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse BVH nodes for box
    const Node* node = _nodes.data();

    while (node != nullptr) {
        if (node->isLeaf()) {
            if (testFunc(_items[node->item], box)) {
                visitorFunc(_items[node->item]);
            }

            // grab next node to process from todo stack
            if (todoPos > 0) {
                // Dequeue
                --todoPos;
                node = todo[todoPos];
            } else {
                break;
            }
        } else {
            // get node children pointers for box
            const Node* firstChild = node + 1;
            const Node* secondChild = (Node*)&_nodes[node->child];

            // advance to next child node, possibly enqueue other child
            if (!firstChild->bound.overlaps(box)) {
                node = secondChild;
            } else if (!secondChild->bound.overlaps(box)) {
                node = firstChild;
            } else {
                // enqueue secondChild in todo stack
                todo[todoPos] = secondChild;
                ++todoPos;
                node = firstChild;
            }
        }
    }
}

template <typename T, size_t N>
inline void Bvh<T, N>::forEachIntersectingItem(
    const Ray<double, N>& ray, const RayIntersectionTestFunc<T, N>& testFunc,
    const IntersectionVisitorFunc<T>& visitorFunc) const {
    if (!_bound.intersects(ray)) {
        return;
    }

    // prepare to traverse BVH for ray
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse BVH nodes for ray
    const Node* node = _nodes.data();

    while (node != nullptr) {
        if (node->isLeaf()) {
            if (testFunc(_items[node->item], ray)) {
                visitorFunc(_items[node->item]);
            }

            // grab next node to process from todo stack
            if (todoPos > 0) {
                // Dequeue
                --todoPos;
                node = todo[todoPos];
            } else {
                break;
            }
        } else {
            // get node children pointers for ray
            const Node* firstChild;
            const Node* secondChild;
            if (ray.direction[node->flags] > 0.0) {
                firstChild = node + 1;
                secondChild = (Node*)&_nodes[node->child];
            } else {
                firstChild = (Node*)&_nodes[node->child];
                secondChild = node + 1;
            }

            // advance to next child node, possibly enqueue other child
            if (!firstChild->bound.intersects(ray)) {
                node = secondChild;
            } else if (!secondChild->bound.intersects(ray)) {
                node = firstChild;
            } else {
                // enqueue secondChild in todo stack
                todo[todoPos] = secondChild;
                ++todoPos;
                node = firstChild;
            }
        }
    }
}

template <typename T, size_t N>
inline ClosestIntersectionQueryResult<T, N> Bvh<T, N>::closestIntersection(
    const Ray<double, N>& ray,
    const GetRayIntersectionFunc<T, N>& testFunc) const {
    ClosestIntersectionQueryResult<T, N> best;
    best.distance = kMaxD;
    best.item = nullptr;

    if (!_bound.intersects(ray)) {
        return best;
    }

    // prepare to traverse BVH for ray
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse BVH nodes for ray
    const Node* node = _nodes.data();

    while (node != nullptr) {
        if (node->isLeaf()) {
            double dist = testFunc(_items[node->item], ray);
            if (dist < best.distance) {
                best.distance = dist;
                best.item = _items.data() + node->item;
            }

            // grab next node to process from todo stack
            if (todoPos > 0) {
                // Dequeue
                --todoPos;
                node = todo[todoPos];
            } else {
                break;
            }
        } else {
            // get node children pointers for ray
            const Node* firstChild;
            const Node* secondChild;
            if (ray.direction[node->flags] > 0.0) {
                firstChild = node + 1;
                secondChild = (Node*)&_nodes[node->child];
            } else {
                firstChild = (Node*)&_nodes[node->child];
                secondChild = node + 1;
            }

            // advance to next child node, possibly enqueue other child
            if (!firstChild->bound.intersects(ray)) {
                node = secondChild;
            } else if (!secondChild->bound.intersects(ray)) {
                node = firstChild;
            } else {
                // enqueue secondChild in todo stack
                todo[todoPos] = secondChild;
                ++todoPos;
                node = firstChild;
            }
        }
    }

    return best;
}

template <typename T, size_t N>
void Bvh<T, N>::preOrderTraversal(
    const TraveralVisitorFunc& visitorFunc) const {
    if (_nodes.isEmpty()) {
        return;
    }

    const Node* node = _nodes.data();

    preOrderTraversal(node, visitorFunc);
}

template <typename T, size_t N>
void Bvh<T, N>::postOrderTraversal(
    const TraveralVisitorFunc& visitorFunc) const {
    if (_nodes.isEmpty()) {
        return;
    }

    const Node* node = _nodes.data();

    postOrderTraversal(node, visitorFunc);
}

template <typename T, size_t N>
template <typename ReduceData>
void Bvh<T, N>::postOrderTraversal(
    const TraveralVisitorReduceDataFunc<ReduceData>& visitorFunc,
    const TraveralLeafReduceDataFunc<ReduceData>& leafFunc,
    const ReduceData& initData) const {
    if (_nodes.isEmpty()) {
        return;
    }

    const Node* node = _nodes.data();
    postOrderTraversal(node, visitorFunc, leafFunc, initData);
}

template <typename T, size_t N>
const BoundingBox<double, N>& Bvh<T, N>::boundingBox() const {
    return _bound;
}

template <typename T, size_t N>
typename Bvh<T, N>::iterator Bvh<T, N>::begin() {
    return _items.begin();
}

template <typename T, size_t N>
typename Bvh<T, N>::iterator Bvh<T, N>::end() {
    return _items.end();
}

template <typename T, size_t N>
typename Bvh<T, N>::const_iterator Bvh<T, N>::begin() const {
    return _items.begin();
}

template <typename T, size_t N>
typename Bvh<T, N>::const_iterator Bvh<T, N>::end() const {
    return _items.end();
}

template <typename T, size_t N>
size_t Bvh<T, N>::numberOfItems() const {
    return _items.size();
}

template <typename T, size_t N>
const T& Bvh<T, N>::item(size_t i) const {
    return _items[i];
}

template <typename T, size_t N>
size_t Bvh<T, N>::numberOfNodes() const {
    return _nodes.length();
}

template <typename T, size_t N>
std::pair<size_t, size_t> Bvh<T, N>::children(size_t i) const {
    if (isLeaf(i)) {
        return std::make_pair(kMaxSize, kMaxSize);
    } else {
        return std::make_pair(i + 1, _nodes[i].child);
    }
}

template <typename T, size_t N>
bool Bvh<T, N>::isLeaf(size_t i) const {
    return _nodes[i].isLeaf();
}

template <typename T, size_t N>
const BoundingBox<double, N>& Bvh<T, N>::nodeBound(size_t i) const {
    return _nodes[i].bound;
}

template <typename T, size_t N>
typename Bvh<T, N>::iterator Bvh<T, N>::itemOfNode(size_t i) {
    if (isLeaf(i)) {
        return _nodes[i].item + begin();
    } else {
        return end();
    }
}

template <typename T, size_t N>
typename Bvh<T, N>::const_iterator Bvh<T, N>::itemOfNode(size_t i) const {
    if (isLeaf(i)) {
        return _nodes[i].item + begin();
    } else {
        return end();
    }
}

template <typename T, size_t N>
size_t Bvh<T, N>::build(size_t nodeIndex, size_t* itemIndices, size_t nItems,
                        size_t currentDepth) {
    // add a node
    _nodes.append(Node());

    // initialize leaf node if termination criteria met
    if (nItems == 1) {
        _nodes[nodeIndex].initLeaf(itemIndices[0], _itemBounds[itemIndices[0]]);
        return currentDepth + 1;
    }

    // find the mid-point of the bounding box to use as a qsplit pivot
    BoundingBox<double, N> nodeBound;
    for (size_t i = 0; i < nItems; ++i) {
        nodeBound.merge(_itemBounds[itemIndices[i]]);
    }

    Vector<double, N> d = nodeBound.upperCorner - nodeBound.lowerCorner;

    // choose which axis to split along
    uint8_t axis = static_cast<uint8_t>(d.dominantAxis());

    double pivot =
        0.5 * (nodeBound.upperCorner[axis] + nodeBound.lowerCorner[axis]);

    // classify primitives with respect to split
    size_t midPoint = qsplit(itemIndices, nItems, pivot, axis);

    // recursively initialize children _nodes
    size_t d0 = build(nodeIndex + 1, itemIndices, midPoint, currentDepth + 1);
    _nodes[nodeIndex].initInternal(axis, _nodes.length(), nodeBound);
    size_t d1 = build(_nodes[nodeIndex].child, itemIndices + midPoint,
                      nItems - midPoint, currentDepth + 1);

    return std::max(d0, d1);
}

template <typename T, size_t N>
size_t Bvh<T, N>::qsplit(size_t* itemIndices, size_t numItems, double pivot,
                         uint8_t axis) {
    double centroid;
    size_t ret = 0;
    for (size_t i = 0; i < numItems; ++i) {
        BoundingBox<double, N> b = _itemBounds[itemIndices[i]];
        centroid = 0.5f * (b.lowerCorner[axis] + b.upperCorner[axis]);
        if (centroid < pivot) {
            std::swap(itemIndices[i], itemIndices[ret]);
            ret++;
        }
    }
    if (ret == 0 || ret == numItems) {
        ret = numItems >> 1;
    }
    return ret;
}

template <typename T, size_t N>
void Bvh<T, N>::preOrderTraversal(
    const Node* node, const TraveralVisitorFunc& visitorFunc) const {
    size_t nodeIndex = node - _nodes.data();

    visitorFunc(nodeIndex);

    if (!node->isLeaf()) {
        const Node* child1 = node + 1;
        const Node* child2 = (Node*)&_nodes[node->child];

        preOrderTraversal(child1, visitorFunc);
        preOrderTraversal(child2, visitorFunc);
    }
}

template <typename T, size_t N>
void Bvh<T, N>::postOrderTraversal(
    const Node* node, const TraveralVisitorFunc& visitorFunc) const {
    size_t nodeIndex = node - _nodes.data();

    if (!node->isLeaf()) {
        const Node* child1 = node + 1;
        const Node* child2 = (Node*)&_nodes[node->child];

        postOrderTraversal(child1, visitorFunc);
        postOrderTraversal(child2, visitorFunc);
    }

    visitorFunc(nodeIndex);
}

template <typename T, size_t N>
template <typename ReduceData>
ReduceData Bvh<T, N>::postOrderTraversal(
    const Node* node,
    const TraveralVisitorReduceDataFunc<ReduceData>& visitorFunc,
    const TraveralLeafReduceDataFunc<ReduceData>& leafFunc,
    const ReduceData& initReduceData) const {
    ReduceData data = initReduceData;

    size_t nodeIndex = node - _nodes.data();

    if (node->isLeaf()) {
        data = leafFunc(nodeIndex);
    } else {
        const Node* child1 = node + 1;
        const Node* child2 = (Node*)&_nodes[node->child];

        data = data + postOrderTraversal(child1, visitorFunc, leafFunc,
                                         initReduceData);
        data = data + postOrderTraversal(child2, visitorFunc, leafFunc,
                                         initReduceData);
    }
    visitorFunc(nodeIndex, data);

    return data;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_BVH_INL_H_

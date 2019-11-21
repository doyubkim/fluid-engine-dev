// Copyright (c) Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_BVH3_INL_H_
#define INCLUDE_JET_DETAIL_BVH3_INL_H_

#include <jet/bvh3.h>
#include <jet/constants.h>
#include <jet/math_utils.h>

#include <numeric>

namespace jet {

template <typename T>
Bvh3<T>::Node::Node() : flags(0) {
    child = kMaxSize;
}

template <typename T>
void Bvh3<T>::Node::initLeaf(size_t it, const BoundingBox3D& b) {
    flags = 3;
    item = it;
    bound = b;
}

template <typename T>
void Bvh3<T>::Node::initInternal(uint8_t axis, size_t c,
                                 const BoundingBox3D& b) {
    flags = axis;
    child = c;
    bound = b;
}

template <typename T>
bool Bvh3<T>::Node::isLeaf() const {
    return flags == 3;
}

//

template <typename T>
Bvh3<T>::Bvh3() {}

template <typename T>
void Bvh3<T>::build(const std::vector<T>& items,
                    const std::vector<BoundingBox3D>& itemsBounds) {
    _items = items;
    _itemBounds = itemsBounds;

    if (_items.empty()) {
        return;
    }

    _nodes.clear();
    _bound = BoundingBox3D();

    for (size_t i = 0; i < _items.size(); ++i) {
        _bound.merge(_itemBounds[i]);
    }

    std::vector<size_t> itemIndices(_items.size());
    std::iota(std::begin(itemIndices), std::end(itemIndices), 0);

    build(0, itemIndices.data(), _items.size(), 0);
}

template <typename T>
void Bvh3<T>::clear() {
    _bound = BoundingBox3D();
    _items.clear();
    _itemBounds.clear();
    _nodes.clear();
}

template <typename T>
inline NearestNeighborQueryResult3<T> Bvh3<T>::nearest(
    const Vector3D& pt,
    const NearestNeighborDistanceFunc3<T>& distanceFunc) const {
    NearestNeighborQueryResult3<T> best;
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
            Vector3D closestLeft = left->bound.clamp(pt);
            Vector3D closestRight = right->bound.clamp(pt);

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

template <typename T>
inline bool Bvh3<T>::intersects(
    const BoundingBox3D& box,
    const BoxIntersectionTestFunc3<T>& testFunc) const {
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

template <typename T>
inline bool Bvh3<T>::intersects(
    const Ray3D& ray, const RayIntersectionTestFunc3<T>& testFunc) const {
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

template <typename T>
inline void Bvh3<T>::forEachIntersectingItem(
    const BoundingBox3D& box, const BoxIntersectionTestFunc3<T>& testFunc,
    const IntersectionVisitorFunc3<T>& visitorFunc) const {
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

template <typename T>
inline void Bvh3<T>::forEachIntersectingItem(
    const Ray3D& ray, const RayIntersectionTestFunc3<T>& testFunc,
    const IntersectionVisitorFunc3<T>& visitorFunc) const {
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

template <typename T>
inline ClosestIntersectionQueryResult3<T> Bvh3<T>::closestIntersection(
    const Ray3D& ray, const GetRayIntersectionFunc3<T>& testFunc) const {
    ClosestIntersectionQueryResult3<T> best;
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

template <typename T>
const BoundingBox3D& Bvh3<T>::boundingBox() const {
    return _bound;
}

template <typename T>
typename Bvh3<T>::Iterator Bvh3<T>::begin() {
    return _items.begin();
}

template <typename T>
typename Bvh3<T>::Iterator Bvh3<T>::end() {
    return _items.end();
}

template <typename T>
typename Bvh3<T>::ConstIterator Bvh3<T>::begin() const {
    return _items.begin();
}

template <typename T>
typename Bvh3<T>::ConstIterator Bvh3<T>::end() const {
    return _items.end();
}

template <typename T>
size_t Bvh3<T>::numberOfItems() const {
    return _items.size();
}

template <typename T>
const T& Bvh3<T>::item(size_t i) const {
    return _items[i];
}

template <typename T>
size_t Bvh3<T>::numberOfNodes() const {
    return _nodes.size();
}

template <typename T>
std::pair<size_t, size_t> Bvh3<T>::children(size_t i) const {
    if (isLeaf(i)) {
        return std::make_pair(kMaxSize, kMaxSize);
    } else {
        return std::make_pair(i + 1, _nodes[i].child);
    }
}

template <typename T>
bool Bvh3<T>::isLeaf(size_t i) const {
    return _nodes[i].isLeaf();
}

template <typename T>
const BoundingBox3D& Bvh3<T>::nodeBound(size_t i) const {
    return _nodes[i].bound;
}

template <typename T>
typename Bvh3<T>::Iterator Bvh3<T>::itemOfNode(size_t i) {
    if (isLeaf(i)) {
        return _nodes[i].item + begin();
    } else {
        return end();
    }
}

template <typename T>
typename Bvh3<T>::ConstIterator Bvh3<T>::itemOfNode(size_t i) const {
    if (isLeaf(i)) {
        return _nodes[i].item + begin();
    } else {
        return end();
    }
}

template <typename T>
size_t Bvh3<T>::build(size_t nodeIndex, size_t* itemIndices, size_t nItems,
                      size_t currentDepth) {
    // add a node
    _nodes.push_back(Node());

    // initialize leaf node if termination criteria met
    if (nItems == 1) {
        _nodes[nodeIndex].initLeaf(itemIndices[0], _itemBounds[itemIndices[0]]);
        return currentDepth + 1;
    }

    // find the mid-point of the bounding box to use as a qsplit pivot
    BoundingBox3D nodeBound;
    for (size_t i = 0; i < nItems; ++i) {
        nodeBound.merge(_itemBounds[itemIndices[i]]);
    }

    Vector3D d = nodeBound.upperCorner - nodeBound.lowerCorner;

    // choose which axis to split along
    uint8_t axis;
    if (d.x > d.y && d.x > d.z) {
        axis = 0;
    } else {
        axis = (d.y > d.z) ? 1 : 2;
    }

    double pivot =
        0.5 * (nodeBound.upperCorner[axis] + nodeBound.lowerCorner[axis]);

    // classify primitives with respect to split
    size_t midPoint = qsplit(itemIndices, nItems, pivot, axis);

    // recursively initialize children _nodes
    size_t d0 = build(nodeIndex + 1, itemIndices, midPoint, currentDepth + 1);
    _nodes[nodeIndex].initInternal(axis, _nodes.size(), nodeBound);
    size_t d1 = build(_nodes[nodeIndex].child, itemIndices + midPoint,
                      nItems - midPoint, currentDepth + 1);

    return std::max(d0, d1);
}

template <typename T>
size_t Bvh3<T>::qsplit(size_t* itemIndices, size_t numItems, double pivot,
                       uint8_t axis) {
    double centroid;
    size_t ret = 0;
    for (size_t i = 0; i < numItems; ++i) {
        BoundingBox3D b = _itemBounds[itemIndices[i]];
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

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_BVH3_INL_H_

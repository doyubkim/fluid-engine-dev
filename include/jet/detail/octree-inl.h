// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_OCTREE_INL_H_
#define INCLUDE_JET_DETAIL_OCTREE_INL_H_

#include <jet/math_utils.h>
#include <jet/octree.h>

#include <numeric>
#include <stack>

namespace jet {

template <typename T>
bool Octree<T>::Node::isLeaf() const {
    return firstChild == kMaxSize;
}

//

template <typename T>
Octree<T>::Octree() {}

template <typename T>
void Octree<T>::build(const std::vector<T>& items, const BoundingBox3D& bound,
                      const BoxIntersectionTestFunc3<T>& testFunc,
                      size_t maxDepth) {
    // Reset items
    _maxDepth = maxDepth;
    _items = items;
    _nodes.clear();

    // Normalize bounding box
    _bbox = bound;
    double maxEdgeLen = max3(_bbox.width(), _bbox.height(), _bbox.depth());
    _bbox.upperCorner =
        _bbox.lowerCorner + Vector3D(maxEdgeLen, maxEdgeLen, maxEdgeLen);

    // Build
    _nodes.resize(1);
    _nodes[0].items.resize(_items.size());
    std::iota(_nodes[0].items.begin(), _nodes[0].items.end(), kZeroSize);

    build(0, 1, _bbox, testFunc);
}

template <typename T>
void Octree<T>::clear() {
    _maxDepth = 1;
    _items.clear();
    _nodes.cloear();
    _bbox = BoundingBox3D();
}

template <typename T>
NearestNeighborQueryResult3<T> Octree<T>::nearest(
    const Vector3D& pt,
    const NearestNeighborDistanceFunc3<T>& distanceFunc) const {
    NearestNeighborQueryResult3<T> best;
    best.distance = kMaxD;
    best.item = nullptr;

    // Prepare to traverse octree
    std::stack<std::pair<const Node*, BoundingBox3D>> todo;

    // Traverse octree nodes
    const Node* node = _nodes.data();
    BoundingBox3D bound = _bbox;
    while (node != nullptr) {
        if (node->isLeaf()) {
            for (size_t itemIdx : node->items) {
                double d = distanceFunc(_items[itemIdx], pt);
                if (d < best.distance) {
                    best.distance = d;
                    best.item = &_items[itemIdx];
                }
            }

            // Grab next node to process from todo stack
            if (todo.empty()) {
                break;
            } else {
                node = todo.top().first;
                bound = todo.top().second;
                todo.pop();
            }
        } else {
            const double bestDistSqr = best.distance * best.distance;

            typedef std::tuple<const Node*, double, BoundingBox3D> NodeDistBox;
            std::array<NodeDistBox, 8> childDistSqrPairs;
            const auto midPoint = bound.midPoint();
            for (int i = 0; i < 8; ++i) {
                const Node* child = &_nodes[node->firstChild + i];
                const auto childBound =
                        BoundingBox3D(bound.corner(i), midPoint);
                Vector3D cp = childBound.clamp(pt);
                double distMinSqr = cp.distanceSquaredTo(pt);

                childDistSqrPairs[i] =
                        std::make_tuple(child, distMinSqr, childBound);
            }
            std::sort(childDistSqrPairs.begin(), childDistSqrPairs.end(),
                      [](const NodeDistBox& a, const NodeDistBox& b) {
                          return std::get<1>(a) > std::get<1>(b);
                      });

            for (int i = 0; i < 8; ++i) {
                const auto& childPair = childDistSqrPairs[i];
                if (std::get<1>(childPair) < bestDistSqr) {
                    todo.emplace(std::get<0>(childPair),
                                 std::get<2>(childPair));
                }
            }

            if (todo.empty()) {
                break;
            }

            node = todo.top().first;
            bound = todo.top().second;
            todo.pop();
        }
    }

    return best;
}

template <typename T>
bool Octree<T>::intersects(const BoundingBox3D& box,
                           const BoxIntersectionTestFunc3<T>& testFunc) const {
    return intersects(box, testFunc, 0, _bbox);
}

template <typename T>
bool Octree<T>::intersects(const Ray3D& ray,
                           const RayIntersectionTestFunc3<T>& testFunc) const {
    return intersects(ray, testFunc, 0, _bbox);
}

template <typename T>
void Octree<T>::forEachIntersectingItem(
    const BoundingBox3D& box, const BoxIntersectionTestFunc3<T>& testFunc,
    const IntersectionVisitorFunc3<T>& visitorFunc) const {
    forEachIntersectingItem(box, testFunc, visitorFunc, 0, _bbox);
}

template <typename T>
void Octree<T>::forEachIntersectingItem(
    const Ray3D& ray, const RayIntersectionTestFunc3<T>& testFunc,
    const IntersectionVisitorFunc3<T>& visitorFunc) const {
    forEachIntersectingItem(ray, testFunc, visitorFunc, 0, _bbox);
}

template <typename T>
ClosestIntersectionQueryResult3<T> Octree<T>::closestIntersection(
    const Ray3D& ray, const GetRayIntersectionFunc3<T>& testFunc) const {
    ClosestIntersectionQueryResult3<T> best;
    best.distance = kMaxD;
    best.item = nullptr;

    return closestIntersection(ray, testFunc, 0, _bbox, best);
}

template <typename T>
typename Octree<T>::Iterator Octree<T>::begin() {
    return _items.begin();
}

template <typename T>
typename Octree<T>::Iterator Octree<T>::end() {
    return _items.end();
}

template <typename T>
typename Octree<T>::ConstIterator Octree<T>::begin() const {
    return _items.begin();
}

template <typename T>
typename Octree<T>::ConstIterator Octree<T>::end() const {
    return _items.end();
}

template <typename T>
size_t Octree<T>::numberOfItems() const {
    return _items.size();
}

template <typename T>
const T& Octree<T>::item(size_t i) const {
    return _items[i];
}

template <typename T>
size_t Octree<T>::numberOfNodes() const {
    return _nodes.size();
}

template <typename T>
const std::vector<size_t>& Octree<T>::itemsAtNode(size_t nodeIdx) const {
    return _nodes[nodeIdx].items;
}

template <typename T>
size_t Octree<T>::childIndex(size_t nodeIdx, size_t childIdx) const {
    return _nodes[nodeIdx].firstChild + childIdx;
}

template <typename T>
const BoundingBox3D& Octree<T>::boundingBox() const {
    return _bbox;
}

template <typename T>
size_t Octree<T>::maxDepth() const {
    return _maxDepth;
}

template <typename T>
void Octree<T>::build(size_t nodeIdx, size_t depth, const BoundingBox3D& bound,
                      const BoxIntersectionTestFunc3<T>& testFunc) {
    if (depth < _maxDepth && !_nodes[nodeIdx].items.empty()) {
        size_t firstChild = _nodes[nodeIdx].firstChild = _nodes.size();
        _nodes.resize(_nodes[nodeIdx].firstChild + 8);

        BoundingBox3D bboxPerNode[8];

        for (int i = 0; i < 8; ++i) {
            bboxPerNode[i] = BoundingBox3D(bound.corner(i), bound.midPoint());
        }

        auto& currentItems = _nodes[nodeIdx].items;
        for (size_t i = 0; i < currentItems.size(); ++i) {
            size_t currentItem = currentItems[i];
            for (int j = 0; j < 8; ++j) {
                if (testFunc(_items[currentItem], bboxPerNode[j])) {
                    _nodes[firstChild + j].items.push_back(currentItem);
                }
            }
        }

        // Remove non-leaf data
        currentItems.clear();

        // Refine
        for (int i = 0; i < 8; ++i) {
            build(firstChild + i, depth + 1, bboxPerNode[i], testFunc);
        }
    }
}

template <typename T>
bool Octree<T>::intersects(const BoundingBox3D& box,
                           const BoxIntersectionTestFunc3<T>& testFunc,
                           size_t nodeIdx, const BoundingBox3D& bound) const {
    if (!box.overlaps(bound)) {
        return false;
    }

    const Node& node = _nodes[nodeIdx];

    if (node.items.size() > 0) {
        for (size_t itemIdx : node.items) {
            if (testFunc(_items[itemIdx], box)) {
                return true;
            }
        }
    }

    if (node.firstChild != kMaxSize) {
        for (int i = 0; i < 8; ++i) {
            if (intersects(box, testFunc, node.firstChild + i,
                           BoundingBox3D(bound.corner(i), bound.midPoint()))) {
                return true;
            }
        }
    }

    return false;
}

template <typename T>
bool Octree<T>::intersects(const Ray3D& ray,
                           const RayIntersectionTestFunc3<T>& testFunc,
                           size_t nodeIdx, const BoundingBox3D& bound) const {
    if (!bound.intersects(ray)) {
        return false;
    }

    const Node& node = _nodes[nodeIdx];

    if (node.items.size() > 0) {
        for (size_t itemIdx : node.items) {
            if (testFunc(_items[itemIdx], ray)) {
                return true;
            }
        }
    }

    if (node.firstChild != kMaxSize) {
        for (int i = 0; i < 8; ++i) {
            if (intersects(ray, testFunc, node.firstChild + i,
                           BoundingBox3D(bound.corner(i), bound.midPoint()))) {
                return true;
            }
        }
    }

    return false;
}

template <typename T>
void Octree<T>::forEachIntersectingItem(
    const BoundingBox3D& box, const BoxIntersectionTestFunc3<T>& testFunc,
    const IntersectionVisitorFunc3<T>& visitorFunc, size_t nodeIdx,
    const BoundingBox3D& bound) const {
    if (!box.overlaps(bound)) {
        return;
    }

    const Node& node = _nodes[nodeIdx];

    if (node.items.size() > 0) {
        for (size_t itemIdx : node.items) {
            if (testFunc(_items[itemIdx], box)) {
                visitorFunc(_items[itemIdx]);
            }
        }
    }

    if (node.firstChild != kMaxSize) {
        for (int i = 0; i < 8; ++i) {
            forEachIntersectingItem(
                box, testFunc, visitorFunc, node.firstChild + i,
                BoundingBox3D(bound.corner(i), bound.midPoint()));
        }
    }
}

template <typename T>
void Octree<T>::forEachIntersectingItem(
    const Ray3D& ray, const RayIntersectionTestFunc3<T>& testFunc,
    const IntersectionVisitorFunc3<T>& visitorFunc, size_t nodeIdx,
    const BoundingBox3D& bound) const {
    if (!bound.intersects(ray)) {
        return;
    }

    const Node& node = _nodes[nodeIdx];

    if (node.items.size() > 0) {
        for (size_t itemIdx : node.items) {
            if (testFunc(_items[itemIdx], ray)) {
                visitorFunc(_items[itemIdx]);
            }
        }
    }

    if (node.firstChild != kMaxSize) {
        for (int i = 0; i < 8; ++i) {
            forEachIntersectingItem(
                ray, testFunc, visitorFunc, node.firstChild + i,
                BoundingBox3D(bound.corner(i), bound.midPoint()));
        }
    }
}

template <typename T>
ClosestIntersectionQueryResult3<T> Octree<T>::closestIntersection(
    const Ray3D& ray, const GetRayIntersectionFunc3<T>& testFunc,
    size_t nodeIdx, const BoundingBox3D& bound,
    ClosestIntersectionQueryResult3<T> best) const {
    if (!bound.intersects(ray)) {
        return best;
    }

    const Node& node = _nodes[nodeIdx];

    if (node.items.size() > 0) {
        for (size_t itemIdx : node.items) {
            double dist = testFunc(_items[itemIdx], ray);
            if (dist < best.distance) {
                best.distance = dist;
                best.item = &_items[itemIdx];
            }
        }
    }

    if (node.firstChild != kMaxSize) {
        for (int i = 0; i < 8; ++i) {
            best = closestIntersection(
                ray, testFunc, node.firstChild + i,
                BoundingBox3D(bound.corner(i), bound.midPoint()), best);
        }
    }

    return best;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_OCTREE_INL_H_

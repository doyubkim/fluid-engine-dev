// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_KDTREE_INL_H_
#define INCLUDE_JET_DETAIL_KDTREE_INL_H_

#include <jet/kdtree.h>

#include <numeric>

namespace jet {

template <typename T, size_t K>
KdTree<T, K>::Node::Node() {
    child = kMaxSize;
}

template <typename T, size_t K>
void KdTree<T, K>::Node::initLeaf(size_t it, const Point& pt) {
    flags = K;
    item = it;
    child = kMaxSize;
    point = pt;
}

template <typename T, size_t K>
void KdTree<T, K>::Node::initInternal(size_t axis, size_t it, size_t c,
                                      const Point& pt) {
    flags = axis;
    item = it;
    child = c;
    point = pt;
}

template <typename T, size_t K>
bool KdTree<T, K>::Node::isLeaf() const {
    return flags == K;
}

//

template <typename T, size_t K>
KdTree<T, K>::KdTree() {}

template <typename T, size_t K>
void KdTree<T, K>::build(const ConstArrayAccessor1<Point>& points) {
    _points.resize(points.size());
    std::copy(points.begin(), points.end(), _points.begin());

    if (_points.empty()) {
        return;
    }

    _nodes.clear();

    std::vector<size_t> itemIndices(_points.size());
    std::iota(std::begin(itemIndices), std::end(itemIndices), 0);

    build(0, itemIndices.data(), _points.size(), 0);
}

template <typename T, size_t K>
void KdTree<T, K>::forEachNearbyPoint(
    const Point& origin, T radius,
    const std::function<void(size_t, const Point&)>& callback) const {
    const T r2 = radius * radius;

    // prepare to traverse the tree for sphere
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse the tree nodes for sphere
    const Node* node = _nodes.data();

    while (node != nullptr) {
        if (node->item != kMaxSize &&
            (node->point - origin).lengthSquared() <= r2) {
            callback(node->item, node->point);
        }

        if (node->isLeaf()) {
            // grab next node to process from todo stack
            if (todoPos > 0) {
                // Dequeue
                --todoPos;
                node = todo[todoPos];
            } else {
                break;
            }
        } else {
            // get node children pointers for sphere
            const Node* firstChild = node + 1;
            const Node* secondChild = (Node*)&_nodes[node->child];

            // advance to next child node, possibly enqueue other child
            const size_t axis = node->flags;
            const T plane = node->point[axis];
            if (plane - origin[axis] > radius) {
                node = firstChild;
            } else if (origin[axis] - plane > radius) {
                node = secondChild;
            } else {
                // enqueue secondChild in todo stack
                todo[todoPos] = secondChild;
                ++todoPos;
                node = firstChild;
            }
        }
    }
}

template <typename T, size_t K>
bool KdTree<T, K>::hasNearbyPoint(const Point& origin, T radius) const {
    const T r2 = radius * radius;

    // prepare to traverse the tree for sphere
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse the tree nodes for sphere
    const Node* node = _nodes.data();

    while (node != nullptr) {
        if (node->item != kMaxSize &&
            (node->point - origin).lengthSquared() <= r2) {
            return true;
        }

        if (node->isLeaf()) {
            // grab next node to process from todo stack
            if (todoPos > 0) {
                // Dequeue
                --todoPos;
                node = todo[todoPos];
            } else {
                break;
            }
        } else {
            // get node children pointers for sphere
            const Node* firstChild = node + 1;
            const Node* secondChild = (Node*)&_nodes[node->child];

            // advance to next child node, possibly enqueue other child
            const size_t axis = node->flags;
            const T plane = node->point[axis];
            if (origin[axis] < plane && plane - origin[axis] > radius) {
                node = firstChild;
            } else if (origin[axis] > plane && origin[axis] - plane > radius) {
                node = secondChild;
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

template <typename T, size_t K>
size_t KdTree<T, K>::nearestPoint(const Point& origin) const {
    // prepare to traverse the tree for sphere
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse the tree nodes for sphere
    const Node* node = _nodes.data();
    size_t nearest = 0;
    T minDist2 = (node->point - origin).lengthSquared();

    while (node != nullptr) {
        const T newDist2 = (node->point - origin).lengthSquared();
        if (newDist2 <= minDist2) {
            nearest = node->item;
            minDist2 = newDist2;
        }

        if (node->isLeaf()) {
            // grab next node to process from todo stack
            if (todoPos > 0) {
                // Dequeue
                --todoPos;
                node = todo[todoPos];
            } else {
                break;
            }
        } else {
            // get node children pointers for sphere
            const Node* firstChild = node + 1;
            const Node* secondChild = (Node*)&_nodes[node->child];

            // advance to next child node, possibly enqueue other child
            const size_t axis = node->flags;
            const T plane = node->point[axis];
            const T minDist = std::sqrt(minDist2);
            if (plane - origin[axis] > minDist) {
                node = firstChild;
            } else if (origin[axis] - plane > minDist) {
                node = secondChild;
            } else {
                // enqueue secondChild in todo stack
                todo[todoPos] = secondChild;
                ++todoPos;
                node = firstChild;
            }
        }
    }

    return nearest;
}

template <typename T, size_t K>
void KdTree<T, K>::reserve(size_t numPoints, size_t numNodes) {
    _points.resize(numPoints);
    _nodes.resize(numNodes);
}

template <typename T, size_t K>
typename KdTree<T, K>::Iterator KdTree<T, K>::begin() {
    return _points.begin();
};

template <typename T, size_t K>
typename KdTree<T, K>::Iterator KdTree<T, K>::end() {
    return _points.end();
};

template <typename T, size_t K>
typename KdTree<T, K>::ConstIterator KdTree<T, K>::begin() const {
    return _points.begin();
};

template <typename T, size_t K>
typename KdTree<T, K>::ConstIterator KdTree<T, K>::end() const {
    return _points.end();
};

template <typename T, size_t K>
typename KdTree<T, K>::NodeIterator KdTree<T, K>::beginNode() {
    return _nodes.begin();
};

template <typename T, size_t K>
typename KdTree<T, K>::NodeIterator KdTree<T, K>::endNode() {
    return _nodes.end();
};

template <typename T, size_t K>
typename KdTree<T, K>::ConstNodeIterator KdTree<T, K>::beginNode() const {
    return _nodes.begin();
};

template <typename T, size_t K>
typename KdTree<T, K>::ConstNodeIterator KdTree<T, K>::endNode() const {
    return _nodes.end();
};

template <typename T, size_t K>
size_t KdTree<T, K>::build(size_t nodeIndex, size_t* itemIndices, size_t nItems,
                           size_t currentDepth) {
    // add a node
    _nodes.emplace_back();

    // initialize leaf node if termination criteria met
    if (nItems == 0) {
        _nodes[nodeIndex].initLeaf(kMaxSize, {});
        return currentDepth + 1;
    }
    if (nItems == 1) {
        _nodes[nodeIndex].initLeaf(itemIndices[0], _points[itemIndices[0]]);
        return currentDepth + 1;
    }

    // choose which axis to split along
    BBox nodeBound;
    for (size_t i = 0; i < nItems; ++i) {
        nodeBound.merge(_points[itemIndices[i]]);
    }
    Point d = nodeBound.upperCorner - nodeBound.lowerCorner;
    size_t axis = static_cast<size_t>(d.dominantAxis());

    // pick mid point
    std::nth_element(itemIndices, itemIndices + nItems / 2,
                     itemIndices + nItems, [&](size_t a, size_t b) {
                         return _points[a][axis] < _points[b][axis];
                     });
    size_t midPoint = nItems / 2;

    // recursively initialize children nodes
    size_t d0 = build(nodeIndex + 1, itemIndices, midPoint, currentDepth + 1);
    _nodes[nodeIndex].initInternal(axis, itemIndices[midPoint], _nodes.size(),
                                   _points[itemIndices[midPoint]]);
    size_t d1 = build(_nodes[nodeIndex].child, itemIndices + midPoint + 1,
                      nItems - midPoint - 1, currentDepth + 1);

    return std::max(d0, d1);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_KDTREE_INL_H_

// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <fbs_helpers.h>

#include <jet/bounding_box2.h>
#include <jet/point_kdtree_searcher2.h>

#include <numeric>

using namespace jet;

PointKdTreeSearcher2::Node::Node() { child = kMaxSize; }

void PointKdTreeSearcher2::Node::initLeaf(size_t it, const Vector2D& pt) {
    flags = 2;
    item = it;
    child = kMaxSize;
    point = pt;
}

void PointKdTreeSearcher2::Node::initInternal(uint8_t axis, size_t it, size_t c,
                                              const Vector2D& pt) {
    flags = axis;
    item = it;
    child = c;
    point = pt;
}

bool PointKdTreeSearcher2::Node::isLeaf() const { return flags == 2; }

//

PointKdTreeSearcher2::PointKdTreeSearcher2() {}

PointKdTreeSearcher2::PointKdTreeSearcher2(const PointKdTreeSearcher2& other) {
    set(other);
}

void PointKdTreeSearcher2::build(const ConstArrayAccessor1<Vector2D>& points) {
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

void PointKdTreeSearcher2::forEachNearbyPoint(
        const Vector2D& origin, double radius,
        const ForEachNearbyPointFunc& callback) const {
    const double r2 = radius * radius;

    // prepare to traverse the tree for sphere
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse the tree nodes for box
    const Node* node = _nodes.data();

    while (node != nullptr) {
        if ((node->point - origin).lengthSquared() <= r2) {
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
            // get node children pointers for box
            const Node* firstChild = node + 1;
            const Node* secondChild = (Node*)&_nodes[node->child];

            // advance to next child node, possibly enqueue other child
            const uint8_t axis = node->flags;
            const double plane = node->point[axis];
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

bool PointKdTreeSearcher2::hasNearbyPoint(const Vector2D& origin,
                                          double radius) const {
    const double r2 = radius * radius;

    // prepare to traverse the tree for sphere
    static const int kMaxTreeDepth = 8 * sizeof(size_t);
    const Node* todo[kMaxTreeDepth];
    size_t todoPos = 0;

    // traverse the tree nodes for box
    const Node* node = _nodes.data();

    while (node != nullptr) {
        if ((node->point - origin).lengthSquared() <= r2) {
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
            // get node children pointers for box
            const Node* firstChild = node + 1;
            const Node* secondChild = (Node*)&_nodes[node->child];

            // advance to next child node, possibly enqueue other child
            const uint8_t axis = node->flags;
            const double plane = node->point[axis];
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

PointNeighborSearcher2Ptr PointKdTreeSearcher2::clone() const {
    return CLONE_W_CUSTOM_DELETER(PointKdTreeSearcher2);
}

PointKdTreeSearcher2& PointKdTreeSearcher2::operator=(
        const PointKdTreeSearcher2& other) {
    set(other);
    return *this;
}

void PointKdTreeSearcher2::set(const PointKdTreeSearcher2& other) {
    _points = other._points;
    _nodes = other._nodes;
}

void PointKdTreeSearcher2::serialize(std::vector<uint8_t>* buffer) const {
    // TODO: More
    (void)buffer;
}

void PointKdTreeSearcher2::deserialize(const std::vector<uint8_t>& buffer) {
    // TODO: More
    (void)buffer;
}

PointKdTreeSearcher2::Builder PointKdTreeSearcher2::builder() {
    return Builder{};
}

size_t PointKdTreeSearcher2::build(size_t nodeIndex, size_t* itemIndices,
                                   size_t nItems, size_t currentDepth) {
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

    // find the mid-point of the bounding box to use as a qsplit pivot
    BoundingBox2D nodeBound;
    for (size_t i = 0; i < nItems; ++i) {
        nodeBound.merge(_points[itemIndices[i]]);
    }
    Vector2D d = nodeBound.upperCorner - nodeBound.lowerCorner;

    // choose which axis to split along
    uint8_t axis = static_cast<uint8_t>(d.dominantAxis());

    // sort itemIndices along the axis
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

//

PointKdTreeSearcher2 PointKdTreeSearcher2::Builder::build() const {
    return PointKdTreeSearcher2{};
}

PointKdTreeSearcher2Ptr PointKdTreeSearcher2::Builder::makeShared() const {
    return std::shared_ptr<PointKdTreeSearcher2>(
            new PointKdTreeSearcher2,
            [](PointKdTreeSearcher2* obj) { delete obj; });
}

PointNeighborSearcher2Ptr
PointKdTreeSearcher2::Builder::buildPointNeighborSearcher() const {
    return makeShared();
}

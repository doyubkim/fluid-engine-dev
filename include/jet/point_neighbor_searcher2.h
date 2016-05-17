// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_NEIGHBOR_SEARCHER2_H_
#define INCLUDE_JET_POINT_NEIGHBOR_SEARCHER2_H_

#include <jet/array_accessor1.h>
#include <jet/vector2.h>
#include <functional>
#include <memory>

namespace jet {

class PointNeighborSearcher2 {
 public:
    typedef std::function<void(size_t, const Vector2D&)>
        ForEachNearbyPointFunc;

    PointNeighborSearcher2();
    virtual ~PointNeighborSearcher2();

    virtual void build(const ConstArrayAccessor1<Vector2D>& points) = 0;

    virtual void forEachNearbyPoint(
        const Vector2D& origin,
        double radius,
        const ForEachNearbyPointFunc& callback) const = 0;

    virtual bool hasNearbyPoint(
        const Vector2D& origin, double radius) const = 0;
};

typedef std::shared_ptr<PointNeighborSearcher2> PointNeighborSearcher2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_POINT_NEIGHBOR_SEARCHER2_H_

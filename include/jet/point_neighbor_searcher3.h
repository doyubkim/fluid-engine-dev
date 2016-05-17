// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_NEIGHBOR_SEARCHER3_H_
#define INCLUDE_JET_POINT_NEIGHBOR_SEARCHER3_H_

#include <jet/array_accessor1.h>
#include <jet/vector3.h>
#include <functional>
#include <memory>

namespace jet {

class PointNeighborSearcher3 {
 public:
    typedef std::function<void(size_t, const Vector3D&)>
        ForEachNearbyPointFunc;

    PointNeighborSearcher3();
    virtual ~PointNeighborSearcher3();

    virtual void build(const ConstArrayAccessor1<Vector3D>& points) = 0;

    virtual void forEachNearbyPoint(
        const Vector3D& origin,
        double radius,
        const ForEachNearbyPointFunc& callback) const = 0;

    virtual bool hasNearbyPoint(
        const Vector3D& origin, double radius) const = 0;
};

typedef std::shared_ptr<PointNeighborSearcher3> PointNeighborSearcher3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_POINT_NEIGHBOR_SEARCHER3_H_


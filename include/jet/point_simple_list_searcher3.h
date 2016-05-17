// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER3_H_
#define INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER3_H_

#include <jet/point_neighbor_searcher3.h>
#include <vector>

namespace jet {

class PointSimpleListSearcher3 final : public PointNeighborSearcher3 {
 public:
    PointSimpleListSearcher3();

    void build(const ConstArrayAccessor1<Vector3D>& points) override;

    void forEachNearbyPoint(
        const Vector3D& origin,
        double radius,
        const ForEachNearbyPointFunc& callback) const override;

    bool hasNearbyPoint(
        const Vector3D& origin, double radius) const override;

 private:
    std::vector<Vector3D> _points;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER3_H_

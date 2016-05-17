// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER2_H_
#define INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER2_H_

#include <jet/point_neighbor_searcher2.h>
#include <vector>

namespace jet {

class PointSimpleListSearcher2 final : public PointNeighborSearcher2 {
 public:
    PointSimpleListSearcher2();

    void build(const ConstArrayAccessor1<Vector2D>& points) override;

    void forEachNearbyPoint(
        const Vector2D& origin,
        double radius,
        const ForEachNearbyPointFunc& callback) const override;

    bool hasNearbyPoint(
        const Vector2D& origin, double radius) const override;

 private:
    std::vector<Vector2D> _points;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER2_H_

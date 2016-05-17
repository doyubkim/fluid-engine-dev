// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_HASH_GRID_SEARCHER2_H_
#define INCLUDE_JET_POINT_HASH_GRID_SEARCHER2_H_

#include <jet/point_neighbor_searcher2.h>
#include <jet/point2.h>
#include <jet/size2.h>
#include <vector>

namespace jet {

class PointHashGridSearcher2 final : public PointNeighborSearcher2 {
 public:
    PointHashGridSearcher2(const Size2& resolution, double gridSpacing);
    PointHashGridSearcher2(
        size_t resolutionX,
        size_t resolutionY,
        double gridSpacing);

    void build(const ConstArrayAccessor1<Vector2D>& points) override;

    void forEachNearbyPoint(
        const Vector2D& origin,
        double radius,
        const ForEachNearbyPointFunc& callback) const override;

    bool hasNearbyPoint(
        const Vector2D& origin, double radius) const override;

    void add(const Vector2D& point);

    const std::vector<std::vector<size_t>>& buckets() const;

    size_t getHashKeyFromBucketIndex(const Point2I& bucketIndex) const;

 private:
    double _gridSpacing = 1.0;
    Point2I _resolution = Point2I(1, 1);
    std::vector<Vector2D> _points;
    std::vector<std::vector<size_t>> _buckets;

    Point2I getBucketIndex(const Vector2D& position) const;

    size_t getHashKeyFromPosition(const Vector2D& position) const;

    void getNearbyKeys(const Vector2D& position, size_t* bucketIndices) const;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_HASH_GRID_SEARCHER2_H_

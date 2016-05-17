// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_HASH_GRID_SEARCHER3_H_
#define INCLUDE_JET_POINT_HASH_GRID_SEARCHER3_H_

#include <jet/point_neighbor_searcher3.h>
#include <jet/point3.h>
#include <jet/size3.h>

#include <vector>

namespace jet {

class PointHashGridSearcher3 final : public PointNeighborSearcher3 {
 public:
    PointHashGridSearcher3(const Size3& resolution, double gridSpacing);
    PointHashGridSearcher3(
        size_t resolutionX,
        size_t resolutionY,
        size_t resolutionZ,
        double gridSpacing);

    void build(const ConstArrayAccessor1<Vector3D>& points) override;

    void forEachNearbyPoint(
        const Vector3D& origin,
        double radius,
        const ForEachNearbyPointFunc& callback) const override;

    bool hasNearbyPoint(
        const Vector3D& origin, double radius) const override;

    void add(const Vector3D& point);

    const std::vector<std::vector<size_t>>& buckets() const;

    size_t getHashKeyFromBucketIndex(const Point3I& bucketIndex) const;

 private:
    double _gridSpacing = 1.0;
    Point3I _resolution = Point3I(1, 1, 1);
    std::vector<Vector3D> _points;
    std::vector<std::vector<size_t>> _buckets;

    Point3I getBucketIndex(const Vector3D& position) const;

    size_t getHashKeyFromPosition(const Vector3D& position) const;

    void getNearbyKeys(const Vector3D& position, size_t* bucketIndices) const;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_HASH_GRID_SEARCHER3_H_

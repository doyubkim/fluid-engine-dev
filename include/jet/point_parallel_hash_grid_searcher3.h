// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_PARALLEL_HASH_GRID_SEARCHER3_H_
#define INCLUDE_JET_POINT_PARALLEL_HASH_GRID_SEARCHER3_H_

#include <jet/point_neighbor_searcher3.h>
#include <jet/point3.h>
#include <jet/size3.h>
#include <vector>

class PointParallelHashGridSearcher3Tests;

namespace jet {

class PointParallelHashGridSearcher3 final : public PointNeighborSearcher3 {
 public:
    PointParallelHashGridSearcher3(
        const Size3& resolution, double gridSpacing);

    PointParallelHashGridSearcher3(
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

    const std::vector<size_t>& startIndexTable() const;

    const std::vector<size_t>& endIndexTable() const;

    const std::vector<size_t>& sortedIndices() const;

    size_t getHashKeyFromBucketIndex(const Point3I& bucketIndex) const;

 private:
    friend class PointParallelHashGridSearcher3Tests;

    double _gridSpacing = 1.0;
    Point3I _resolution = Point3I(1, 1, 1);
    std::vector<Vector3D> _points;
    std::vector<size_t> _keys;
    std::vector<size_t> _startIndexTable;
    std::vector<size_t> _endIndexTable;
    std::vector<size_t> _sortedIndices;

    Point3I getBucketIndex(const Vector3D& position) const;

    size_t getHashKeyFromPosition(const Vector3D& position) const;

    void getNearbyKeys(const Vector3D& position, size_t* bucketIndices) const;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_PARALLEL_HASH_GRID_SEARCHER3_H_

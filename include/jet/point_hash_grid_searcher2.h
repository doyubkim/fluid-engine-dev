// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_HASH_GRID_SEARCHER2_H_
#define INCLUDE_JET_POINT_HASH_GRID_SEARCHER2_H_

#include <jet/point_neighbor_searcher2.h>
#include <jet/point2.h>
#include <jet/size2.h>

#include <vector>

namespace jet {

//!
//! \brief Hash grid-based 2-D point searcher.
//!
//! This class implements 2-D point searcher by using hash grid for its internal
//! acceleration data structure. Each point is recorded to its corresponding
//! bucket where the hashing function is 2-D grid mapping.
//!
class PointHashGridSearcher2 final : public PointNeighborSearcher2 {
 public:
    JET_NEIGHBOR_SEARCHER2_TYPE_NAME(PointHashGridSearcher2)

    class Builder;

    //!
    //! \brief      Constructs hash grid with given resolution and grid spacing.
    //!
    //! This constructor takes hash grid resolution and its grid spacing as
    //! its input parameters. The grid spacing must be 2x or greater than
    //! search radius.
    //!
    //! \param[in]  resolution  The resolution.
    //! \param[in]  gridSpacing The grid spacing.
    //!
    PointHashGridSearcher2(const Size2& resolution, double gridSpacing);

    //!
    //! \brief      Constructs hash grid with given resolution and grid spacing.
    //!
    //! This constructor takes hash grid resolution and its grid spacing as
    //! its input parameters. The grid spacing must be 2x or greater than
    //! search radius.
    //!
    //! \param[in]  resolutionX The resolution x.
    //! \param[in]  resolutionY The resolution y.
    //! \param[in]  gridSpacing The grid spacing.
    //!
    PointHashGridSearcher2(
        size_t resolutionX,
        size_t resolutionY,
        double gridSpacing);

    //! Copy constructor.
    PointHashGridSearcher2(const PointHashGridSearcher2& other);

    //! Builds internal acceleration structure for given points list.
    void build(const ConstArrayAccessor1<Vector2D>& points) override;

    //!
    //! Invokes the callback function for each nearby point around the origin
    //! within given radius.
    //!
    //! \param[in]  origin   The origin position.
    //! \param[in]  radius   The search radius.
    //! \param[in]  callback The callback function.
    //!
    void forEachNearbyPoint(
        const Vector2D& origin,
        double radius,
        const ForEachNearbyPointFunc& callback) const override;

    //!
    //! Returns true if there are any nearby points for given origin within
    //! radius.
    //!
    //! \param[in]  origin The origin.
    //! \param[in]  radius The radius.
    //!
    //! \return     True if has nearby point, false otherwise.
    //!
    bool hasNearbyPoint(
        const Vector2D& origin, double radius) const override;

    //!
    //! \brief      Adds a single point to the hash grid.
    //!
    //! This function adds a single point to the hash grid for future queries.
    //! It can be used for a hash grid that is already built by calling function
    //! PointHashGridSearcher2::build.
    //!
    //! \param[in]  point The point to be added.
    //!
    void add(const Vector2D& point);

    //!
    //! \brief      Returns the internal bucket.
    //!
    //! A bucket is a list of point indices that has same hash value. This
    //! function returns the (immutable) internal bucket structure.
    //!
    //! \return     List of buckets.
    //!
    const std::vector<std::vector<size_t>>& buckets() const;

    //!
    //! Returns the hash value for given 2-D bucket index.
    //!
    //! \param[in]  bucketIndex The bucket index.
    //!
    //! \return     The hash key from bucket index.
    //!
    size_t getHashKeyFromBucketIndex(const Point2I& bucketIndex) const;

    //!
    //! Gets the bucket index from a point.
    //!
    //! \param[in]  position The position of the point.
    //!
    //! \return     The bucket index.
    //!
    Point2I getBucketIndex(const Vector2D& position) const;

    //!
    //! \brief      Creates a new instance of the object with same properties
    //!             than original.
    //!
    //! \return     Copy of this object.
    //!
    PointNeighborSearcher2Ptr clone() const override;

    //! Assignment operator.
    PointHashGridSearcher2& operator=(const PointHashGridSearcher2& other);

    //! Copy from the other instance.
    void set(const PointHashGridSearcher2& other);

    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Returns builder fox PointHashGridSearcher2.
    static Builder builder();

 private:
    double _gridSpacing = 1.0;
    Point2I _resolution = Point2I(1, 1);
    std::vector<Vector2D> _points;
    std::vector<std::vector<size_t>> _buckets;

    size_t getHashKeyFromPosition(const Vector2D& position) const;

    void getNearbyKeys(const Vector2D& position, size_t* bucketIndices) const;
};

//! Shared pointer for the PointHashGridSearcher2 type.
typedef std::shared_ptr<PointHashGridSearcher2> PointHashGridSearcher2Ptr;

//!
//! \brief Front-end to create PointHashGridSearcher2 objects step by step.
//!
class PointHashGridSearcher2::Builder final
    : public PointNeighborSearcherBuilder2 {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Size2& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(double gridSpacing);

    //! Builds PointHashGridSearcher2 instance.
    PointHashGridSearcher2 build() const;

    //! Builds shared pointer of PointHashGridSearcher2 instance.
    PointHashGridSearcher2Ptr makeShared() const;

    //! Returns shared pointer of PointNeighborSearcher3 type.
    PointNeighborSearcher2Ptr buildPointNeighborSearcher() const override;

 private:
    Size2 _resolution{64, 64};
    double _gridSpacing = 1.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_HASH_GRID_SEARCHER2_H_

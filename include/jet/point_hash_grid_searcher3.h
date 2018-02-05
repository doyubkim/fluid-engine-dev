// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_HASH_GRID_SEARCHER3_H_
#define INCLUDE_JET_POINT_HASH_GRID_SEARCHER3_H_

#include <jet/point_neighbor_searcher3.h>
#include <jet/point3.h>
#include <jet/size3.h>

#include <vector>

namespace jet {

//!
//! \brief Hash grid-based 3-D point searcher.
//!
//! This class implements 3-D point searcher by using hash grid for its internal
//! acceleration data structure. Each point is recorded to its corresponding
//! bucket where the hashing function is 3-D grid mapping.
//!
class PointHashGridSearcher3 final : public PointNeighborSearcher3 {
 public:
    JET_NEIGHBOR_SEARCHER3_TYPE_NAME(PointHashGridSearcher3)

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
    PointHashGridSearcher3(const Size3& resolution, double gridSpacing);

    //!
    //! \brief      Constructs hash grid with given resolution and grid spacing.
    //!
    //! This constructor takes hash grid resolution and its grid spacing as
    //! its input parameters. The grid spacing must be 2x or greater than
    //! search radius.
    //!
    //! \param[in]  resolutionX The resolution x.
    //! \param[in]  resolutionY The resolution y.
    //! \param[in]  resolutionY The resolution z.
    //! \param[in]  gridSpacing The grid spacing.
    //!
    PointHashGridSearcher3(
        size_t resolutionX,
        size_t resolutionY,
        size_t resolutionZ,
        double gridSpacing);

    //! Copy constructor.
    PointHashGridSearcher3(const PointHashGridSearcher3& other);

    //! Builds internal acceleration structure for given points list.
    void build(const ConstArrayAccessor1<Vector3D>& points) override;

    //!
    //! Invokes the callback function for each nearby point around the origin
    //! within given radius.
    //!
    //! \param[in]  origin   The origin position.
    //! \param[in]  radius   The search radius.
    //! \param[in]  callback The callback function.
    //!
    void forEachNearbyPoint(
        const Vector3D& origin,
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
        const Vector3D& origin, double radius) const override;

    //!
    //! \brief      Adds a single point to the hash grid.
    //!
    //! This function adds a single point to the hash grid for future queries.
    //! It can be used for a hash grid that is already built by calling function
    //! PointHashGridSearcher3::build.
    //!
    //! \param[in]  point The point to be added.
    //!
    void add(const Vector3D& point);

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
    //! Returns the hash value for given 3-D bucket index.
    //!
    //! \param[in]  bucketIndex The bucket index.
    //!
    //! \return     The hash key from bucket index.
    //!
    size_t getHashKeyFromBucketIndex(const Point3I& bucketIndex) const;

    //!
    //! Gets the bucket index from a point.
    //!
    //! \param[in]  position The position of the point.
    //!
    //! \return     The bucket index.
    //!
    Point3I getBucketIndex(const Vector3D& position) const;

    //!
    //! \brief      Creates a new instance of the object with same properties
    //!             than original.
    //!
    //! \return     Copy of this object.
    //!
    PointNeighborSearcher3Ptr clone() const override;

    //! Assignment operator.
    PointHashGridSearcher3& operator=(const PointHashGridSearcher3& other);

    //! Copy from the other instance.
    void set(const PointHashGridSearcher3& other);

    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Returns builder fox PointHashGridSearcher3.
    static Builder builder();

 private:
    double _gridSpacing = 1.0;
    Point3I _resolution = Point3I(1, 1, 1);
    std::vector<Vector3D> _points;
    std::vector<std::vector<size_t>> _buckets;

    size_t getHashKeyFromPosition(const Vector3D& position) const;

    void getNearbyKeys(const Vector3D& position, size_t* bucketIndices) const;
};

//! Shared pointer for the PointHashGridSearcher3 type.
typedef std::shared_ptr<PointHashGridSearcher3> PointHashGridSearcher3Ptr;

//!
//! \brief Front-end to create PointHashGridSearcher3 objects step by step.
//!
class PointHashGridSearcher3::Builder final
    : public PointNeighborSearcherBuilder3 {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Size3& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(double gridSpacing);

    //! Builds PointHashGridSearcher3 instance.
    PointHashGridSearcher3 build() const;

    //! Builds shared pointer of PointHashGridSearcher3 instance.
    PointHashGridSearcher3Ptr makeShared() const;

    //! Returns shared pointer of PointHashGridSearcher3 type.
    PointNeighborSearcher3Ptr buildPointNeighborSearcher() const override;

 private:
    Size3 _resolution{64, 64, 64};
    double _gridSpacing = 1.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_HASH_GRID_SEARCHER3_H_

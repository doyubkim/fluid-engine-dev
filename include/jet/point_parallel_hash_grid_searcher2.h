// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_PARALLEL_HASH_GRID_SEARCHER2_H_
#define INCLUDE_JET_POINT_PARALLEL_HASH_GRID_SEARCHER2_H_

#include <jet/point_neighbor_searcher2.h>
#include <jet/point2.h>
#include <jet/size2.h>
#include <vector>

class PointParallelHashGridSearcher2Tests;

namespace jet {

//!
//! \brief Parallel version of hash grid-based 2-D point searcher.
//!
//! This class implements parallel version of 2-D point searcher by using hash
//! grid for its internal acceleration data structure. Each point is recorded to
//! its corresponding bucket where the hashing function is 2-D grid mapping.
//!
class PointParallelHashGridSearcher2 final : public PointNeighborSearcher2 {
 public:
    JET_NEIGHBOR_SEARCHER2_TYPE_NAME(PointParallelHashGridSearcher2)

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
    PointParallelHashGridSearcher2(
        const Size2& resolution, double gridSpacing);

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
    PointParallelHashGridSearcher2(
        size_t resolutionX,
        size_t resolutionY,
        double gridSpacing);

    //! Copy constructor.
    PointParallelHashGridSearcher2(const PointParallelHashGridSearcher2& other);

    //!
    //! \brief Builds internal acceleration structure for given points list.
    //!
    //! This function builds the hash grid for given points in parallel.
    //!
    //! \param[in]  points The points to be added.
    //!
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
    //! \brief      Returns the hash key list.
    //!
    //! The hash key list maps sorted point index i to its hash key value.
    //! The sorting order is based on the key value itself.
    //!
    //! \return     The hash key list.
    //!
    const std::vector<size_t>& keys() const;

    //!
    //! \brief      Returns the start index table.
    //!
    //! The start index table maps the hash grid bucket index to starting index
    //! of the sorted point list. Assume the hash key list looks like:
    //!
    //! \code
    //! [5|8|8|10|10|10]
    //! \endcode
    //!
    //! Then startIndexTable and endIndexTable should be like:
    //!
    //! \code
    //! [.....|0|...|1|..|3|..]
    //! [.....|1|...|3|..|6|..]
    //!       ^5    ^8   ^10
    //! \endcode
    //!
    //! So that endIndexTable[i] - startIndexTable[i] is the number points
    //! in i-th table bucket.
    //!
    //! \return     The start index table.
    //!
    const std::vector<size_t>& startIndexTable() const;

    //!
    //! \brief      Returns the end index table.
    //!
    //! The end index table maps the hash grid bucket index to starting index
    //! of the sorted point list. Assume the hash key list looks like:
    //!
    //! \code
    //! [5|8|8|10|10|10]
    //! \endcode
    //!
    //! Then startIndexTable and endIndexTable should be like:
    //!
    //! \code
    //! [.....|0|...|1|..|3|..]
    //! [.....|1|...|3|..|6|..]
    //!       ^5    ^8   ^10
    //! \endcode
    //!
    //! So that endIndexTable[i] - startIndexTable[i] is the number points
    //! in i-th table bucket.
    //!
    //! \return     The end index table.
    //!
    const std::vector<size_t>& endIndexTable() const;

    //!
    //! \brief      Returns the sorted indices of the points.
    //!
    //! When the hash grid is built, it sorts the points in hash key order. But
    //! rather than sorting the original points, this class keeps the shuffled
    //! indices of the points. The list this function returns maps sorted index
    //! i to original index j.
    //!
    //! \return     The sorted indices of the points.
    //!
    const std::vector<size_t>& sortedIndices() const;

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
    PointParallelHashGridSearcher2& operator=(
        const PointParallelHashGridSearcher2& other);

    //! Copy from the other instance.
    void set(const PointParallelHashGridSearcher2& other);

    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Returns builder fox PointParallelHashGridSearcher2.
    static Builder builder();

 private:
    friend class PointParallelHashGridSearcher2Tests;

    double _gridSpacing = 1.0;
    Point2I _resolution = Point2I(1, 1);
    std::vector<Vector2D> _points;
    std::vector<size_t> _keys;
    std::vector<size_t> _startIndexTable;
    std::vector<size_t> _endIndexTable;
    std::vector<size_t> _sortedIndices;

    size_t getHashKeyFromPosition(const Vector2D& position) const;

    void getNearbyKeys(const Vector2D& position, size_t* bucketIndices) const;
};

//! Shared pointer for the PointParallelHashGridSearcher2 type.
typedef std::shared_ptr<PointParallelHashGridSearcher2>
    PointParallelHashGridSearcher2Ptr;

//!
//! \brief Front-end to create PointParallelHashGridSearcher2 objects step by
//!        step.
//!
class PointParallelHashGridSearcher2::Builder final
    : public PointNeighborSearcherBuilder2 {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Size2& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(double gridSpacing);

    //! Builds PointParallelHashGridSearcher2 instance.
    PointParallelHashGridSearcher2 build() const;

    //! Builds shared pointer of PointParallelHashGridSearcher2 instance.
    PointParallelHashGridSearcher2Ptr makeShared() const;

    //! Returns shared pointer of PointNeighborSearcher3 type.
    PointNeighborSearcher2Ptr buildPointNeighborSearcher() const override;

 private:
    Size2 _resolution{64, 64};
    double _gridSpacing = 1.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_PARALLEL_HASH_GRID_SEARCHER2_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_HASH_GRID_SEARCHER_H_
#define INCLUDE_JET_POINT_HASH_GRID_SEARCHER_H_

#include <jet/array.h>
#include <jet/matrix.h>
#include <jet/point_neighbor_searcher.h>

namespace jet {

//!
//! \brief Hash grid-based N-D point searcher.
//!
//! This class implements N-D point searcher by using hash grid for its internal
//! acceleration data structure. Each point is recorded to its corresponding
//! bucket where the hashing function is N-D grid mapping.
//!
template <size_t N>
class PointHashGridSearcher final : public PointNeighborSearcher<N> {
 public:
    JET_NEIGHBOR_SEARCHER_TYPE_NAME(PointHashGridSearcher, N)

    class Builder;

    using typename PointNeighborSearcher<N>::ForEachNearbyPointFunc;

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
    PointHashGridSearcher(const Vector<size_t, N>& resolution,
                          double gridSpacing);

    //! Copy constructor.
    PointHashGridSearcher(const PointHashGridSearcher& other);

    //! Builds internal acceleration structure for given points list.
    void build(const ConstArrayView1<Vector<double, N>>& points) override;

    //!
    //! Invokes the callback function for each nearby point around the origin
    //! within given radius.
    //!
    //! \param[in]  origin   The origin position.
    //! \param[in]  radius   The search radius.
    //! \param[in]  callback The callback function.
    //!
    void forEachNearbyPoint(
        const Vector<double, N>& origin, double radius,
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
    bool hasNearbyPoint(const Vector<double, N>& origin,
                        double radius) const override;

    //!
    //! \brief      Adds a single point to the hash grid.
    //!
    //! This function adds a single point to the hash grid for future queries.
    //! It can be used for a hash grid that is already built by calling function
    //! PointHashGridSearcher::build.
    //!
    //! \param[in]  point The point to be added.
    //!
    void add(const Vector<double, N>& point);

    //!
    //! \brief      Returns the internal bucket.
    //!
    //! A bucket is a list of point indices that has same hash value. This
    //! function returns the (immutable) internal bucket structure.
    //!
    //! \return     List of buckets.
    //!
    const Array1<Array1<size_t>>& buckets() const;

    //!
    //! Returns the hash value for given N-D bucket index.
    //!
    //! \param[in]  bucketIndex The bucket index.
    //!
    //! \return     The hash key from bucket index.
    //!
    size_t getHashKeyFromBucketIndex(
        const Vector<ssize_t, N>& bucketIndex) const;

    //!
    //! Gets the bucket index from a point.
    //!
    //! \param[in]  position The position of the point.
    //!
    //! \return     The bucket index.
    //!
    Vector<ssize_t, N> getBucketIndex(const Vector<double, N>& position) const;

    //!
    //! \brief      Creates a new instance of the object with same properties
    //!             than original.
    //!
    //! \return     Copy of this object.
    //!
    std::shared_ptr<PointNeighborSearcher<N>> clone() const override;

    //! Assignment operator.
    PointHashGridSearcher& operator=(const PointHashGridSearcher& other);

    //! Copy from the other instance.
    void set(const PointHashGridSearcher& other);

    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Returns builder fox PointHashGridSearcher.
    static Builder builder();

 private:
    double _gridSpacing = 1.0;
    Vector<ssize_t, N> _resolution = Vector<ssize_t, N>::makeConstant(1);
    Array1<Vector<double, N>> _points;
    Array1<Array1<size_t>> _buckets;

    size_t getHashKeyFromPosition(const Vector<double, N>& position) const;

    void getNearbyKeys(const Vector<double, N>& position,
                       size_t* bucketIndices) const;

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> serialize(
        const PointHashGridSearcher<2>& searcher, std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> serialize(
        const PointHashGridSearcher<3>& searcher, std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> deserialize(
        const std::vector<uint8_t>& buffer, PointHashGridSearcher<2>& searcher);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> deserialize(
        const std::vector<uint8_t>& buffer, PointHashGridSearcher<3>& searcher);
};

//! 2-D PointHashGridSearcher type.
using PointHashGridSearcher2 = PointHashGridSearcher<2>;

//! 3-D PointHashGridSearcher type.
using PointHashGridSearcher3 = PointHashGridSearcher<3>;

//! Shared pointer for the PointHashGridSearcher2 type.
using PointHashGridSearcher2Ptr = std::shared_ptr<PointHashGridSearcher2>;

//! Shared pointer for the PointHashGridSearcher3 type.
using PointHashGridSearcher3Ptr = std::shared_ptr<PointHashGridSearcher3>;

//!
//! \brief Front-end to create PointHashGridSearcher objects step by step.
//!
template <size_t N>
class PointHashGridSearcher<N>::Builder final
    : public PointNeighborSearcherBuilder<N> {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Vector<size_t, N>& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(double gridSpacing);

    //! Builds PointHashGridSearcher instance.
    PointHashGridSearcher<N> build() const;

    //! Builds shared pointer of PointHashGridSearcher instance.
    std::shared_ptr<PointHashGridSearcher<N>> makeShared() const;

    //! Returns shared pointer of PointNeighborSearcher3 type.
    std::shared_ptr<PointNeighborSearcher<N>> buildPointNeighborSearcher()
        const override;

 private:
    Vector<size_t, N> _resolution = Vector<size_t, N>::makeConstant(64);
    double _gridSpacing = 1.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_HASH_GRID_SEARCHER_H_

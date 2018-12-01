// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_PARALLEL_HASH_GRID_SEARCHER_H_
#define INCLUDE_JET_POINT_PARALLEL_HASH_GRID_SEARCHER_H_

#include <jet/matrix.h>
#include <jet/point_neighbor_searcher.h>

namespace jet {

//!
//! \brief Parallel version of hash grid-based N-D point searcher.
//!
//! This class implements parallel version of N-D point searcher by using hash
//! grid for its internal acceleration data structure. Each point is recorded to
//! its corresponding bucket where the hashing function is N-D grid mapping.
//!
template <size_t N>
class PointParallelHashGridSearcher final : public PointNeighborSearcher<N> {
 public:
    JET_NEIGHBOR_SEARCHER_TYPE_NAME(PointParallelHashGridSearcher, N)

    class Builder;

    using typename PointNeighborSearcher<N>::ForEachNearbyPointFunc;
    using PointNeighborSearcher<N>::build;

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
    PointParallelHashGridSearcher(const Vector<size_t, N>& resolution,
                                  double gridSpacing);

    //! Copy constructor.
    PointParallelHashGridSearcher(const PointParallelHashGridSearcher& other);

    //!
    //! \brief Builds internal acceleration structure for given points list and max search radius.
    //!
    //! This function builds the hash grid for given points in parallel.
    //!
    //! \param[in]  points          The points to be added.
    //! \param[in]  maxSearchRadius Max search radius.
    //!
    void build(const ConstArrayView1<Vector<double, N>>& points, double maxSearchRadius) override;

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
    //! \brief      Returns the hash key list.
    //!
    //! The hash key list maps sorted point index i to its hash key value.
    //! The sorting order is based on the key value itself.
    //!
    //! \return     The hash key list.
    //!
    ConstArrayView1<size_t> keys() const;

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
    ConstArrayView1<size_t> startIndexTable() const;

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
    ConstArrayView1<size_t> endIndexTable() const;

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
    ConstArrayView1<size_t> sortedIndices() const;

    //!
    //! \brief      Creates a new instance of the object with same properties
    //!             than original.
    //!
    //! \return     Copy of this object.
    //!
    std::shared_ptr<PointNeighborSearcher<N>> clone() const override;

    //! Assignment operator.
    PointParallelHashGridSearcher& operator=(
        const PointParallelHashGridSearcher& other);

    //! Copy from the other instance.
    void set(const PointParallelHashGridSearcher& other);

    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Returns builder fox PointParallelHashGridSearcher.
    static Builder builder();

 private:
    friend class PointParallelHashGridSearcherTests;

    double _gridSpacing = 1.0;
    Vector<ssize_t, N> _resolution = Vector<ssize_t, N>::makeConstant(1);
    Array1<Vector<double, N>> _points;
    Array1<size_t> _keys;
    Array1<size_t> _startIndexTable;
    Array1<size_t> _endIndexTable;
    Array1<size_t> _sortedIndices;

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> serialize(
        const PointParallelHashGridSearcher<2>& searcher, std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> serialize(
        const PointParallelHashGridSearcher<3>& searcher, std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> deserialize(
        const std::vector<uint8_t>& buffer, PointParallelHashGridSearcher<2>& searcher);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> deserialize(
        const std::vector<uint8_t>& buffer, PointParallelHashGridSearcher<3>& searcher);
};

//! 2-D PointParallelHashGridSearcher type.
using PointParallelHashGridSearcher2 = PointParallelHashGridSearcher<2>;

//! 3-D PointParallelHashGridSearcher type.
using PointParallelHashGridSearcher3 = PointParallelHashGridSearcher<3>;

//! Shared pointer for the PointParallelHashGridSearcher2 type.
using PointParallelHashGridSearcher2Ptr =
    std::shared_ptr<PointParallelHashGridSearcher2>;

//! Shared pointer for the PointParallelHashGridSearcher3 type.
using PointParallelHashGridSearcher3Ptr =
    std::shared_ptr<PointParallelHashGridSearcher3>;

//!
//! \brief Front-end to create PointParallelHashGridSearcher objects step by
//!        step.
//!
template <size_t N>
class PointParallelHashGridSearcher<N>::Builder final
    : public PointNeighborSearcherBuilder<N> {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Vector<size_t, N>& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(double gridSpacing);

    //! Builds PointParallelHashGridSearcher instance.
    PointParallelHashGridSearcher<N> build() const;

    //! Builds shared pointer of PointParallelHashGridSearcher instance.
    std::shared_ptr<PointParallelHashGridSearcher<N>> makeShared() const;

    //! Returns shared pointer of PointNeighborSearcher type.
    std::shared_ptr<PointNeighborSearcher<N>> buildPointNeighborSearcher()
        const override;

 private:
    Vector<size_t, N> _resolution = Vector<size_t, N>::makeConstant(64);
    double _gridSpacing = 1.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_PARALLEL_HASH_GRID_SEARCHER_H_

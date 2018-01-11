// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_POINT_HASH_GRID_SEARCHER3_H_
#define INCLUDE_JET_CUDA_POINT_HASH_GRID_SEARCHER3_H_

#ifdef JET_USE_CUDA

#include <jet/cuda_array_view1.h>
#include <jet/point3.h>
#include <jet/size3.h>
#include <jet/vector3.h>

#include <thrust/device_vector.h>

namespace jet {

namespace experimental {

//!
//! \brief Parallel version of hash grid-based 3-D point searcher.
//!
//! This class implements parallel version of 3-D point searcher by using hash
//! grid for its internal acceleration data structure. Each point is recorded to
//! its corresponding bucket where the hashing function is 3-D grid mapping.
//!
class CudaPointHashGridSearcher3 final {
 public:
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
    CudaPointHashGridSearcher3(const Size3& resolution, float gridSpacing);

    //!
    //! \brief      Constructs hash grid with given resolution and grid spacing.
    //!
    //! This constructor takes hash grid resolution and its grid spacing as
    //! its input parameters. The grid spacing must be 2x or greater than
    //! search radius.
    //!
    //! \param[in]  resolutionX The resolution x.
    //! \param[in]  resolutionY The resolution y.
    //! \param[in]  resolutionZ The resolution z.
    //! \param[in]  gridSpacing The grid spacing.
    //!
    CudaPointHashGridSearcher3(size_t resolutionX, size_t resolutionY,
                               size_t resolutionZ, float gridSpacing);

    //! Copy constructor
    CudaPointHashGridSearcher3(const CudaPointHashGridSearcher3& other);

    //!
    //! \brief Builds internal acceleration structure for given points list.
    //!
    //! This function builds the hash grid for given points in parallel.
    //!
    //! \param[in]  points The points to be added.
    //!
    void build(const CudaArrayView1<float4>& points);
#if 0
    //!
    //! Invokes the callback function for each nearby point around the origin
    //! within given radius.
    //!
    //! \param[in]  origin   The origin position.
    //! \param[in]  radius   The search radius.
    //! \param[in]  callback The callback function.
    //!
    void forEachNearbyPoint(
        const Vector3D& origin, float radius,
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
    bool hasNearbyPoint(const Vector3D& origin, float radius) const override;
#endif
    //!
    //! \brief      Returns the hash key list.
    //!
    //! The hash key list maps sorted point index i to its hash key value.
    //! The sorting order is based on the key value itself.
    //!
    //! \return     The hash key list.
    //!
    CudaArrayView1<size_t> keys() const;

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
    CudaArrayView1<size_t> startIndexTable() const;

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
    CudaArrayView1<size_t> endIndexTable() const;

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
    CudaArrayView1<size_t> sortedIndices() const;
#if 0

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
    CudaPointHashGridSearcher3& operator=(
        const CudaPointHashGridSearcher3& other);
#endif
    //! Copy from the other instance.
    void set(const CudaPointHashGridSearcher3& other);
#if 0
    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;
#endif
    //! Returns builder fox CudaPointHashGridSearcher3.
    static Builder builder();

 private:
    float _gridSpacing = 1.0f;
    Point3I _resolution = Point3I(1, 1, 1);
    CudaArray1<float4> _points;
    CudaArray1<size_t> _keys;
    CudaArray1<size_t> _startIndexTable;
    CudaArray1<size_t> _endIndexTable;
    CudaArray1<size_t> _sortedIndices;
    /*
        size_t getHashKeyFromPosition(const Vector3D& position) const;

        void getNearbyKeys(const Vector3D& position, size_t* bucketIndices)
       const;
        */
};

//! Shared pointer for the CudaPointHashGridSearcher3 type.
typedef std::shared_ptr<CudaPointHashGridSearcher3>
    CudaPointHashGridSearcher3Ptr;

//!
//! \brief Front-end to create CudaPointHashGridSearcher3 objects step by
//!        step.
//!
class CudaPointHashGridSearcher3::Builder final {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Size3& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(float gridSpacing);

    //! Builds CudaPointHashGridSearcher3 instance.
    CudaPointHashGridSearcher3 build() const;

    //! Builds shared pointer of CudaPointHashGridSearcher3 instance.
    CudaPointHashGridSearcher3Ptr makeShared() const;

 private:
    Size3 _resolution{64, 64, 64};
    float _gridSpacing = 1.0f;
};

}  // namespace experimental

}  // namespace jet

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_POINT_HASH_GRID_SEARCHER3_H_

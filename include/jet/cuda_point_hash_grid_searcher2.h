// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_POINT_HASH_GRID_SEARCHER2_H_
#define INCLUDE_JET_CUDA_POINT_HASH_GRID_SEARCHER2_H_

#ifdef JET_USE_CUDA

#include <jet/cuda_array_view1.h>
#include <jet/point2.h>
#include <jet/size2.h>
#include <jet/vector2.h>

#include <cuda_runtime.h>

namespace jet {

namespace experimental {

//!
//! \brief Parallel version of hash grid-based 2-D point searcher.
//!
//! This class implements parallel version of 2-D point searcher by using hash
//! grid for its internal acceleration data structure. Each point is recorded to
//! its corresponding bucket where the hashing function is 2-D grid mapping.
//!
class CudaPointHashGridSearcher2 final {
 public:
    class Builder;

    class HashUtils {
     public:
        inline JET_CUDA_HOST_DEVICE HashUtils();

        inline JET_CUDA_HOST_DEVICE HashUtils(float gridSpacing,
                                              uint2 resolution);

        inline JET_CUDA_HOST_DEVICE void getNearbyKeys(
            float2 position, uint32_t* nearbyKeys) const;

        inline JET_CUDA_HOST_DEVICE int2 getBucketIndex(float2 position) const;

        inline JET_CUDA_HOST_DEVICE uint32_t
        getHashKeyFromBucketIndex(int2 bucketIndex) const;

        inline JET_CUDA_HOST_DEVICE uint32_t
        getHashKeyFromPosition(float2 position) const;

     private:
        float _gridSpacing;
        uint2 _resolution;
    };

    template <typename Callback>
    class ForEachNearbyPointFunc {
     public:
        inline JET_CUDA_HOST_DEVICE ForEachNearbyPointFunc(
            float r, float gridSpacing, uint2 resolution, const uint32_t* sit,
            const uint32_t* eit, const uint32_t* si, const float2* p,
            const float2* o, Callback cb);

        template <typename Index>
        inline JET_CUDA_HOST_DEVICE void operator()(Index idx);

     private:
        HashUtils _hashUtils;
        float _radius;
        const uint32_t* _startIndexTable;
        const uint32_t* _endIndexTable;
        const uint32_t* _sortedIndices;
        const float2* _points;
        const float2* _origins;
        Callback _callback;
    };

    //!
    //! \brief Constructs hash grid with given resolution and grid spacing.
    //!
    //! This constructor takes hash grid resolution and its grid spacing as
    //! its input parameters. The grid spacing must be 2x or greater than
    //! search radius.
    //!
    //! \param[in]  resolution  The resolution.
    //! \param[in]  gridSpacing The grid spacing.
    //!
    CudaPointHashGridSearcher2(const uint2& resolution, float gridSpacing);

    //!
    //! \brief Constructs hash grid with given resolution and grid spacing.
    //!
    //! This constructor takes hash grid resolution and its grid spacing as
    //! its input parameters. The grid spacing must be 2x or greater than
    //! search radius.
    //!
    //! \param[in]  resolutionX The resolution x.
    //! \param[in]  resolutionY The resolution y.
    //! \param[in]  gridSpacing The grid spacing.
    //!
    CudaPointHashGridSearcher2(uint32_t resolutionX, uint32_t resolutionY,
                               float gridSpacing);

    //! Copy constructor
    CudaPointHashGridSearcher2(const CudaPointHashGridSearcher2& other);

    //!
    //! \brief Builds internal acceleration structure for given points list.
    //!
    //! This function builds the hash grid for given points in parallel.
    //!
    //! \param[in]  points The points to be added.
    //!
    void build(const ConstCudaArrayView1<float2>& points);

    //!
    //! Invokes the callback function for each nearby point around the
    //! origin within given radius.
    //!
    //! \param[in]  origins  Array of the origin positions.
    //! \param[in]  radii    The search radius.
    //! \param[in]  callback The callback function.
    //!
    template <typename Callback>
    void forEachNearbyPoint(const ConstCudaArrayView1<float2>& origins,
                            float radius, Callback callback) const;

#if 0
    //!
    //! Returns true if there are any nearby points for given origin within
    //! radius.
    //!
    //! \param[in]  origin The origin.
    //! \param[in]  radius The radius.
    //!
    //! \return     True if has nearby point, false otherwise.
    //!
    bool hasNearbyPoint(const Vector2D& origin, float radius) const override;
#endif

    float gridSpacing() const;

    Size2 resolution() const;

    ConstCudaArrayView1<float2> sortedPoints() const;

    //!
    //! \brief Returns the hash key list.
    //!
    //! The hash key list maps sorted point index i to its hash key value.
    //! The sorting order is based on the key value itself.
    //!
    //! \return The hash key list.
    //!
    ConstCudaArrayView1<uint32_t> keys() const;

    //!
    //! \brief Returns the start index table.
    //!
    //! The start index table maps the hash grid bucket index to starting
    //! index of the sorted point list. Assume the hash key list looks like:
    //!
    //! \code
    //! [5|8|8|10|10|10]
    //! \endcode
    //!
    //! Then startIndexTable and endIndexTable should be like:
    //!
    //! \code
    //! [.....|0|...|1|..|2|..]
    //! [.....|1|...|2|..|6|..]
    //!       ^5    ^8   ^10
    //! \endcode
    //!
    //! So that endIndexTable[i] - startIndexTable[i] is the number points
    //! in i-th table bucket.
    //!
    //! \return The start index table.
    //!
    ConstCudaArrayView1<uint32_t> startIndexTable() const;

    //!
    //! \brief Returns the end index table.
    //!
    //! The end index table maps the hash grid bucket index to starting
    //! index of the sorted point list. Assume the hash key list looks like:
    //!
    //! \code
    //! [5|8|8|10|10|10]
    //! \endcode
    //!
    //! Then startIndexTable and endIndexTable should be like:
    //!
    //! \code
    //! [.....|0|...|1|..|2|..]
    //! [.....|1|...|2|..|6|..]
    //!       ^5    ^8   ^10
    //! \endcode
    //!
    //! So that endIndexTable[i] - startIndexTable[i] is the number points
    //! in i-th table bucket.
    //!
    //! \return The end index table.
    //!
    ConstCudaArrayView1<uint32_t> endIndexTable() const;

    //!
    //! \brief Returns the sorted indices of the points.
    //!
    //! When the hash grid is built, it sorts the points in hash key order.
    //! But rather than sorting the original points, this class keeps the
    //! shuffled indices of the points. The list this function returns maps
    //! sorted index i to original index j.
    //!
    //! \return The sorted indices of the points.
    //!
    ConstCudaArrayView1<uint32_t> sortedIndices() const;

    //! Assignment operator.
    CudaPointHashGridSearcher2& operator=(
        const CudaPointHashGridSearcher2& other);

    //! Copy from the other instance.
    void set(const CudaPointHashGridSearcher2& other);

#if 0
    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;
#endif

    //! Returns builder fox CudaPointHashGridSearcher2.
    static Builder builder();

 private:
    float _gridSpacing = 1.0f;
    uint2 _resolution = make_uint2(1, 1);
    CudaArray1<float2> _points;
    CudaArray1<uint32_t> _keys;
    CudaArray1<uint32_t> _startIndexTable;
    CudaArray1<uint32_t> _endIndexTable;
    CudaArray1<uint32_t> _sortedIndices;
};

//! Shared pointer for the CudaPointHashGridSearcher2 type.
typedef std::shared_ptr<CudaPointHashGridSearcher2>
    CudaPointHashGridSearcher2Ptr;

//!
//! \brief Front-end to create CudaPointHashGridSearcher2 objects step by
//!        step.
//!
class CudaPointHashGridSearcher2::Builder final {
 public:
    //! Returns builder with resolution.
    Builder& withResolution(const Size2& resolution);

    //! Returns builder with grid spacing.
    Builder& withGridSpacing(float gridSpacing);

    //! Builds CudaPointHashGridSearcher2 instance.
    CudaPointHashGridSearcher2 build() const;

    //! Builds shared pointer of CudaPointHashGridSearcher2 instance.
    CudaPointHashGridSearcher2Ptr makeShared() const;

 private:
    Size2 _resolution{64, 64};
    float _gridSpacing = 1.0f;
};

}  // namespace experimental

}  // namespace jet

#include "detail/cuda_point_hash_grid_searcher2-inl.h"

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_POINT_HASH_GRID_SEARCHER2_H_

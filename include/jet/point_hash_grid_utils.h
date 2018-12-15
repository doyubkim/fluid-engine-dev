// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_HASH_GRID_UTILS_H_
#define INCLUDE_JET_POINT_HASH_GRID_UTILS_H_

#include <jet/matrix.h>

namespace jet {

//!
//! \brief Hash grid common utilities for N-D point searchers.
//!
template <size_t N>
class PointHashGridUtils {
 public:
    static size_t hashKey(const Vector<ssize_t, N> &index,
                          const Vector<ssize_t, N> &resolution);

    //!
    //! Returns the hash value for given N-D bucket index.
    //!
    //! \param[in]  bucketIndex The bucket index.
    //!
    //! \return     The hash key from bucket index.
    //!
    static size_t getHashKeyFromBucketIndex(
        const Vector<ssize_t, N> &bucketIndex,
        const Vector<ssize_t, N> &resolution);

    //!
    //! Gets the bucket index from a point.
    //!
    //! \param[in]  position The position of the point.
    //!
    //! \return     The bucket index.
    //!
    static Vector<ssize_t, N> getBucketIndex(const Vector<double, N> &position,
                                             double gridSpacing);

    static size_t getHashKeyFromPosition(const Vector<double, N> &position,
                                         double gridSpacing,
                                         const Vector<ssize_t, N> &resolution);

    static void getNearbyKeys(const Vector<double, N> &position,
                              double gridSpacing,
                              const Vector<ssize_t, N> &resolution,
                              size_t *nearbyKeys);
};

using PointHashGridUtils2 = PointHashGridUtils<2>;

using PointHashGridUtils3 = PointHashGridUtils<3>;

}  // namespace jet

#endif  // INCLUDE_JET_POINT_HASH_GRID_UTILS_H_

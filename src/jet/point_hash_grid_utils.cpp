// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/point_hash_grid_utils.h>

namespace jet {

namespace {

size_t _hashKey(const Vector<ssize_t, 2> &index,
                const Vector<ssize_t, 2> &resolution) {
    return static_cast<size_t>(index.y * resolution.x + index.x);
}

size_t _hashKey(const Vector<ssize_t, 3> &index,
                const Vector<ssize_t, 3> &resolution) {
    return static_cast<size_t>(
        (index.z * resolution.y + index.y) * resolution.x + index.x);
}
}

template <size_t N>
size_t PointHashGridUtils<N>::hashKey(const Vector<ssize_t, N> &index,
                                      const Vector<ssize_t, N> &resolution) {
    return _hashKey(index, resolution);
}

template <size_t N>
size_t PointHashGridUtils<N>::getHashKeyFromBucketIndex(
    const Vector<ssize_t, N> &bucketIndex,
    const Vector<ssize_t, N> &resolution) {
    Vector<ssize_t, N> wrappedIndex = bucketIndex;
    for (size_t i = 0; i < N; ++i) {
        wrappedIndex[i] = bucketIndex[i] % resolution[i];

        if (wrappedIndex[i] < 0) {
            wrappedIndex[i] += resolution[i];
        }
    }

    return hashKey(wrappedIndex, resolution);
}

template <size_t N>
Vector<ssize_t, N> PointHashGridUtils<N>::getBucketIndex(
    const Vector<double, N> &position, double gridSpacing) {
    Vector<ssize_t, N> bucketIndex;
    bucketIndex = floor(position / gridSpacing).template castTo<ssize_t>();
    return bucketIndex;
}

template <size_t N>
size_t PointHashGridUtils<N>::getHashKeyFromPosition(
    const Vector<double, N> &position, double gridSpacing,
    const Vector<ssize_t, N> &resolution) {
    Vector<ssize_t, N> bucketIndex = getBucketIndex(position, gridSpacing);

    return getHashKeyFromBucketIndex(bucketIndex, resolution);
}

template <size_t N>
void PointHashGridUtils<N>::getNearbyKeys(const Vector<double, N> &position,
                                          double gridSpacing,
                                          const Vector<ssize_t, N> &resolution,
                                          size_t *nearbyKeys) {
    constexpr int kNumKeys = 1 << N;

    Vector<ssize_t, N> originIndex = getBucketIndex(position, gridSpacing);
    Vector<ssize_t, N> nearbyBucketIndices[kNumKeys];

    for (int i = 0; i < kNumKeys; i++) {
        nearbyBucketIndices[i] = originIndex;
    }

    for (size_t axis = 0; axis < N; axis++) {
        int offset =
            (originIndex[axis] + 0.5) * gridSpacing <= position[axis] ? 1 : -1;
        for (int j = 0; j < kNumKeys; j++) {
            if (j & (1 << axis)) {
                nearbyBucketIndices[j][axis] += offset;
            }
        }
    }

    for (int i = 0; i < kNumKeys; i++) {
        nearbyKeys[i] =
            getHashKeyFromBucketIndex(nearbyBucketIndices[i], resolution);
    }
}

template class PointHashGridUtils<2>;

template class PointHashGridUtils<3>;

}  // namespace jet

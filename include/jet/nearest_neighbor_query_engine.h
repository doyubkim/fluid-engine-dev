// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_NEAREST_NEIGHBOR_QUERY_ENGINE_H_
#define INCLUDE_JET_NEAREST_NEIGHBOR_QUERY_ENGINE_H_

#include <jet/constants.h>
#include <jet/matrix.h>

#include <functional>

namespace jet {

//! N-D nearest neighbor query result.
template <typename T, size_t N>
struct NearestNeighborQueryResult {
    const T* item = nullptr;
    double distance = kMaxD;
};

//! 2-D nearest neighbor query result.
template <typename T>
using NearestNeighborQueryResult2 = NearestNeighborQueryResult<T, 2>;

//! 3-D nearest neighbor query result.
template <typename T>
using NearestNeighborQueryResult3 = NearestNeighborQueryResult<T, 3>;

//! N-D nearest neighbor distance measure function.
template <typename T, size_t N>
using NearestNeighborDistanceFunc =
    std::function<double(const T&, const Vector<double, N>&)>;

//! 2-D nearest neighbor distance measure function.
template <typename T>
using NearestNeighborDistanceFunc2 = NearestNeighborDistanceFunc<T, 2>;

//! 3-D nearest neighbor distance measure function.
template <typename T>
using NearestNeighborDistanceFunc3 = NearestNeighborDistanceFunc<T, 3>;

//! Abstract base class for N-D nearest neighbor query engine.
template <typename T, size_t N>
class NearestNeighborQueryEngine {
 public:
    virtual ~NearestNeighborQueryEngine() = default;

    //! Returns the nearest neighbor for given point and distance measure
    //! function.
    virtual NearestNeighborQueryResult<T, N> nearest(
        const Vector<double, N>& pt,
        const NearestNeighborDistanceFunc<T, N>& distanceFunc) const = 0;
};

//! Abstract base class for 2-D nearest neighbor query engine.
template <typename T>
using NearestNeighborQueryEngine2 = NearestNeighborQueryEngine<T, 2>;

//! Abstract base class for 3-D nearest neighbor query engine.
template <typename T>
using NearestNeighborQueryEngine3 = NearestNeighborQueryEngine<T, 3>;

}  // namespace jet

#endif  // INCLUDE_JET_NEAREST_NEIGHBOR_QUERY_ENGINE_H_

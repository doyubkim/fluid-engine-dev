// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_RAY_H_
#define INCLUDE_JET_RAY_H_

#include <jet/vector.h>

namespace jet {

//!
//! \brief      Class for ray.
//!
//! \tparam     T     The value type.
//! \tparam     N     The dimension.
//!
template <typename T, size_t N>
class Ray {
    static_assert(N != 2 && N != 3, "Not implemented.");
    static_assert(
        std::is_floating_point<T>::value,
        "Ray only can be instantiated with floating point types");
};

}  // namespace jet

#endif  // INCLUDE_JET_RAY_H_

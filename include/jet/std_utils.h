// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_STD_UTILS_H_
#define INCLUDE_JET_STD_UTILS_H_

#include <array>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: std::array Helpers

//! \brief Returns the first M elements from given \p src array.
template <typename T, size_t N, size_t M>
constexpr std::array<T, M> takeFirstM(std::array<T, N> src);

//! \brief Returns the last M elements from given \p src array.
template <typename T, size_t N, size_t M>
constexpr std::array<T, M> takeLastM(std::array<T, N> src);

}  // namespace jet

#include <jet/detail/std_utils-inl.h>

#endif  // INCLUDE_JET_STD_UTILS_H_

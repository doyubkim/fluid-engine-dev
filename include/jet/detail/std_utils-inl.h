// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_STD_UTILS_INL_H_
#define INCLUDE_JET_DETAIL_STD_UTILS_INL_H_

#include <jet/std_utils.h>

#include <utility>

namespace jet {

namespace internal {

template <size_t N>
struct StdUtilsSizeHint {};

template <typename T, size_t N, size_t M, size_t... I>
constexpr std::array<T, M> takeFirstM(const std::array<T, N> &a,
                                      StdUtilsSizeHint<M>,
                                      std::index_sequence<I...>) {
    return std::array<T, M>{{a[I]...}};
}

template <typename T, size_t N, size_t M, size_t... I>
constexpr std::array<T, M> takeLastM(const std::array<T, N> &a,
                                     StdUtilsSizeHint<M>,
                                     std::index_sequence<I...>) {
    return std::array<T, M>{{a[I + N - M]...}};
}

}  // namespace internal

template <typename T, size_t N, size_t M>
constexpr std::array<T, M> takeFirstM(std::array<T, N> src) {
    static_assert(M < N, "Return array size should be smaller than the input.");
    return internal::takeFirstM(src, internal::StdUtilsSizeHint<M>{},
                                std::make_index_sequence<M>{});
}

template <typename T, size_t N, size_t M>
constexpr std::array<T, M> takeLastM(std::array<T, N> src) {
    static_assert(M < N, "Return array size should be smaller than the input.");
    return internal::takeLastM(src, internal::StdUtilsSizeHint<M>{},
                               std::make_index_sequence<M>{});
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_STD_UTILS_INL_H_

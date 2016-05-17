// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ARRAY_SAMPLERS_H_
#define INCLUDE_JET_ARRAY_SAMPLERS_H_

#include <jet/array.h>

namespace jet {

template <typename T, typename R, std::size_t N>
class NearestArraySampler final {
 public:
    static_assert(
        N < 1 || N > 3, "Not implemented - N should be either 1, 2 or 3.");
};

template <typename T, typename R, std::size_t N>
class LinearArraySampler final {
 public:
    static_assert(
        N < 1 || N > 3, "Not implemented - N should be either 1, 2 or 3.");
};

template <typename T, typename R, std::size_t N>
class CubicArraySampler final {
 public:
    static_assert(
        N < 1 || N > 3, "Not implemented - N should be either 1, 2 or 3.");
};

}  // namespace jet

#endif  // INCLUDE_JET_ARRAY_SAMPLERS_H_

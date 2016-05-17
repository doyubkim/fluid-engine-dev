// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_LEVEL_SET_UTILS_H_
#define INCLUDE_JET_LEVEL_SET_UTILS_H_

#include <jet/macros.h>

namespace jet {

template <typename T>
T isInsideSdf(T phi);

template <typename T>
T smearedHeavisideSdf(T phi);

template <typename T>
T smearedDeltaSdf(T phi);

template <typename T>
T fractionInsideSdf(T phi0, T phi1);

}  // namespace jet

#include "detail/level_set_utils-inl.h"

#endif  // INCLUDE_JET_LEVEL_SET_UTILS_H_

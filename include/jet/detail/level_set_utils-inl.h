// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_DETAIL_LEVEL_SET_UTILS_INL_H_
#define INCLUDE_JET_DETAIL_LEVEL_SET_UTILS_INL_H_

// Function fractionInside is from
// http://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/

#include <jet/constants.h>
#include <cmath>

namespace jet {

template <typename T>
bool isInsideSdf(T phi) {
    return phi < 0;
}

template <typename T>
inline T smearedHeavisideSdf(T phi) {
    if (phi > 1.5) {
        return 1;
    } else {
        if (phi < -1.5) {
            return 0;
        } else {
            return 0.5f
                + phi / 3.0
                + 0.5f * invPi<T>() * std::sin(pi<T>() * phi / 1.5);
        }
    }
}

template <typename T>
inline T smearedDeltaSdf(T phi) {
    if (std::fabs(phi) > 1.5) {
        return 0;
    } else {
        return 1.0 / 3.0
            + 1.0/3.0 * std::cos(pi<T>() * phi / 1.5);
    }
}

template <typename T>
T fractionInsideSdf(T phi0, T phi1) {
    if (isInsideSdf(phi0) && isInsideSdf(phi1)) {
        return 1;
    } else if (isInsideSdf(phi0) && !isInsideSdf(phi1)) {
        return phi0 / (phi0 - phi1);
    } else if (!isInsideSdf(phi0) && isInsideSdf(phi1)) {
        return phi1 / (phi1 - phi0);
    } else {
        return 0;
    }
}

template <typename T>
T distanceToZeroLevelSet(T phi0, T phi1) {
    if (std::fabs(phi0) + std::fabs(phi1) > kEpsilonD) {
        return std::fabs(phi0) / (std::fabs(phi0) + std::fabs(phi1));
    } else {
        return static_cast<T>(0.5);
    }
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_LEVEL_SET_UTILS_INL_H_

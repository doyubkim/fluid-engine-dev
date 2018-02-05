// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// Function fractionInside is originally from Christopher Batty's code:
//
// http://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/
//
// and
//
// https://github.com/christopherbatty/Fluid3D

#ifndef INCLUDE_JET_DETAIL_LEVEL_SET_UTILS_INL_H_
#define INCLUDE_JET_DETAIL_LEVEL_SET_UTILS_INL_H_

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
            return 0.5f + phi / 3.0 +
                   0.5f * invPi<T>() * std::sin(pi<T>() * phi / 1.5);
        }
    }
}

template <typename T>
inline T smearedDeltaSdf(T phi) {
    if (std::fabs(phi) > 1.5) {
        return 0;
    } else {
        return 1.0 / 3.0 + 1.0 / 3.0 * std::cos(pi<T>() * phi / 1.5);
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
void cycleArray(T* arr, int size) {
    T t = arr[0];
    for (int i = 0; i < size - 1; ++i) arr[i] = arr[i + 1];
    arr[size - 1] = t;
}

template <typename T>
T fractionInside(T phiBottomLeft, T phiBottomRight, T phiTopLeft,
                 T phiTopRight) {
    int inside_count = (phiBottomLeft < 0 ? 1 : 0) + (phiTopLeft < 0 ? 1 : 0) +
                       (phiBottomRight < 0 ? 1 : 0) + (phiTopRight < 0 ? 1 : 0);
    T list[] = {phiBottomLeft, phiBottomRight, phiTopRight, phiTopLeft};

    if (inside_count == 4) {
        return 1;
    } else if (inside_count == 3) {
        // rotate until the positive value is in the first position
        while (list[0] < 0) {
            cycleArray(list, 4);
        }

        // Work out the area of the exterior triangle
        T side0 = 1 - fractionInsideSdf(list[0], list[3]);
        T side1 = 1 - fractionInsideSdf(list[0], list[1]);
        return 1 - 0.5f * side0 * side1;
    } else if (inside_count == 2) {
        // rotate until a negative value is in the first position, and the next
        // negative is in either slot 1 or 2.
        while (list[0] >= 0 || !(list[1] < 0 || list[2] < 0)) {
            cycleArray(list, 4);
        }

        if (list[1] < 0) {  // the matching signs are adjacent
            T side_left = fractionInsideSdf(list[0], list[3]);
            T side_right = fractionInsideSdf(list[1], list[2]);
            return 0.5f * (side_left + side_right);
        } else {  // matching signs are diagonally opposite
            // determine the centre point's sign to disambiguate this case
            T middle_point = 0.25f * (list[0] + list[1] + list[2] + list[3]);
            if (middle_point < 0) {
                T area = 0;

                // first triangle (top left)
                T side1 = 1 - fractionInsideSdf(list[0], list[3]);
                T side3 = 1 - fractionInsideSdf(list[2], list[3]);

                area += 0.5f * side1 * side3;

                // second triangle (top right)
                T side2 = 1 - fractionInsideSdf(list[2], list[1]);
                T side0 = 1 - fractionInsideSdf(list[0], list[1]);
                area += 0.5f * side0 * side2;

                return 1 - area;
            } else {
                T area = 0;

                // first triangle (bottom left)
                T side0 = fractionInsideSdf(list[0], list[1]);
                T side1 = fractionInsideSdf(list[0], list[3]);
                area += 0.5f * side0 * side1;

                // second triangle (top right)
                T side2 = fractionInsideSdf(list[2], list[1]);
                T side3 = fractionInsideSdf(list[2], list[3]);
                area += 0.5f * side2 * side3;
                return area;
            }
        }
    } else if (inside_count == 1) {
        // rotate until the negative value is in the first position
        while (list[0] >= 0) {
            cycleArray(list, 4);
        }

        // Work out the area of the interior triangle, and subtract from 1.
        T side0 = fractionInsideSdf(list[0], list[3]);
        T side1 = fractionInsideSdf(list[0], list[1]);
        return 0.5f * side0 * side1;
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

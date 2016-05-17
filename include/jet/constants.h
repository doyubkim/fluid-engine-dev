// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CONSTANTS_H_
#define INCLUDE_JET_CONSTANTS_H_

#include <jet/macros.h>
#include <cmath>
#include <limits>

namespace jet {

// Zero
const size_t kZeroSize = 0;
const ssize_t kZeroSSize = 0;

template <typename T>
inline T zero() {
    return 0;
}

template <>
inline float zero<float>() {
    return 0.f;
}

template <>
inline double zero<double>() {
    return 0.0;
}

// One
const size_t kOneSize = 1;
const ssize_t kOneSSize = 1;

template <typename T>
inline T one() {
    return 1;
}

template <>
inline float one<float>() {
    return 1.f;
}

template <>
inline double one<double>() {
    return 1.0;
}

// Epsilon
const float kEpsilonF = std::numeric_limits<float>::epsilon();
const double kEpsilonD = std::numeric_limits<double>::epsilon();

// Max
const size_t kMaxSize = std::numeric_limits<size_t>::max();
const ssize_t kMaxSSize = std::numeric_limits<ssize_t>::max();
const float kMaxF = std::numeric_limits<float>::max();
const double kMaxD = std::numeric_limits<double>::max();

// Pi
const float kPiF = 3.14159265358979323846264338327950288f;
const double kPiD = 3.14159265358979323846264338327950288;

template <typename T>
inline T pi() {
    static const T result = static_cast<T>(kPiD);
    return result;
}

template <>
inline float pi<float>() {
    return kPiF;
}

template <>
inline double pi<double>() {
    return kPiD;
}

// Pi/2
const float kHalfPiF = 1.57079632679489661923132169163975144f;
const double kHalfPiD = 1.57079632679489661923132169163975144;

template <typename T>
inline T halfPi() {
    static const T result = static_cast<T>(kHalfPiD);
    return result;
}

template <>
inline float halfPi<float>() {
    return kHalfPiF;
}

template <>
inline double halfPi<double>() {
    return kHalfPiD;
}

// Pi/4
const float kQuaterPiF = 0.785398163397448309615660845819875721f;
const double kQuaterPiD = 0.785398163397448309615660845819875721;

template <typename T>
inline T quaterPi() {
    static const T result = static_cast<T>(kQuaterPiD);
    return result;
}

template <>
inline float quaterPi<float>() {
    return kQuaterPiF;
}

template <>
inline double quaterPi<double>() {
    return kQuaterPiD;
}

// 2*Pi
const float kTwoPiF = static_cast<float>(2.0 * kPiD);
const double kTwoPiD = 2.0 * kPiD;

template <typename T>
inline T twoPi() {
    static const T result = static_cast<T>(kTwoPiD);
    return result;
}

template <>
inline float twoPi<float>() {
    return kTwoPiF;
}

template <>
inline double twoPi<double>() {
    return kTwoPiD;
}

// 1/Pi
const float kInvPiF = static_cast<float>(1.0 / kPiD);
const double kInvPiD = 1.0 / kPiD;

template <typename T>
inline T invPi() {
    static const T result = static_cast<T>(kInvPiD);
    return result;
}

template <>
inline float invPi<float>() {
    return kInvPiF;
}

template <>
inline double invPi<double>() {
    return kInvPiD;
}

// 1/2*Pi
const float kInvTwoPiF = static_cast<float>(0.5 / kPiD);
const double kInvTwoPiD = 0.5 / kPiD;

template <typename T>
inline T invTwoPi() {
    static const T result = static_cast<T>(kInvTwoPiD);
    return result;
}

template <>
inline float invTwoPi<float>() {
    return kInvTwoPiF;
}

template <>
inline double invTwoPi<double>() {
    return kInvTwoPiD;
}

// Physics

// Gravity
const double kGravity = -9.8;

// Speed of sound in water at 20 degrees celcius.
const double kSpeedOfSoundInWater = 1482.0;

// Common enums
const int kDirectionLeft = 1 << 0;
const int kDirectionRight = 1 << 1;
const int kDirectionDown = 1 << 2;
const int kDirectionUp = 1 << 3;
const int kDirectionBack = 1 << 4;
const int kDirectionFront = 1 << 5;
const int kDirectionAll
    = kDirectionLeft | kDirectionRight
    | kDirectionDown | kDirectionUp
    | kDirectionBack | kDirectionFront;

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANTS_H_

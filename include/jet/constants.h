// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CONSTANTS_H_
#define INCLUDE_JET_CONSTANTS_H_

#include <jet/macros.h>
#include <cmath>
#include <limits>

namespace jet {

// MARK: Zero

//! Zero size_t.
constexpr size_t kZeroSize = 0;

//! Zero ssize_t.
constexpr ssize_t kZeroSSize = 0;

//! Zero for type T.
template <typename T>
inline T zero() {
    return 0;
}

//! Zero for float.
template <>
inline float zero<float>() {
    return 0.f;
}

//! Zero for double.
template <>
inline double zero<double>() {
    return 0.0;
}

// MARK: One

//! One size_t.
constexpr size_t kOneSize = 1;

//! One ssize_t.
constexpr ssize_t kOneSSize = 1;

//! One for type T.
template <typename T>
constexpr T one() {
    return 1;
}

//! One for float.
template <>
constexpr float one<float>() {
    return 1.f;
}

//! One for double.
template <>
constexpr double one<double>() {
    return 1.0;
}

// MARK: Epsilon

//! Float-type epsilon.
constexpr float kEpsilonF = std::numeric_limits<float>::epsilon();

//! Double-type epsilon.
constexpr double kEpsilonD = std::numeric_limits<double>::epsilon();


// MARK: Max

//! Max size_t.
constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();

//! Max ssize_t.
constexpr ssize_t kMaxSSize = std::numeric_limits<ssize_t>::max();

//! Max float.
constexpr float kMaxF = std::numeric_limits<float>::max();

//! Max double.
constexpr double kMaxD = std::numeric_limits<double>::max();


// MARK: Pi

//! Float-type pi.
constexpr float kPiF = 3.14159265358979323846264338327950288f;

//! Double-type pi.
constexpr double kPiD = 3.14159265358979323846264338327950288;

//! Pi for type T.
template <typename T>
constexpr T pi() {
    return static_cast<T>(kPiD);
}

//! Pi for float.
template <>
constexpr float pi<float>() {
    return kPiF;
}

//! Pi for double.
template <>
constexpr double pi<double>() {
    return kPiD;
}


// MARK: Pi/2

//! Float-type pi/2.
constexpr float kHalfPiF = 1.57079632679489661923132169163975144f;

//! Double-type pi/2.
constexpr double kHalfPiD = 1.57079632679489661923132169163975144;

//! Pi/2 for type T.
template <typename T>
constexpr T halfPi() {
    return static_cast<T>(kHalfPiD);
}

//! Pi/2 for float.
template <>
constexpr float halfPi<float>() {
    return kHalfPiF;
}

//! Pi/2 for double.
template <>
constexpr double halfPi<double>() {
    return kHalfPiD;
}


// MARK: Pi/4

//! Float-type pi/4.
constexpr float kQuaterPiF = 0.785398163397448309615660845819875721f;

//! Double-type pi/4.
constexpr double kQuaterPiD = 0.785398163397448309615660845819875721;

//! Pi/4 for type T.
template <typename T>
constexpr T quaterPi() {
    return static_cast<T>(kQuaterPiD);
}

//! Pi/2 for float.
template <>
constexpr float quaterPi<float>() {
    return kQuaterPiF;
}

//! Pi/2 for double.
template <>
constexpr double quaterPi<double>() {
    return kQuaterPiD;
}

// MARK: 2*Pi

//! Float-type 2*pi.
constexpr float kTwoPiF = static_cast<float>(2.0 * kPiD);

//! Double-type 2*pi.
constexpr double kTwoPiD = 2.0 * kPiD;

//! 2*pi for type T.
template <typename T>
constexpr T twoPi() {
    return static_cast<T>(kTwoPiD);
}

//! 2*pi for float.
template <>
constexpr float twoPi<float>() {
    return kTwoPiF;
}

//! 2*pi for double.
template <>
constexpr double twoPi<double>() {
    return kTwoPiD;
}

// MARK: 1/Pi

//! Float-type 1/pi.
constexpr float kInvPiF = static_cast<float>(1.0 / kPiD);

//! Double-type 1/pi.
constexpr double kInvPiD = 1.0 / kPiD;

//! 1/pi for type T.
template <typename T>
constexpr T invPi() {
    return static_cast<T>(kInvPiD);
}

//! 1/pi for float.
template <>
constexpr float invPi<float>() {
    return kInvPiF;
}

//! 1/pi for double.
template <>
constexpr double invPi<double>() {
    return kInvPiD;
}


// MARK: 1/2*Pi

//! Float-type 1/2*pi.
constexpr float kInvTwoPiF = static_cast<float>(0.5 / kPiD);

//! Double-type 1/2*pi.
constexpr double kInvTwoPiD = 0.5 / kPiD;

//! 1/2*pi for type T.
template <typename T>
constexpr T invTwoPi() {
    return static_cast<T>(kInvTwoPiD);
}

//! 1/2*pi for float.
template <>
constexpr float invTwoPi<float>() {
    return kInvTwoPiF;
}

//! 1/2*pi for double.
template <>
constexpr double invTwoPi<double>() {
    return kInvTwoPiD;
}


// MARK: Physics

//! Gravity.
constexpr double kGravity = -9.8;

//! Speed of sound in water at 20 degrees celcius.
constexpr double kSpeedOfSoundInWater = 1482.0;


// MARK: Common enums

//! No direction.
constexpr int kDirectionNone = 0;

//! Left direction.
constexpr int kDirectionLeft = 1 << 0;

//! RIght direction.
constexpr int kDirectionRight = 1 << 1;

//! Down direction.
constexpr int kDirectionDown = 1 << 2;

//! Up direction.
constexpr int kDirectionUp = 1 << 3;

//! Back direction.
constexpr int kDirectionBack = 1 << 4;

//! Front direction.
constexpr int kDirectionFront = 1 << 5;

//! All direction.
constexpr int kDirectionAll
    = kDirectionLeft | kDirectionRight
    | kDirectionDown | kDirectionUp
    | kDirectionBack | kDirectionFront;

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANTS_H_

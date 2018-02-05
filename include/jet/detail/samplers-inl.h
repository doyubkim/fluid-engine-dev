// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// Copyright (c) 1998-2014, Matt Pharr and Greg Humphreys.
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef INCLUDE_JET_DETAIL_SAMPLERS_INL_H_
#define INCLUDE_JET_DETAIL_SAMPLERS_INL_H_

#include <jet/constants.h>
#include <jet/math_utils.h>
#include <algorithm>

namespace jet {

template <typename T>
inline Vector3<T> uniformSampleCone(
    T u1, T u2, const Vector3<T>& axis, T angle) {
    T cosAngle_2 = std::cos(angle / 2);
    T y = 1 - (1 - cosAngle_2) * u1;
    T r = std::sqrt(std::max<T>(0, 1 - y * y));
    T phi = twoPi<T>() * u2;
    T x = r * std::cos(phi);
    T z = r * std::sin(phi);
    auto ts = axis.tangential();
    return std::get<0>(ts) * x + axis * y + std::get<1>(ts) * z;
}

template <typename T>
inline Vector3<T> uniformSampleHemisphere(
    T u1, T u2, const Vector3<T>& normal) {
    T y = u1;
    T r = std::sqrt(std::max<T>(0, 1 - y*y));
    T phi = twoPi<T>() * u2;
    T x = r * std::cos(phi);
    T z = r * std::sin(phi);
    auto ts = normal.tangential();
    return std::get<0>(ts) * x + normal * y + std::get<1>(ts) * z;
}

template <typename T>
inline Vector3<T> cosineWeightedSampleHemisphere(
    T u1, T u2, const Vector3<T> &normal) {
    T phi = twoPi<T>()*u1;
    T y = std::sqrt(u2);
    T theta = std::acos(y);
    T x = std::cos(phi) * std::sin(theta);
    T z = std::sin(phi) * std::sin(theta);
    Vector3<T> t = tangential(normal);
    auto ts = normal.tangential();
    return std::get<0>(ts) * x + normal * y + std::get<1>(ts) * z;
}

template <typename T>
inline Vector3<T> uniformSampleSphere(T u1, T u2) {
    T y = 1 - 2 * u1;
    T r = std::sqrt(std::max<T>(0, 1 - y * y));
    T phi = twoPi<T>() * u2;
    T x = r * std::cos(phi);
    T z = r * std::sin(phi);
    return Vector3<T>(x, y, z);
}

template <typename T>
inline Vector2<T> uniformSampleDisk(T u1, T u2) {
    T r = std::sqrt(u1);
    T theta = twoPi<T>() * u2;

    return Vector2<T>(r * std::cos(theta), r * std::sin(theta));
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SAMPLERS_INL_H_

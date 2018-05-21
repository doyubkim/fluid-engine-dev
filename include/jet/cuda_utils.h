// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_UTILS_H_
#define INCLUDE_JET_CUDA_UTILS_H_

#include <jet/macros.h>

#include <cuda_runtime.h>

namespace jet {

inline JET_CUDA_HOST void checkResult(cudaError_t err) {
    if (err != cudaSuccess) {
        std::string msg("CUDA error: ");
        msg += cudaGetErrorString(err);
        throw std::runtime_error(msg.c_str());
    }
}

inline JET_CUDA_HOST_DEVICE float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline JET_CUDA_HOST_DEVICE float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline JET_CUDA_HOST_DEVICE float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline JET_CUDA_HOST_DEVICE float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline JET_CUDA_HOST_DEVICE float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline JET_CUDA_HOST_DEVICE float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline JET_CUDA_HOST_DEVICE float2 operator*(float a, float2 b) {
    return make_float2(a * b.x, a * b.y);
}

inline JET_CUDA_HOST_DEVICE float3 operator*(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline JET_CUDA_HOST_DEVICE float4 operator*(float a, float4 b) {
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

inline JET_CUDA_HOST_DEVICE float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}

inline JET_CUDA_HOST_DEVICE float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline JET_CUDA_HOST_DEVICE float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline JET_CUDA_HOST_DEVICE float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

inline JET_CUDA_HOST_DEVICE float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline JET_CUDA_HOST_DEVICE float4 operator*(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline JET_CUDA_HOST_DEVICE float2 operator/(float a, float2 b) {
    return make_float2(a / b.x, a / b.y);
}

inline JET_CUDA_HOST_DEVICE float3 operator/(float a, float3 b) {
    return make_float3(a / b.x, a / b.y, a / b.z);
}

inline JET_CUDA_HOST_DEVICE float4 operator/(float a, float4 b) {
    return make_float4(a / b.x, a / b.y, a / b.z, a / b.w);
}

inline JET_CUDA_HOST_DEVICE float2 operator/(float2 a, float b) {
    return make_float2(a.x / b, a.y / b);
}

inline JET_CUDA_HOST_DEVICE float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline JET_CUDA_HOST_DEVICE float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

inline JET_CUDA_HOST_DEVICE float2 operator/(float2 a, float2 b) {
    return make_float2(a.x / b.x, a.y / b.y);
}

inline JET_CUDA_HOST_DEVICE float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline JET_CUDA_HOST_DEVICE float4 operator/(float4 a, float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline JET_CUDA_HOST_DEVICE void operator+=(float2& a, float b) {
    a.x += b;
    a.y += b;
}

inline JET_CUDA_HOST_DEVICE void operator+=(float3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

inline JET_CUDA_HOST_DEVICE void operator+=(float4& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline JET_CUDA_HOST_DEVICE void operator+=(float2& a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}

inline JET_CUDA_HOST_DEVICE void operator+=(float3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline JET_CUDA_HOST_DEVICE void operator+=(float4& a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline JET_CUDA_HOST_DEVICE void operator-=(float2& a, float b) {
    a.x -= b;
    a.y -= b;
}

inline JET_CUDA_HOST_DEVICE void operator-=(float3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline JET_CUDA_HOST_DEVICE void operator-=(float4& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline JET_CUDA_HOST_DEVICE void operator-=(float2& a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
}

inline JET_CUDA_HOST_DEVICE void operator-=(float3& a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline JET_CUDA_HOST_DEVICE void operator-=(float4& a, float4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline JET_CUDA_HOST_DEVICE void operator*=(float2& a, float b) {
    a.x *= b;
    a.y *= b;
}

inline JET_CUDA_HOST_DEVICE void operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline JET_CUDA_HOST_DEVICE void operator*=(float4& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline JET_CUDA_HOST_DEVICE void operator*=(float2& a, float2 b) {
    a.x *= b.x;
    a.y *= b.y;
}

inline JET_CUDA_HOST_DEVICE void operator*=(float3& a, float4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

inline JET_CUDA_HOST_DEVICE void operator*=(float4& a, float4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

inline JET_CUDA_HOST_DEVICE void operator/=(float2& a, float b) {
    a.x /= b;
    a.y /= b;
}

inline JET_CUDA_HOST_DEVICE void operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

inline JET_CUDA_HOST_DEVICE void operator/=(float4& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

inline JET_CUDA_HOST_DEVICE void operator/=(float2& a, float4 b) {
    a.x /= b.x;
    a.y /= b.y;
}

inline JET_CUDA_HOST_DEVICE void operator/=(float3& a, float4 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

inline JET_CUDA_HOST_DEVICE void operator/=(float4& a, float4 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

inline JET_CUDA_HOST_DEVICE bool operator==(float2 a, float2 b) {
    return a.x == b.x && a.y == b.y;
}

inline JET_CUDA_HOST_DEVICE bool operator==(float3 a, float3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline JET_CUDA_HOST_DEVICE bool operator==(float4 a, float4 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline JET_CUDA_HOST_DEVICE float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

inline JET_CUDA_HOST_DEVICE float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline JET_CUDA_HOST_DEVICE float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline JET_CUDA_HOST_DEVICE float lengthSquared(float2 v) { return dot(v, v); }

inline JET_CUDA_HOST_DEVICE float lengthSquared(float3 v) { return dot(v, v); }

inline JET_CUDA_HOST_DEVICE float lengthSquared(float4 v) { return dot(v, v); }

inline JET_CUDA_HOST_DEVICE float length(float2 v) {
    return sqrtf(lengthSquared(v));
}

inline JET_CUDA_HOST_DEVICE float length(float3 v) {
    return sqrtf(lengthSquared(v));
}

inline JET_CUDA_HOST_DEVICE float length(float4 v) {
    return sqrtf(lengthSquared(v));
}

// MARK: Converters

template <typename VectorType>
inline JET_CUDA_HOST_DEVICE float2 toFloat2(const VectorType& vec) {
    return make_float2(vec.x, vec.y);
}

template <typename VectorType>
inline JET_CUDA_HOST_DEVICE float3 toFloat3(const VectorType& vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

template <typename VectorType>
inline JET_CUDA_HOST_DEVICE float4 toFloat4(const VectorType& vec, float w) {
    return make_float4(vec.x, vec.y, vec.z, w);
}

template <typename VectorType>
inline JET_CUDA_HOST_DEVICE float4 toFloat4(const VectorType& vec) {
    return make_float4(vec.x, vec.y, vec.z, vec.w);
}

template <typename SizeType>
inline JET_CUDA_HOST_DEVICE int2 toInt2(const SizeType& size) {
    return make_int2(static_cast<int>(size.x), static_cast<int>(size.y));
}

template <typename SizeType>
inline JET_CUDA_HOST_DEVICE int3 toInt3(const SizeType& size) {
    return make_int3(static_cast<int>(size.x), static_cast<int>(size.y),
                     static_cast<int>(size.z));
}

template <typename SizeType>
inline JET_CUDA_HOST_DEVICE uint2 toUInt2(const SizeType& size) {
    return make_uint2(static_cast<uint32_t>(size.x),
                      static_cast<uint32_t>(size.y));
}

template <typename SizeType>
inline JET_CUDA_HOST_DEVICE uint3 toUInt3(const SizeType& size) {
    return make_uint3(static_cast<uint32_t>(size.x),
                      static_cast<uint32_t>(size.y),
                      static_cast<uint32_t>(size.z));
}

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_UTILS_H_

#endif  // JET_USE_CUDA

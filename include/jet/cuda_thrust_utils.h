// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_THRUST_UTILS_H_
#define INCLUDE_JET_CUDA_THRUST_UTILS_H_

#ifdef JET_USE_CUDA

#include <jet/matrix.h>

#include <thrust/memory.h>

#include <array>

namespace thrust {

template <typename T>
class host_device_ptr
    : public pointer<T, host_system_tag, T&, host_device_ptr<T>> {
    using super_t = pointer<T, host_system_tag, T&, host_device_ptr<T>>;

 public:
    __host__ __device__ host_device_ptr() : super_t() {}

    template <typename U>
    __host__ __device__ explicit host_device_ptr(U* ptr) : super_t(ptr) {}

    template <typename U>
    __host__ __device__ host_device_ptr(const host_device_ptr<U>& other)
        : super_t(other) {}

    template <typename U>
    __host__ __device__ host_device_ptr& operator=(
        const host_device_ptr<U>& other) {
        super_t::operator=(other);
        return *this;
    }
};

template <typename T, size_t N>
class array {
public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    __host__ __device__ array() {
        fill(T{});
    }

    template <typename... Args>
    __host__ __device__ array(const_reference first, Args... rest) {
        static_assert(
            sizeof...(Args) == N - 1,
            "Number of arguments should be equal to the size of the array.");
        set_at(0, first, rest...);
    }

    __host__ array(const std::array<T, N>& other) {
        for (size_t i = 0; i < N; ++i) {
            _elements[i] = other[i];
        }
    }

    __host__ array(const jet::Vector<T, N>& other) {
        for (size_t i = 0; i < N; ++i) {
            _elements[i] = other[i];
        }
    }

    __host__ __device__ array(const array& other) {
        for (size_t i = 0; i < N; ++i) {
            _elements[i] = other[i];
        }
    }

    __host__ __device__ void fill(const_reference val) {
        for (size_t i = 0; i < N; ++i) {
            _elements[i] = val;
        }
    }

    __host__ __device__ reference operator[](size_t i) {
        return _elements[i];
    }

    __host__ __device__ const_reference operator[](size_t i) const {
        return _elements[i];
    }

private:
    T _elements[N];

    template <typename... Args>
    __host__ __device__ void set_at(size_t i, const_reference first,
                                    Args... rest) {
        _elements[i] = first;
        set_at(i + 1, rest...);
    }

    template <typename... Args>
    __host__ __device__ void set_at(size_t i, const_reference first) {
        _elements[i] = first;
    }
};

}  // namespace thrust

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_THRUST_UTILS_H_

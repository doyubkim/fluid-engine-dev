// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_STD_VECTOR_H_
#define INCLUDE_JET_CUDA_STD_VECTOR_H_

#ifdef JET_USE_CUDA

#include <jet/cuda_algorithms.h>

#include <vector>

namespace jet {

template <typename T>
class CudaStdVector final {
 public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    class Reference {
     public:
        JET_CUDA_HOST_DEVICE Reference(pointer p) : _ptr(p) {}

        JET_CUDA_HOST_DEVICE Reference(const Reference& other)
            : _ptr(other._ptr) {}

#ifdef __CUDA_ARCH__
        __device__ Reference& operator=(const value_type& val) {
            *ptr = val;
            return *this;
        }

        __device__ operator value_type() const { return *ptr; }
#else
        JET_CUDA_HOST Reference& operator=(const value_type& val) {
            cudaCopyHostToDevice(&val, 1, _ptr);
            return *this;
        }

        JET_CUDA_HOST operator value_type() const {
            std::remove_const_t<value_type> tmp{};
            cudaCopyDeviceToHost(_ptr, 1, &tmp);
            return tmp;
        }
#endif
     private:
        pointer _ptr;
    };

    CudaStdVector();

    CudaStdVector(size_t n, const value_type& initVal = value_type{});

    template <typename A>
    CudaStdVector(const std::vector<T, A>& other);

    CudaStdVector(const CudaStdVector& other);

    CudaStdVector(CudaStdVector&& other);

    ~CudaStdVector();

    pointer data();

    const_pointer data() const;

    size_t size() const;

#ifdef __CUDA_ARCH__
    __device__ reference at(size_t i);

    __device__ const_reference at(size_t i) const;
#else
    JET_CUDA_HOST Reference at(size_t i);

    JET_CUDA_HOST T at(size_t i) const;
#endif

    void clear();

    void fill(const value_type& val);

    void resize(size_t n, const value_type& initVal = value_type{});

    void resizeUninitialized(size_t n);

    void swap(CudaStdVector& other);

    void push_back(const value_type& val);

    void append(const value_type& val);

    void append(const CudaStdVector& other);

    template <typename A>
    void copyFrom(const std::vector<T, A>& other);

    void copyFrom(const CudaStdVector& other);

    template <typename A>
    void copyTo(std::vector<T, A>& other);

    template <typename A>
    CudaStdVector& operator=(const std::vector<T, A>& other);

    CudaStdVector& operator=(const CudaStdVector& other);

    CudaStdVector& operator=(CudaStdVector&& other);

#ifdef __CUDA_ARCH__
    reference operator[](size_t i);

    const_reference operator[](size_t i) const;
#else
    Reference operator[](size_t i);

    T operator[](size_t i) const;
#endif

 private:
    pointer _ptr = nullptr;
    size_t _size = 0;
};

}  // namespace jet

#include <jet/detail/cuda_std_vector-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_STD_VECTOR_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY_INL_H_

#ifdef JET_USE_CUDA

#include <jet/_cuda_array.h>
#include <jet/_cuda_array_view.h>

namespace jet {

namespace internal {

template <typename T1, typename T2, size_t N, size_t I>
struct CudaBlockCopyHelper {
    template <typename... RemainingIndices>
    JET_CUDA_HOST_DEVICE static void call(const T1* src,
                                          const thrust::array<size_t, N>& size,
                                          T2* dst,
                                          RemainingIndices... indices) {
        for (size_t i = 0; i < size[I - 1]; ++i) {
            CudaBlockCopyHelper<T1, T2, N, I - 1>::call(src, size, dst, i,
                                                        indices...);
        }
    }
};

template <typename T1, typename T2, size_t N>
struct CudaBlockCopyHelper<T1, T2, N, 1> {
    template <typename... RemainingIndices>
    JET_CUDA_HOST_DEVICE static void call(const T1* src,
                                          const thrust::array<size_t, N>& size,
                                          T2* dst,
                                          RemainingIndices... indices) {
        for (size_t i = 0; i < size[0]; ++i) {
            dst[index(size, i, indices...)] = src[index(size, i, indices...)];
        }
    }

    template <typename... Args>
    JET_CUDA_HOST_DEVICE static size_t index(
        const thrust::array<size_t, N>& size, size_t i, Args... args) {
        return i + size[0] * _index(size, 1, args...);
    }

    template <typename... Args>
    JET_CUDA_HOST_DEVICE static size_t _index(
        const thrust::array<size_t, N>& size, size_t d, size_t i,
        Args... args) {
        return i + size[d] * _index(size, d + 1, args...);
    }

    JET_CUDA_HOST_DEVICE static size_t _index(
        const thrust::array<size_t, N>& size, size_t, size_t i) {
        return i;
    }
};

template <typename T1, typename T2, size_t N>
struct CudaBlockCopy {
    const T1* src;
    T2* dst;
    thrust::array<size_t, N> size;

    CudaBlockCopy(const T1* src_, thrust::array<size_t, N> size_, T2* dst_)
        : src(src_), dst(dst_) {}

    void operator()(size_t i) {
        CudaBlockCopyHelper<T1, T2, N, N - 1>::call(src, size, dst, i);
    }
};

template <typename T1, typename T2>
struct CudaBlockCopy<T1, T2, 1> {
    const T1* src;
    T2* dst;
    thrust::array<size_t, 1> size;

    CudaBlockCopy(const T1* src_, thrust::array<size_t, 1> size_, T2* dst_)
        : src(src_), dst(dst_) {}

    void operator()(size_t i) { dst[i] = src[i]; }
};

}  // namespace internal

template <typename T>
template <typename T1, typename T2, size_t N, typename D1, typename D2>
void CudaDevice<T>::copy(const ArrayBase<T1, N, CudaDevice<T1>, D1>& src,
                         const Vector<size_t, N>& size,
                         ArrayBase<T2, N, CudaDevice<T2>, D2>& dst) {
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(size[N - 1]),
        internal::CudaBlockCopy<T1, T2, N>(
            src.data(), thrust::array<size_t, N>(size), dst.data()));
}

}  // namespace jet

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY_INL_H_

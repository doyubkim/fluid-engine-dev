// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_ARRAY_H_
#define INCLUDE_JET_CUDA_ARRAY_H_

#ifdef JET_USE_CUDA

#include <jet/array.h>
#include <jet/cuda_std_array.h>
#include <jet/cuda_std_vector.h>

namespace jet {

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaArrayBase

template <typename T, size_t N, typename DerivedArray>
class CudaArrayBase {
 public:
    using Derived = DerivedArray;
    using value_type = T;
    using host_reference = typename CudaStdVector<T>::Reference;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

    __host__ __device__ size_t index(size_t i) const;

    template <typename... Args>
    __host__ __device__ size_t index(size_t i, Args... args) const;

    template <size_t... I>
    __host__ __device__ size_t index(const CudaStdArray<size_t, N>& idx) const;

    __host__ __device__ T* data();

    __host__ __device__ const T* data() const;

    __host__ __device__ const CudaStdArray<size_t, N>& size() const;

    template <size_t M = N>
    __host__ __device__ std::enable_if_t<(M > 0), size_t> width() const;

    template <size_t M = N>
    __host__ __device__ std::enable_if_t<(M > 1), size_t> height() const;

    template <size_t M = N>
    __host__ __device__ std::enable_if_t<(M > 2), size_t> depth() const;

    __host__ __device__ size_t length() const;

#ifdef __CUDA_ARCH__
    __device__ reference at(size_t i);

    __device__ const_reference at(size_t i) const;

    template <typename... Args>
    __device__ reference at(size_t i, Args... args);

    template <typename... Args>
    __device__ const_reference at(size_t i, Args... args) const;

    __device__ reference at(const CudaStdArray<size_t, N>& idx);

    __device__ const_reference at(const CudaStdArray<size_t, N>& idx) const;

    __device__ reference operator[](size_t i);

    __device__ const_reference operator[](size_t i) const;

    template <typename... Args>
    __device__ reference operator()(size_t i, Args... args);

    template <typename... Args>
    __device__ const_reference operator()(size_t i, Args... args) const;

    __device__ reference operator()(const CudaStdArray<size_t, N>& idx);

    __device__ const_reference
    operator()(const CudaStdArray<size_t, N>& idx) const;
#else
    __host__ host_reference at(size_t i);

    __host__ value_type at(size_t i) const;

    template <typename... Args>
    __host__ host_reference at(size_t i, Args... args);

    template <typename... Args>
    __host__ value_type at(size_t i, Args... args) const;

    __host__ host_reference at(const CudaStdArray<size_t, N>& idx);

    __host__ value_type at(const CudaStdArray<size_t, N>& idx) const;

    __host__ host_reference operator[](size_t i);

    __host__ value_type operator[](size_t i) const;

    template <typename... Args>
    __host__ host_reference operator()(size_t i, Args... args);

    template <typename... Args>
    __host__ value_type operator()(size_t i, Args... args) const;

    __host__ host_reference operator()(const CudaStdArray<size_t, N>& idx);

    __host__ value_type operator()(const CudaStdArray<size_t, N>& idx) const;
#endif  // __CUDA_ARCH__

 protected:
    pointer _ptr = nullptr;
    CudaStdArray<size_t, N> _size;

    __host__ __device__ CudaArrayBase();

    __host__ __device__ CudaArrayBase(const CudaArrayBase& other);

    __host__ __device__ CudaArrayBase(CudaArrayBase&& other);

    template <typename... Args>
    __host__ __device__ void setPtrAndSize(pointer ptr, size_t ni,
                                           Args... args);

    __host__ __device__ void setPtrAndSize(pointer data,
                                           CudaStdArray<size_t, N> size);

    __host__ __device__ void swapPtrAndSize(CudaArrayBase& other);

    __host__ __device__ void clearPtrAndSize();

    __host__ __device__ CudaArrayBase& operator=(const CudaArrayBase& other);

    __host__ __device__ CudaArrayBase& operator=(CudaArrayBase&& other);

 private:
    template <typename... Args>
    __host__ __device__ size_t _index(size_t d, size_t i, Args... args) const;

    __host__ __device__ size_t _index(size_t, size_t i) const;

    template <size_t... I>
    __host__ __device__ size_t _index(const CudaStdArray<size_t, N>& idx,
                                      std::index_sequence<I...>) const;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaArray

template <typename T, size_t N>
class CudaArrayView;

template <typename T, size_t N>
class CudaArray final : public CudaArrayBase<T, N, Array<T, N>> {
    using Base = CudaArrayBase<T, N, Array<T, N>>;
    using Base::_size;
    using Base::setPtrAndSize;
    using Base::swapPtrAndSize;

 public:
    using Base::at;
    using Base::clearPtrAndSize;
    using Base::data;
    using Base::length;

    // CTOR
    CudaArray();

    CudaArray(const CudaStdArray<size_t, N>& size_, const T& initVal = T{});

    template <typename... Args>
    CudaArray(size_t nx, Args... args);

    CudaArray(NestedInitializerListsT<T, N> lst);

    template <size_t M = N>
    CudaArray(const std::enable_if_t<(M == 1), std::vector<T>>& vec);

    template <typename OtherDerived>
    CudaArray(const ArrayBase<T, N, OtherDerived>& other);

    template <typename OtherDerived>
    CudaArray(const CudaArrayBase<T, N, OtherDerived>& other);

    CudaArray(const CudaArray& other);

    CudaArray(CudaArray&& other);

    template <typename A, size_t M = N>
    std::enable_if_t<(M == 1), void> copyFrom(const std::vector<T, A>& vec);

    template <typename OtherDerived>
    void copyFrom(const ArrayBase<T, N, OtherDerived>& other);

    template <typename OtherDerived>
    void copyFrom(const CudaArrayBase<T, N, OtherDerived>& other);

    template <typename A, size_t M = N>
    std::enable_if_t<(M == 1), void> copyTo(std::vector<T, A>& vec);

    void copyTo(Array<T, N>& other);

    void copyTo(ArrayView<T, N>& other);

    void copyTo(CudaArray<T, N>& other);

    void copyTo(CudaArrayView<T, N>& other);

    void fill(const T& val);

    // resize
    void resize(CudaStdArray<size_t, N> size_, const T& initVal = T{});

    template <typename... Args>
    void resize(size_t nx, Args... args);

    template <size_t M = N>
    std::enable_if_t<(M == 1), void> append(const T& val);

    template <typename A, size_t M = N>
    std::enable_if_t<(M == 1), void> append(const std::vector<T, A>& extra);

    template <typename OtherDerived, size_t M = N>
    std::enable_if_t<(M == 1), void> append(
        const ArrayBase<T, N, OtherDerived>& extra);

    template <typename OtherDerived, size_t M = N>
    std::enable_if_t<(M == 1), void> append(
        const CudaArrayBase<T, N, OtherDerived>& extra);

    void clear();

    void swap(CudaArray& other);

    // Views
    CudaArrayView<T, N> view();

    CudaArrayView<const T, N> view() const;

    // Assignment Operators
    template <size_t M = N>
    CudaArray& operator=(const std::enable_if_t<(M == 1), std::vector<T>>& vec);

    template <typename OtherDerived>
    CudaArray& operator=(const ArrayBase<T, N, OtherDerived>& other);

    template <typename OtherDerived>
    CudaArray& operator=(const CudaArrayBase<T, N, OtherDerived>& other);

    CudaArray& operator=(const CudaArray& other);

    CudaArray& operator=(CudaArray&& other);

 private:
    CudaStdVector<T> _data;
};

template <class T>
using CudaArray1 = CudaArray<T, 1>;

template <class T>
using CudaArray2 = CudaArray<T, 2>;

template <class T>
using CudaArray3 = CudaArray<T, 3>;

template <class T>
using CudaArray4 = CudaArray<T, 4>;

}  // namespace jet

#include <jet/detail/cuda_array-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_ARRAY_H_

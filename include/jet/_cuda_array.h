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
    using reference = typename CudaStdVector<T>::Reference;
    using const_reference = const reference;
    using pointer = T*;
    using const_pointer = const T*;

    virtual ~CudaArrayBase() = default;

    size_t index(size_t i) const;

    template <typename... Args>
    size_t index(size_t i, Args... args) const;

    template <size_t... I>
    size_t index(const CudaStdArray<size_t, N>& idx) const;

    T* data();

    const T* data() const;

    const CudaStdArray<size_t, N>& size() const;

    template <size_t M = N>
    std::enable_if_t<(M > 0), size_t> width() const;

    template <size_t M = N>
    std::enable_if_t<(M > 1), size_t> height() const;

    template <size_t M = N>
    std::enable_if_t<(M > 2), size_t> depth() const;

    size_t length() const;

    reference at(size_t i);

    value_type at(size_t i) const;

    template <typename... Args>
    reference at(size_t i, Args... args);

    template <typename... Args>
    value_type at(size_t i, Args... args) const;

    reference at(const CudaStdArray<size_t, N>& idx);

    value_type at(const CudaStdArray<size_t, N>& idx) const;

    reference operator[](size_t i);

    value_type operator[](size_t i) const;

    template <typename... Args>
    reference operator()(size_t i, Args... args);

    template <typename... Args>
    value_type operator()(size_t i, Args... args) const;

    reference operator()(const CudaStdArray<size_t, N>& idx);

    value_type operator()(const CudaStdArray<size_t, N>& idx) const;

 protected:
    pointer _ptr = nullptr;
    CudaStdArray<size_t, N> _size;

    CudaArrayBase();

    CudaArrayBase(const CudaArrayBase& other);

    CudaArrayBase(CudaArrayBase&& other);

    template <typename... Args>
    void setPtrAndSize(pointer ptr, size_t ni, Args... args);

    void setPtrAndSize(pointer data, CudaStdArray<size_t, N> size);

    void swapPtrAndSize(CudaArrayBase& other);

    void clearPtrAndSize();

    CudaArrayBase& operator=(const CudaArrayBase& other);

    CudaArrayBase& operator=(CudaArrayBase&& other);

 private:
    template <typename... Args>
    size_t _index(size_t d, size_t i, Args... args) const;

    size_t _index(size_t, size_t i) const;

    template <size_t... I>
    size_t _index(const CudaStdArray<size_t, N>& idx,
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
    using Base::length;
    using Base::data;

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

    template <size_t M = N>
    void copyFrom(const std::enable_if_t<(M == 1), std::vector<T>>& vec);

    template <typename OtherDerived>
    void copyFrom(const ArrayBase<T, N, OtherDerived>& other);

    template <typename OtherDerived>
    void copyFrom(const CudaArrayBase<T, N, OtherDerived>& other);

    void fill(const T& val);

    // resize
    void resize(CudaStdArray<size_t, N> size_, const T& initVal = T{});

    template <typename... Args>
    void resize(size_t nx, Args... args);

    template <size_t M = N>
    std::enable_if_t<(M == 1), void> append(const T& val);

    template <typename OtherDerived, size_t M = N>
    std::enable_if_t<(M == 1), void> append(
        const CudaArrayBase<T, N, OtherDerived>& extra);

    void clear();

    void swap(CudaArray& other);

    // Views
    CudaArrayView<T, N> view();

    CudaArrayView<const T, N> view() const;

    // Assignment Operators
    template <typename OtherDerived>
    CudaArray& operator=(const CudaArrayBase<T, N, OtherDerived>& other);

    CudaArray& operator=(const CudaArray& other);

    CudaArray& operator=(CudaArray&& other);

 private:
    CudaStdVector<T> _data;
};

template <class T>
using NewCudaArray1 = CudaArray<T, 1>;

template <class T>
using NewCudaArray2 = CudaArray<T, 2>;

template <class T>
using NewCudaArray3 = CudaArray<T, 3>;

template <class T>
using NewCudaArray4 = CudaArray<T, 4>;

}  // namespace jet

#include <jet/detail/_cuda_array-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_ARRAY_H_

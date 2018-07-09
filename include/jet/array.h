// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_H_
#define INCLUDE_JET_ARRAY_H_

#include <jet/matrix.h>
#include <jet/nested_initializer_list.h>

#ifdef JET_USE_CUDA
#include <thrust/device_vector.h>
#endif

#include <algorithm>
#include <functional>
#include <vector>

namespace jet {

template <typename T>
struct CpuMemory {
    using DevicePtrType = T*;
    using ContainerType = std::vector<T>;

    using reference = T&;
    using const_reference = const T&;
    using iterator = DevicePtrType;
    using const_iterator = const DevicePtrType;

    DevicePtrType ptr = nullptr;

    CpuMemory() {}

    CpuMemory(T* ptr_) { set(ptr_); }

    T* data() { return ptr; }

    const T* data() const { return ptr; }

    void set(T* p) { ptr = p; }

    static CpuMemory handleFromContainer(ContainerType& cnt) {
        CpuMemory handle;
        handle.ptr = cnt.data();
        return handle;
    }
};

#ifdef JET_USE_CUDA
template <typename T>
struct CudaMemory {
    using DevicePtrType = thrust::device_ptr<T>;
    using ContainerType = thrust::device_vector<T>;

    using reference = typename ContainerType::reference;
    using const_reference = typename ContainerType::const_reference;
    using iterator = DevicePtrType;
    using const_iterator = const DevicePtrType;

    DevicePtrType ptr;

    CudaMemory() {}

    CudaMemory(T* ptr_) { set(ptr_); }

    T* data() { return thrust::raw_pointer_cast(ptr); }

    const T* data() const { return thrust::raw_pointer_cast(ptr); }

    void set(T* p) { ptr = thrust::device_pointer_cast<T>(p); }

    static CudaMemory handleFromContainer(ContainerType& cnt) {
        CudaMemory handle;
        handle.ptr = cnt.data();
        return handle;
    }
};
#endif

////////////////////////////////////////////////////////////////////////////////
// MARK: ArrayBase

template <typename T, size_t N, typename Handle, typename Derived>
class ArrayBase {
 public:
    using value_type = T;
    using reference = typename Handle::reference;
    using const_reference = typename Handle::const_reference;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = typename Handle::iterator;
    using const_iterator = typename Handle::const_iterator;

    virtual ~ArrayBase() = default;

    size_t index(size_t i) const;

    template <typename... Args>
    size_t index(size_t i, Args... args) const;

    template <size_t... I>
    size_t index(const Vector<size_t, N>& idx) const;

    T* data();

    const T* data() const;

    const Vector<size_t, N>& size() const;

    template <size_t M = N>
    std::enable_if_t<(M > 0), size_t> width() const;

    template <size_t M = N>
    std::enable_if_t<(M > 1), size_t> height() const;

    template <size_t M = N>
    std::enable_if_t<(M > 2), size_t> depth() const;

    size_t length() const;

    iterator begin();

    const_iterator begin() const;

    iterator end();

    const_iterator end() const;

    Handle devicePtr() const;

    reference at(size_t i);

    const_reference at(size_t i) const;

    template <typename... Args>
    reference at(size_t i, Args... args);

    template <typename... Args>
    const_reference at(size_t i, Args... args) const;

    reference at(const Vector<size_t, N>& idx);

    const_reference at(const Vector<size_t, N>& idx) const;

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;

    template <typename... Args>
    reference operator()(size_t i, Args... args);

    template <typename... Args>
    const_reference operator()(size_t i, Args... args) const;

    reference operator()(const Vector<size_t, N>& idx);

    const_reference operator()(const Vector<size_t, N>& idx) const;

 protected:
    Handle _handle;
    Vector<size_t, N> _size;

    ArrayBase();

    ArrayBase(const ArrayBase& other);

    ArrayBase(ArrayBase&& other);

    template <typename... Args>
    void setHandleAndSize(Handle handle, size_t ni, Args... args);

    void setHandleAndSize(Handle handle, Vector<size_t, N> size);

    void swapHandleAndSize(ArrayBase& other);

    void clear();

    ArrayBase& operator=(const ArrayBase& other);

    ArrayBase& operator=(ArrayBase&& other);

 private:
    template <typename... Args>
    size_t _index(size_t d, size_t i, Args... args) const;

    size_t _index(size_t, size_t i) const;

    template <size_t... I>
    size_t _index(const Vector<size_t, N>& idx,
                  std::index_sequence<I...>) const;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: Array

template <typename T, size_t N, typename Handle>
class ArrayView;

template <typename T, size_t N, typename Handle>
class Array final : public ArrayBase<T, N, Handle, Array<T, N, Handle>> {
    typedef ArrayBase<T, N, Handle, Array<T, N, Handle>> Base;
    using Base::_size;
    using Base::at;
    using Base::clear;
    using Base::setHandleAndSize;
    using Base::swapHandleAndSize;

 public:
    using ContainerType = typename Handle::ContainerType;

    // CTOR
    Array();

    Array(const Vector<size_t, N>& size_, const T& initVal = T{});

    template <typename... Args>
    Array(size_t nx, Args... args);

    Array(NestedInitializerListsT<T, N> lst);

    template <size_t M = N>
    Array(const std::enable_if_t<(M == 1), std::vector<T>>& vec);

#ifdef JET_USE_CUDA
    template <size_t M = N>
    Array(
        const std::enable_if_t<(M == 1), thrust::device_vector<T>>& vec);

    template <size_t M = N>
    Array(const std::enable_if_t<(N == 1), thrust::host_vector<T>>& vec);
#endif

    template <typename OtherHandle, typename OtherDerived>
    Array(const ArrayBase<T, N, OtherHandle, OtherDerived>& other);

    Array(const Array& other);
    
    Array(Array&& other);

    template <typename OtherHandle, typename OtherDerived>
    void copyFrom(const ArrayBase<T, N, OtherHandle, OtherDerived>& other);

    void fill(const T& val);

    // resize
    void resize(Vector<size_t, N> size_, const T& initVal = T{});

    template <typename... Args>
    void resize(size_t nx, Args... args);

    template <size_t M = N>
    std::enable_if_t<(M == 1), void> append(const T& val);

    template <typename OtherHandle, typename OtherDerived, size_t M = N>
    std::enable_if_t<(M == 1), void> append(
        const ArrayBase<T, N, OtherHandle, OtherDerived>& extra);

    void clear();

    void swap(Array& other);

    // Views
    ArrayView<T, N, Handle> view();

    ArrayView<const T, N, Handle> view() const;

    // Assignment Operators
    Array& operator=(const Array& other);

    Array& operator=(Array&& other);

 private:
    ContainerType _data;
};

template <class T>
using Array1 = Array<T, 1, CpuMemory<T>>;

template <class T>
using Array2 = Array<T, 2, CpuMemory<T>>;

template <class T>
using Array3 = Array<T, 3, CpuMemory<T>>;

template <class T>
using Array4 = Array<T, 4, CpuMemory<T>>;

#ifdef JET_USE_CUDA
template <class T>
using NewCudaArray1 = Array<T, 1, CudaMemory<T>>;

template <class T>
using NewCudaArray2 = Array<T, 2, CudaMemory<T>>;

template <class T>
using NewCudaArray3 = Array<T, 3, CudaMemory<T>>;

template <class T>
using NewCudaArray4 = Array<T, 4, CudaMemory<T>>;
#endif

}  // namespace jet

#include <jet/detail/array-inl.h>

#endif  // INCLUDE_JET_ARRAY_H_

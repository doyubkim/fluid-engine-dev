// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ARRAY_H_
#define INCLUDE_JET_ARRAY_H_

#include <jet/matrix.h>
#include <jet/nested_initializer_list.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace jet {

template <typename T>
struct CpuMemoryHandle {
    T* ptr = nullptr;

    CpuMemoryHandle() {}

    CpuMemoryHandle(T* ptr_) { set(ptr_); }

    T* data() { return ptr; }

    const T* data() const { return ptr; }

    void set(T* p) { ptr = p; }
};

template <typename T, size_t N, typename Device, typename Derived>
class ArrayBase;

template <typename T>
struct CpuDevice {
    using BufferType = std::vector<T>;
    using MemoryHandle = CpuMemoryHandle<T>;

    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    static auto handleFromContainer(BufferType& cnt) {
        CpuMemoryHandle<T> handle;
        handle.ptr = cnt.data();
        return handle;
    }

    //! \brief Fills given array with single value.
    template <typename T1, size_t N, typename D>
    static void fill(ArrayBase<T1, N, CpuDevice<T1>, D>& dst, const T1& value) {
        std::fill(dst.begin(), dst.end(), value);
    }

    //! \brief Copies contents from src vector to dst array.
    template <typename T1, typename T2, typename A1, size_t N, typename D2>
    static void copy(const std::vector<T1, A1>& src,
                     ArrayBase<T2, N, CpuDevice<T2>, D2>& dst) {
        JET_ASSERT(src.size() == dst.length());
        for (size_t i = 0; i < dst.length(); ++i) {
            dst[i] = src[i];
        }
    }

    //! \brief Copies contents from src array to dst array.
    template <typename T1, typename T2, size_t N, typename D1, typename D2>
    static void copy(const ArrayBase<T1, N, CpuDevice<T1>, D1>& src,
                     ArrayBase<T2, N, CpuDevice<T2>, D2>& dst) {
        JET_ASSERT(src.length() == dst.length());
        for (size_t i = 0; i < dst.length(); ++i) {
            dst[i] = src[i];
        }
    }

    //!
    //! \brief Copies contents from \p src array to \p dst array for given \p
    //! size.
    //!
    //! This function performs array copy for specified size range.
    //!
    template <typename T1, typename T2, size_t N, typename D1, typename D2>
    static void copy(const ArrayBase<T1, N, CpuDevice<T1>, D1>& src,
                     const Vector<size_t, N>& size,
                     ArrayBase<T2, N, CpuDevice<T2>, D2>& dst) {
        forEachIndex(size, [&](auto... idx) { dst(idx...) = src(idx...); });
    }

    //! \brief Copies contents from \p src array to \p dst array with offsets.
    //!
    //! This function performs offset copy such that
    //!
    //! \code
    //! dst[i + dstOffset] = src[i + srcOffset]
    //! \endcode
    //!
    template <typename T1, typename T2, typename D1, typename D2>
    static void copy(const ArrayBase<T1, 1, CpuDevice<T1>, D1>& src,
                     size_t srcOffset, ArrayBase<T2, 1, CpuDevice<T2>, D2>& dst,
                     size_t dstOffset) {
        JET_ASSERT(src.length() + dstOffset == dst.length() + srcOffset);
        auto dstIter = dst.begin() + dstOffset;
        auto srcIter = src.begin() + srcOffset;
        for (size_t i = 0; i + srcOffset < src.length(); ++i) {
            dstIter[i] = srcIter[i];
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// MARK: ArrayBase

template <typename T, size_t N, typename Device, typename Derived>
class ArrayBase {
 public:
    using value_type = T;
    using reference = typename Device::reference;
    using const_reference = typename Device::const_reference;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = typename Device::iterator;
    using const_iterator = typename Device::const_iterator;

    using MemoryHandle = typename Device::MemoryHandle;

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

    MemoryHandle handle() const;

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
    MemoryHandle _handle;
    Vector<size_t, N> _size;

    ArrayBase();

    ArrayBase(const ArrayBase& other);

    ArrayBase(ArrayBase&& other);

    template <typename... Args>
    void setHandleAndSize(MemoryHandle handle, size_t ni, Args... args);

    void setHandleAndSize(MemoryHandle handle, Vector<size_t, N> size);

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

template <typename T, size_t N, typename Device>
class ArrayView;

template <typename T, size_t N, typename Device = CpuDevice<T>>
class Array final : public ArrayBase<T, N, Device, Array<T, N, Device>> {
    using Base = ArrayBase<T, N, Device, Array<T, N, Device>>;

    using Base::_size;
    using Base::at;
    using Base::clear;
    using Base::setHandleAndSize;
    using Base::swapHandleAndSize;

 public:
    using BufferType = typename Device::BufferType;

    // CTOR
    Array();

    Array(const Vector<size_t, N>& size_, const T& initVal = T{});

    template <typename... Args>
    Array(size_t nx, Args... args);

    Array(NestedInitializerListsT<T, N> lst);

    template <size_t M = N>
    Array(const std::enable_if_t<(M == 1), std::vector<T>>& vec);

    template <typename OtherDevice, typename OtherDerived>
    Array(const ArrayBase<T, N, OtherDevice, OtherDerived>& other);

    Array(const Array& other);

    Array(Array&& other);

    template <typename OtherDevice, typename OtherDerived>
    void copyFrom(const ArrayBase<T, N, OtherDevice, OtherDerived>& other);

    void fill(const T& val);

    // resize
    void resize(Vector<size_t, N> size_, const T& initVal = T{});

    template <typename... Args>
    void resize(size_t nx, Args... args);

    template <size_t M = N>
    std::enable_if_t<(M == 1), void> append(const T& val);

    template <typename OtherDevice, typename OtherDerived, size_t M = N>
    std::enable_if_t<(M == 1), void> append(
        const ArrayBase<T, N, OtherDevice, OtherDerived>& extra);

    void clear();

    void swap(Array& other);

    // Views
    ArrayView<T, N, Device> view();

    ArrayView<const T, N, Device> view() const;

    // Assignment Operators
    Array& operator=(const Array& other);

    Array& operator=(Array&& other);

 private:
    BufferType _data;
};

template <class T>
using Array1 = Array<T, 1, CpuDevice<T>>;

template <class T>
using Array2 = Array<T, 2, CpuDevice<T>>;

template <class T>
using Array3 = Array<T, 3, CpuDevice<T>>;

template <class T>
using Array4 = Array<T, 4, CpuDevice<T>>;

}  // namespace jet

#include <jet/detail/array-inl.h>

#endif  // INCLUDE_JET_ARRAY_H_

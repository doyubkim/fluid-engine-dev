// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_CUDA_ARRAY_H_
#define INCLUDE_JET_CUDA_ARRAY_H_

#ifdef JET_USE_CUDA

#include <jet/array.h>
#include <jet/cuda_thrust_utils.h>

#include <thrust/device_vector.h>

namespace jet {

template <typename T>
struct CudaMemoryHandle {
    thrust::device_ptr<T> ptr;

    __host__ __device__ CudaMemoryHandle() {}

    __host__ __device__ CudaMemoryHandle(T* ptr_) { set(ptr_); }

    __host__ __device__ T* data() { return thrust::raw_pointer_cast(ptr); }

    __host__ __device__ const T* data() const {
        return thrust::raw_pointer_cast(ptr);
    }

    __host__ __device__ void set(T* p) {
        ptr = thrust::device_pointer_cast<T>(p);
    }
};

template <typename T>
struct CudaDevice {
    using BufferType = thrust::device_vector<T>;
    using MemoryHandle = CudaMemoryHandle<T>;

    using reference = typename BufferType::reference;
    using const_reference = typename BufferType::const_reference;
    using iterator = thrust::device_ptr<T>;
    using const_iterator = const thrust::device_ptr<T>;

    static auto handleFromContainer(BufferType& cnt) {
        CudaMemoryHandle<T> handle;
        handle.ptr = cnt.data();
        return handle;
    }

    template <typename T1, size_t N, typename D>
    static void fill(ArrayBase<T1, N, CudaDevice<T>, D>& dst, const T1& value) {
        thrust::fill(dst.begin(), dst.end(), value);
    }

    // Simple copy

    template <typename T1, typename T2, typename A1, size_t N, typename D2>
    static void copy(const std::vector<T1, A1>& src,
                     ArrayBase<T2, N, CudaDevice<T2>, D2>& dst) {
        JET_ASSERT(src.size() == dst.length());
        thrust::copy(thrust::host_device_ptr<const T1>(src.data()),
                     thrust::host_device_ptr<const T1>(src.data() + src.size()),
                     dst.begin());
    }

    template <typename T1, typename T2, size_t N, typename D1, typename D2>
    static void copy(const ArrayBase<T1, N, CpuDevice<T1>, D1>& src,
                     ArrayBase<T2, N, CudaDevice<T2>, D2>& dst) {
        JET_ASSERT(src.length() == dst.length());
        thrust::copy(
            thrust::host_device_ptr<const T1>(src.data()),
            thrust::host_device_ptr<const T1>(src.data() + src.length()),
            dst.begin());
    }

    template <typename T1, typename T2, size_t N, typename D1, typename D2>
    static void copy(const ArrayBase<T1, N, CudaDevice<T1>, D1>& src,
                     ArrayBase<T2, N, CpuDevice<T2>, D2>& dst) {
        JET_ASSERT(src.length() == dst.length());
        thrust::copy(src.begin(), src.end(),
                     thrust::host_device_ptr<T2>(dst.data()));
    }

    template <typename T1, typename T2, size_t N, typename D1, typename D2>
    static void copy(const ArrayBase<T1, N, CudaDevice<T1>, D1>& src,
                     ArrayBase<T2, N, CudaDevice<T2>, D2>& dst) {
        JET_ASSERT(src.length() == dst.length());
        thrust::copy(src.begin(), src.end(), dst.begin());
    }

    // Block copy

    template <typename T1, typename T2, size_t N, typename D1, typename D2>
    static void copy(const ArrayBase<T1, N, CpuDevice<T1>, D1>& src,
                     const thrust::array<size_t, N>& size,
                     ArrayBase<T2, N, CudaDevice<T2>, D2>& dst) {
        Array<T1, N, CudaDevice<T1>> cudaSrc(src);
        copy(cudaSrc, size, dst);
    }

    template <typename T1, typename T2, size_t N, typename D1, typename D2>
    static void copy(const ArrayBase<T1, N, CudaDevice<T1>, D1>& src,
                     const thrust::array<size_t, N>& size,
                     ArrayBase<T2, N, CpuDevice<T2>, D2>& dst) {
        Array<T1, N, CpuDevice<T1>> cpuSrc(src);
        CpuDevice<T>::copy(cpuSrc, size, dst);
    }

    template <typename T1, typename T2, size_t N, typename D1, typename D2>
    static void copy(const ArrayBase<T1, N, CudaDevice<T1>, D1>& src,
                     const thrust::array<size_t, N>& size,
                     ArrayBase<T2, N, CudaDevice<T2>, D2>& dst);

    // Offset copy

    template <typename T1, typename T2, typename D1, typename D2>
    static void copy(const ArrayBase<T1, 1, CpuDevice<T1>, D1>& src,
                     size_t srcOffset,
                     ArrayBase<T2, 1, CudaDevice<T2>, D2>& dst,
                     size_t dstOffset) {
        JET_ASSERT(src.length() + dstOffset == dst.length() + srcOffset);
        thrust::copy(
            thrust::host_device_ptr<const T1>(src.data() + srcOffset),
            thrust::host_device_ptr<const T1>(src.data() + src.length()),
            dst.begin() + dstOffset);
    }

    template <typename T1, typename T2, typename D1, typename D2>
    static void copy(const ArrayBase<T1, 1, CudaDevice<T1>, D1>& src,
                     size_t srcOffset, ArrayBase<T2, 1, CpuDevice<T2>, D2>& dst,
                     size_t dstOffset) {
        JET_ASSERT(src.length() + dstOffset == dst.length() + srcOffset);
        thrust::copy(src.begin() + srcOffset, src.end(),
                     thrust::host_device_ptr<T2>(dst.data() + dstOffset));
    }

    template <typename T1, typename T2, typename D1, typename D2>
    static void copy(const ArrayBase<T1, 1, CudaDevice<T1>, D1>& src,
                     size_t srcOffset,
                     ArrayBase<T2, 1, CudaDevice<T2>, D2>& dst,
                     size_t dstOffset) {
        JET_ASSERT(src.length() + dstOffset == dst.length() + srcOffset);
        thrust::copy(src.begin() + srcOffset, src.end(),
                     dst.begin() + dstOffset);
    }
};

////////////////////////////////////////////////////////////////////////////////
// MARK: ArrayBase Specialized for CUDA

template <typename T, size_t N, typename Derived>
class ArrayBase<T, N, CudaDevice<T>, Derived> {
 public:
    using Device = CudaDevice<T>;

    using value_type = T;
    using reference = typename Device::reference;
    using const_reference = typename Device::const_reference;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = typename Device::iterator;
    using const_iterator = typename Device::const_iterator;

    using MemoryHandle = typename Device::MemoryHandle;

    virtual ~ArrayBase() = default;

    __host__ __device__ size_t index(size_t i) const;

    template <typename... Args>
    __host__ __device__ size_t index(size_t i, Args... args) const;

    template <size_t... I>
    __host__ __device__ size_t index(const thrust::array<size_t, N>& idx) const;

    __host__ __device__ T* data();

    __host__ __device__ const T* data() const;

    __host__ __device__ const thrust::array<size_t, N>& size() const;

    template <size_t M = N>
    __host__ __device__ std::enable_if_t<(M > 0), size_t> width() const;

    template <size_t M = N>
    __host__ __device__ std::enable_if_t<(M > 1), size_t> height() const;

    template <size_t M = N>
    __host__ __device__ std::enable_if_t<(M > 2), size_t> depth() const;

    __host__ __device__ size_t length() const;

    __host__ __device__ iterator begin();

    __host__ __device__ const_iterator begin() const;

    __host__ __device__ iterator end();

    __host__ __device__ const_iterator end() const;

    __host__ __device__ MemoryHandle handle() const;

    __host__ __device__ reference at(size_t i);

    __host__ __device__ value_type at(size_t i) const;

    template <typename... Args>
    __host__ __device__ reference at(size_t i, Args... args);

    template <typename... Args>
    __host__ __device__ value_type at(size_t i, Args... args) const;

    __host__ __device__ reference at(const thrust::array<size_t, N>& idx);

    __host__ __device__ value_type
    at(const thrust::array<size_t, N>& idx) const;

    __host__ __device__ reference operator[](size_t i);

    __host__ __device__ value_type operator[](size_t i) const;

    template <typename... Args>
    __host__ __device__ reference operator()(size_t i, Args... args);

    template <typename... Args>
    __host__ __device__ value_type operator()(size_t i, Args... args) const;

    __host__ __device__ reference
    operator()(const thrust::array<size_t, N>& idx);

    __host__ __device__ value_type
    operator()(const thrust::array<size_t, N>& idx) const;

 protected:
    MemoryHandle _handle;
    thrust::array<size_t, N> _size;

    __host__ __device__ ArrayBase();

    __host__ __device__ ArrayBase(const ArrayBase& other);

    __host__ __device__ ArrayBase(ArrayBase&& other);

    template <typename... Args>
    __host__ __device__ void setHandleAndSize(MemoryHandle handle, size_t ni,
                                              Args... args);

    __host__ __device__ void setHandleAndSize(MemoryHandle handle,
                                              thrust::array<size_t, N> size);

    __host__ __device__ void swapHandleAndSize(ArrayBase& other);

    __host__ __device__ void clear();

    __host__ __device__ ArrayBase& operator=(const ArrayBase& other);

    __host__ __device__ ArrayBase& operator=(ArrayBase&& other);

 private:
    template <typename... Args>
    __host__ __device__ size_t _index(size_t d, size_t i, Args... args) const;

    __host__ __device__ size_t _index(size_t, size_t i) const;

    template <size_t... I>
    __host__ __device__ size_t _index(const thrust::array<size_t, N>& idx,
                                      std::index_sequence<I...>) const;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: Array Specialized for CUDA

template <typename T, size_t N>
class Array<T, N, CudaDevice<T>> final
    : public ArrayBase<T, N, CudaDevice<T>, Array<T, N, CudaDevice<T>>> {
    using Device = CudaDevice<T>;
    using Base = ArrayBase<T, N, Device, Array<T, N, Device>>;

    using Base::_size;
    using Base::setHandleAndSize;
    using Base::swapHandleAndSize;

 public:
    using BufferType = typename Device::BufferType;

    using Base::at;
    using Base::clear;
    using Base::length;

    // CTOR
    __host__ Array();

    __host__ Array(const thrust::array<size_t, N>& size_,
                   const T& initVal = T{});

    template <typename... Args>
    __host__ Array(size_t nx, Args... args);

    __host__ Array(NestedInitializerListsT<T, N> lst);

    template <size_t M = N>
    __host__ Array(const std::enable_if_t<(M == 1), std::vector<T>>& vec);

    template <typename OtherDevice, typename OtherDerived>
    __host__ Array(const ArrayBase<T, N, OtherDevice, OtherDerived>& other);

    __host__ Array(const Array& other);

    __host__ Array(Array&& other);

    template <typename OtherDevice, typename OtherDerived>
    __host__ void copyFrom(
        const ArrayBase<T, N, OtherDevice, OtherDerived>& other);

    __host__ void fill(const T& val);

    // resize
    __host__ void resize(thrust::array<size_t, N> size_,
                         const T& initVal = T{});

    template <typename... Args>
    __host__ void resize(size_t nx, Args... args);

    template <size_t M = N>
    __host__ std::enable_if_t<(M == 1), void> append(const T& val);

    template <typename OtherDevice, typename OtherDerived, size_t M = N>
    __host__ std::enable_if_t<(M == 1), void> append(
        const ArrayBase<T, N, OtherDevice, OtherDerived>& extra);

    __host__ void clear();

    __host__ void swap(Array& other);

    // Views
    __host__ ArrayView<T, N, Device> view();

    __host__ ArrayView<const T, N, Device> view() const;

    // Assignment Operators
    __host__ Array& operator=(const Array& other);

    __host__ Array& operator=(Array&& other);

 private:
    BufferType _data;
};

template <class T>
using NewCudaArray1 = Array<T, 1, CudaDevice<T>>;

template <class T>
using NewCudaArray2 = Array<T, 2, CudaDevice<T>>;

template <class T>
using NewCudaArray3 = Array<T, 3, CudaDevice<T>>;

template <class T>
using NewCudaArray4 = Array<T, 4, CudaDevice<T>>;

}  // namespace jet

#include <jet/detail/_cuda_array-inl.h>

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_CUDA_ARRAY_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_CUDA_ARRAY_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_ARRAY_INL_H_

#ifdef JET_USE_CUDA

#include <jet/cuda_array.h>
#include <jet/cuda_array_view.h>

namespace jet {

namespace internal {

template <typename T, size_t N, size_t I>
struct CudaBlockCopyHelper {
    template <typename... RemainingIndices>
    __host__ __device__ static void call(CudaArrayView<const T, N> src,
                                         CudaStdArray<size_t, N> size,
                                         CudaArrayView<T, N> dst,
                                         RemainingIndices... indices) {
        for (size_t i = 0; i < size[I - 1]; ++i) {
            CudaBlockCopyHelper<T, N, I - 1>::call(src, size, dst, i,
                                                   indices...);
        }
    }
};

template <typename T, size_t N>
struct CudaBlockCopyHelper<T, N, 1> {
    template <typename... RemainingIndices>
    __host__ __device__ static void call(CudaArrayView<const T, N> src,
                                         CudaStdArray<size_t, N> size,
                                         CudaArrayView<T, N> dst,
                                         RemainingIndices... indices) {
        for (size_t i = 0; i < size[0]; ++i) {
            dst(i, indices...) = src(i, indices...);
        }
    }
};

template <typename T, size_t N>
__global__ void cudaBlockCopyKernelN(CudaArrayView<const T, N> src,
                                     CudaStdArray<size_t, N> size,
                                     CudaArrayView<T, N> dst) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size[N - 1]) {
        CudaBlockCopyHelper<T, N, N - 1>::call(src, size, dst, i);
    }
}

template <typename T>
__global__ void cudaBlockCopyKernel1(CudaArrayView<const T, 1> src,
                                     CudaStdArray<size_t, 1> size,
                                     CudaArrayView<T, 1> dst) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size[0]) {
        dst[i] = src[i];
    }
}

template <typename T, size_t N>
struct CudaBlockCopy {
    static void call(CudaArrayView<const T, N> src,
                     CudaStdArray<size_t, N> size, CudaArrayView<T, N> dst) {
        if (size[N - 1] == 0) {
            return;
        }

        // Assuming i-major
        unsigned int numBlocks, numThreads;
        cudaComputeGridSize((unsigned int)size[N - 1], 256, numBlocks,
                            numThreads);
        cudaBlockCopyKernelN<<<numBlocks, numThreads>>>(src, size, dst);
        JET_CUDA_CHECK_LAST_ERROR("Failed executing cudaBlockCopyKernelN");
    }
};

template <typename T>
struct CudaBlockCopy<T, 1> {
    static void call(CudaArrayView<const T, 1> src,
                     CudaStdArray<size_t, 1> size, CudaArrayView<T, 1> dst) {
        if (size[0] == 0) {
            return;
        }

        // Assuming i-major
        unsigned int numBlocks, numThreads;
        cudaComputeGridSize((unsigned int)size[0], 256, numBlocks, numThreads);
        cudaBlockCopyKernel1<<<numBlocks, numThreads>>>(src, size, dst);
        JET_CUDA_CHECK_LAST_ERROR("Failed executing cudaBlockCopyKernel1");
    }
};

}  // namespace internal

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaArrayBase

template <typename T, size_t N, typename Derived>
size_t CudaArrayBase<T, N, Derived>::index(size_t i) const {
    return i;
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
size_t CudaArrayBase<T, N, Derived>::index(size_t i, Args... args) const {
    static_assert(sizeof...(args) == N - 1, "Invalid number of indices.");
    return i + _size[0] * _index(1, args...);
}

template <typename T, size_t N, typename Derived>
template <size_t... I>
size_t CudaArrayBase<T, N, Derived>::index(
    const CudaStdArray<size_t, N>& idx) const {
    return _index(idx, std::make_index_sequence<N>{});
}

template <typename T, size_t N, typename Derived>
T* CudaArrayBase<T, N, Derived>::data() {
    return _ptr;
}

template <typename T, size_t N, typename Derived>
const T* CudaArrayBase<T, N, Derived>::data() const {
    return _ptr;
}

template <typename T, size_t N, typename Derived>
const CudaStdArray<size_t, N>& CudaArrayBase<T, N, Derived>::size() const {
    return _size;
}

template <typename T, size_t N, typename Derived>
template <size_t M>
std::enable_if_t<(M > 0), size_t> CudaArrayBase<T, N, Derived>::width() const {
    return _size[0];
}

template <typename T, size_t N, typename Derived>
template <size_t M>
std::enable_if_t<(M > 1), size_t> CudaArrayBase<T, N, Derived>::height() const {
    return _size[1];
}

template <typename T, size_t N, typename Derived>
template <size_t M>
std::enable_if_t<(M > 2), size_t> CudaArrayBase<T, N, Derived>::depth() const {
    return _size[2];
}

template <typename T, size_t N, typename Derived>
size_t CudaArrayBase<T, N, Derived>::length() const {
    // TODO: Replace CudaStdArray with Vector
    // return product<size_t, N>(_size, 1);
    size_t l = _size[0];
    for (size_t i = 1; i < N; ++i) {
        l *= _size[i];
    }
    return l;
}

#ifdef __CUDA_ARCH__

template <typename T, size_t N, typename Derived>
__device__ typename CudaArrayBase<T, N, Derived>::reference
CudaArrayBase<T, N, Derived>::at(size_t i) {
    return _ptr[i];
}

template <typename T, size_t N, typename Derived>
__device__ typename CudaArrayBase<T, N, Derived>::const_reference
CudaArrayBase<T, N, Derived>::at(size_t i) const {
    return _ptr[i];
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
__device__ typename CudaArrayBase<T, N, Derived>::reference
CudaArrayBase<T, N, Derived>::at(size_t i, Args... args) {
    return at(index(i, args...));
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
__device__ typename CudaArrayBase<T, N, Derived>::const_reference
CudaArrayBase<T, N, Derived>::at(size_t i, Args... args) const {
    return at(index(i, args...));
}

template <typename T, size_t N, typename Derived>
__device__ typename CudaArrayBase<T, N, Derived>::reference
CudaArrayBase<T, N, Derived>::at(const CudaStdArray<size_t, N>& idx) {
    return at(index(idx));
}

template <typename T, size_t N, typename Derived>
__device__ typename CudaArrayBase<T, N, Derived>::const_reference
CudaArrayBase<T, N, Derived>::at(const CudaStdArray<size_t, N>& idx) const {
    return at(index(idx));
}

template <typename T, size_t N, typename Derived>
__device__ typename CudaArrayBase<T, N, Derived>::reference
    CudaArrayBase<T, N, Derived>::operator[](size_t i) {
    return at(i);
}

template <typename T, size_t N, typename Derived>
__device__ typename CudaArrayBase<T, N, Derived>::const_reference
    CudaArrayBase<T, N, Derived>::operator[](size_t i) const {
    return at(i);
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
__device__ typename CudaArrayBase<T, N, Derived>::reference
CudaArrayBase<T, N, Derived>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
__device__ typename CudaArrayBase<T, N, Derived>::const_reference
CudaArrayBase<T, N, Derived>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template <typename T, size_t N, typename Derived>
__device__ typename CudaArrayBase<T, N, Derived>::reference
CudaArrayBase<T, N, Derived>::operator()(const CudaStdArray<size_t, N>& idx) {
    return at(idx);
}

template <typename T, size_t N, typename Derived>
__device__ typename CudaArrayBase<T, N, Derived>::const_reference
CudaArrayBase<T, N, Derived>::operator()(
    const CudaStdArray<size_t, N>& idx) const {
    return at(idx);
}

#else

template <typename T, size_t N, typename Derived>
typename CudaArrayBase<T, N, Derived>::host_reference
CudaArrayBase<T, N, Derived>::at(size_t i) {
    return host_reference(_ptr + i);
}

template <typename T, size_t N, typename Derived>
T CudaArrayBase<T, N, Derived>::at(size_t i) const {
    return (T)host_reference(_ptr + i);
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
typename CudaArrayBase<T, N, Derived>::host_reference
CudaArrayBase<T, N, Derived>::at(size_t i, Args... args) {
    return at(index(i, args...));
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
T CudaArrayBase<T, N, Derived>::at(size_t i, Args... args) const {
    return at(index(i, args...));
}

template <typename T, size_t N, typename Derived>
typename CudaArrayBase<T, N, Derived>::host_reference
CudaArrayBase<T, N, Derived>::at(const CudaStdArray<size_t, N>& idx) {
    return at(index(idx));
}

template <typename T, size_t N, typename Derived>
T CudaArrayBase<T, N, Derived>::at(const CudaStdArray<size_t, N>& idx) const {
    return at(index(idx));
}

template <typename T, size_t N, typename Derived>
typename CudaArrayBase<T, N, Derived>::host_reference
    CudaArrayBase<T, N, Derived>::operator[](size_t i) {
    return at(i);
}

template <typename T, size_t N, typename Derived>
T CudaArrayBase<T, N, Derived>::operator[](size_t i) const {
    return at(i);
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
typename CudaArrayBase<T, N, Derived>::host_reference
CudaArrayBase<T, N, Derived>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
T CudaArrayBase<T, N, Derived>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template <typename T, size_t N, typename Derived>
typename CudaArrayBase<T, N, Derived>::host_reference
CudaArrayBase<T, N, Derived>::operator()(const CudaStdArray<size_t, N>& idx) {
    return at(idx);
}

template <typename T, size_t N, typename Derived>
T CudaArrayBase<T, N, Derived>::operator()(
    const CudaStdArray<size_t, N>& idx) const {
    return at(idx);
}
#endif  // __CUDA_ARCH__

template <typename T, size_t N, typename Derived>
CudaArrayBase<T, N, Derived>::CudaArrayBase() : _size{} {}

template <typename T, size_t N, typename Derived>
CudaArrayBase<T, N, Derived>::CudaArrayBase(const CudaArrayBase& other) {
    setPtrAndSize(other._ptr, other._size);
}

template <typename T, size_t N, typename Derived>
CudaArrayBase<T, N, Derived>::CudaArrayBase(CudaArrayBase&& other) {
    *this = std::move(other);
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
void CudaArrayBase<T, N, Derived>::setPtrAndSize(pointer ptr, size_t ni,
                                                 Args... args) {
    setPtrAndSize(ptr, CudaStdArray<size_t, N>{ni, args...});
}

template <typename T, size_t N, typename Derived>
void CudaArrayBase<T, N, Derived>::setPtrAndSize(pointer ptr,
                                                 CudaStdArray<size_t, N> size) {
    _ptr = ptr;
    _size = size;
}

template <typename T, size_t N, typename Derived>
void CudaArrayBase<T, N, Derived>::swapPtrAndSize(CudaArrayBase& other) {
    cudaSwap(_ptr, other._ptr);
    cudaSwap(_size, other._size);
}

template <typename T, size_t N, typename Derived>
void CudaArrayBase<T, N, Derived>::clearPtrAndSize() {
    setPtrAndSize(nullptr, CudaStdArray<size_t, N>{});
}

template <typename T, size_t N, typename Derived>
CudaArrayBase<T, N, Derived>& CudaArrayBase<T, N, Derived>::operator=(
    const CudaArrayBase& other) {
    setPtrAndSize(other._ptr, other._size);
    return *this;
}

template <typename T, size_t N, typename Derived>
CudaArrayBase<T, N, Derived>& CudaArrayBase<T, N, Derived>::operator=(
    CudaArrayBase&& other) {
    setPtrAndSize(other._ptr, other._size);
    other.setPtrAndSize(nullptr, CudaStdArray<size_t, N>{});
    return *this;
}

template <typename T, size_t N, typename Derived>
template <typename... Args>
size_t CudaArrayBase<T, N, Derived>::_index(size_t d, size_t i,
                                            Args... args) const {
    return i + _size[d] * _index(d + 1, args...);
}

template <typename T, size_t N, typename Derived>
size_t CudaArrayBase<T, N, Derived>::_index(size_t, size_t i) const {
    return i;
}

template <typename T, size_t N, typename Derived>
template <size_t... I>
size_t CudaArrayBase<T, N, Derived>::_index(const CudaStdArray<size_t, N>& idx,
                                            std::index_sequence<I...>) const {
    return index(idx[I]...);
}

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaArray

template <typename T, size_t N>
CudaArray<T, N>::CudaArray() : Base() {}

template <typename T, size_t N>
CudaArray<T, N>::CudaArray(const CudaStdArray<size_t, N>& size_,
                           const T& initVal)
    : CudaArray() {
    // TODO: Replace CudaStdArray with Vector
    size_t l = size_[0];
    for (size_t i = 1; i < N; ++i) {
        l *= size_[i];
    }
    _data.resize(l, initVal);
    Base::setPtrAndSize(_data.data(), size_);
}

template <typename T, size_t N>
template <typename... Args>
CudaArray<T, N>::CudaArray(size_t nx, Args... args) : CudaArray() {
    // TODO: Replace CudaStdArray with Vector
    Vector<size_t, N> newSizeV;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(newSizeV, initVal, nx,
                                                   args...);
    CudaStdArray<size_t, N> newSize(newSizeV);
    CudaArray newArray(newSize, initVal);
    *this = std::move(newArray);
}

template <typename T, size_t N>
CudaArray<T, N>::CudaArray(NestedInitializerListsT<T, N> lst) : CudaArray() {
    Vector<size_t, N> newSize;
    internal::GetSizeFromInitList<T, N, N>::call(newSize, lst);

    Array<T, N> newCpuArray(newSize);
    internal::SetArrayFromInitList<T, N, N>::call(newCpuArray, lst);
    copyFrom(newCpuArray);
}

template <typename T, size_t N>
template <size_t M>
CudaArray<T, N>::CudaArray(
    const std::enable_if_t<(M == 1), std::vector<T>>& vec)
    : CudaArray() {
    copyFrom(vec);
}

template <typename T, size_t N>
template <typename OtherDerived>
CudaArray<T, N>::CudaArray(const ArrayBase<T, N, OtherDerived>& other)
    : CudaArray() {
    copyFrom(other);
}

template <typename T, size_t N>
template <typename OtherDerived>
CudaArray<T, N>::CudaArray(const CudaArrayBase<T, N, OtherDerived>& other)
    : CudaArray() {
    copyFrom(other);
}

template <typename T, size_t N>
CudaArray<T, N>::CudaArray(const CudaArray& other) : CudaArray() {
    copyFrom(other);
}

template <typename T, size_t N>
CudaArray<T, N>::CudaArray(CudaArray&& other) : CudaArray() {
    *this = std::move(other);
}

template <typename T, size_t N>
template <typename A, size_t M>
std::enable_if_t<(M == 1), void> CudaArray<T, N>::copyFrom(
    const std::vector<T, A>& vec) {
    CudaArray newArray(vec.size());
    newArray._data.copyFrom(vec);
    newArray.setPtrAndSize(newArray._data.data(), newArray.size());
    *this = std::move(newArray);
}

template <typename T, size_t N>
template <typename OtherDerived>
void CudaArray<T, N>::copyFrom(const ArrayBase<T, N, OtherDerived>& other) {
    CudaArray newArray(other.size());
    cudaCopyHostToDevice(other.data(), other.length(), newArray.data());
    *this = std::move(newArray);
}

template <typename T, size_t N>
template <typename OtherDerived>
void CudaArray<T, N>::copyFrom(
    const ArrayBase<const T, N, OtherDerived>& other) {
    CudaArray newArray(other.size());
    cudaCopyHostToDevice(other.data(), other.length(), newArray.data());
    *this = std::move(newArray);
}

template <typename T, size_t N>
template <typename OtherDerived>
void CudaArray<T, N>::copyFrom(const CudaArrayBase<T, N, OtherDerived>& other) {
    CudaArray newArray(other.size());
    cudaCopyDeviceToDevice(other.data(), other.length(), newArray.data());
    *this = std::move(newArray);
}

template <typename T, size_t N>
template <typename OtherDerived>
void CudaArray<T, N>::copyFrom(
    const CudaArrayBase<const T, N, OtherDerived>& other) {
    CudaArray newArray(other.size());
    cudaCopyDeviceToDevice(other.data(), other.length(), newArray.data());
    *this = std::move(newArray);
}

template <typename T, size_t N>
template <typename A, size_t M>
std::enable_if_t<(M == 1), void> CudaArray<T, N>::copyTo(
    std::vector<T, A>& vec) {
    vec.resize(length());
    cudaCopyDeviceToHost(data(), length(), vec.data());
}

template <typename T, size_t N>
void CudaArray<T, N>::copyTo(Array<T, N>& other) {
    other.resize(size().toVector());
    cudaCopyDeviceToHost(data(), length(), other.data());
}

template <typename T, size_t N>
void CudaArray<T, N>::copyTo(ArrayView<T, N>& other) {
    JET_ASSERT(size().toVector() == other.size());
    cudaCopyDeviceToHost(data(), length(), other.data());
}

template <typename T, size_t N>
void CudaArray<T, N>::copyTo(CudaArray<T, N>& other) {
    other.resize(size().toVector());
    cudaCopyDeviceToDevice(data(), length(), other.data());
}

template <typename T, size_t N>
void CudaArray<T, N>::copyTo(CudaArrayView<T, N>& other) {
    JET_ASSERT(size() == other.size());
    cudaCopyDeviceToDevice(data(), length(), other.data());
}

template <typename T, size_t N>
void CudaArray<T, N>::fill(const T& val) {
    _data.fill(val);
}

template <typename T, size_t N>
void CudaArray<T, N>::resize(CudaStdArray<size_t, N> newSize,
                             const T& initVal) {
    // TODO: Replace with Vector
    CudaArray newArray(newSize, initVal);
    CudaStdArray<size_t, N> minSize;
    for (size_t i = 0; i < N; ++i) {
        minSize[i] = std::min(_size[i], newArray._size[i]);
    }

    internal::CudaBlockCopy<T, N>::call(view(), minSize, newArray.view());

    *this = std::move(newArray);
}

template <typename T, size_t N>
template <typename... Args>
void CudaArray<T, N>::resize(size_t nx, Args... args) {
    // TODO: Replace CudaStdArray with Vector
    Vector<size_t, N> newSizeV;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(newSizeV, initVal, nx,
                                                   args...);

    CudaStdArray<size_t, N> newSize(newSizeV);
    resize(newSize, initVal);
}

template <typename T, size_t N>
template <size_t M>
std::enable_if_t<(M == 1), void> CudaArray<T, N>::append(const T& val) {
    _data.push_back(val);
    Base::setPtrAndSize(_data.data(), _data.size());
}

template <typename T, size_t N>
template <typename A, size_t M>
std::enable_if_t<(M == 1), void> CudaArray<T, N>::append(
    const std::vector<T, A>& extra) {
    _data.append(extra);
    _size[0] = _data.size();
}

template <typename T, size_t N>
template <typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> CudaArray<T, N>::append(
    const ArrayBase<T, N, OtherDerived>& extra) {
    CudaArray newArray(length() + extra.length());
    cudaCopy(data(), length(), newArray.data());
    cudaCopyHostToDevice(extra.data(), extra.length(),
                         newArray.data() + _size[0]);
    swap(newArray);
}

template <typename T, size_t N>
template <typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> CudaArray<T, N>::append(
    const CudaArrayBase<T, N, OtherDerived>& extra) {
    CudaArray newArray(length() + extra.length());
    cudaCopy(data(), length(), newArray.data());
    cudaCopy(extra.data(), extra.length(), newArray.data() + _size[0]);
    swap(newArray);
}

template <typename T, size_t N>
void CudaArray<T, N>::clear() {
    Base::clearPtrAndSize();
    _data.clear();
}

template <typename T, size_t N>
void CudaArray<T, N>::swap(CudaArray& other) {
    Base::swapPtrAndSize(other);
    _data.swap(other._data);
}

template <typename T, size_t N>
CudaArrayView<T, N> CudaArray<T, N>::view() {
    return CudaArrayView<T, N>(*this);
};

template <typename T, size_t N>
CudaArrayView<const T, N> CudaArray<T, N>::view() const {
    return CudaArrayView<const T, N>(*this);
};

template <typename T, size_t N>
template <size_t M>
CudaArray<T, N>& CudaArray<T, N>::operator=(
    const std::enable_if_t<(M == 1), std::vector<T>>& vec) {
    copyFrom(vec);
    return *this;
}

template <typename T, size_t N>
template <typename OtherDerived>
CudaArray<T, N>& CudaArray<T, N>::operator=(
    const ArrayBase<T, N, OtherDerived>& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N>
template <typename OtherDerived>
CudaArray<T, N>& CudaArray<T, N>::operator=(
    const ArrayBase<const T, N, OtherDerived>& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N>
template <typename OtherDerived>
CudaArray<T, N>& CudaArray<T, N>::operator=(
    const CudaArrayBase<T, N, OtherDerived>& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N>
template <typename OtherDerived>
CudaArray<T, N>& CudaArray<T, N>::operator=(
    const CudaArrayBase<const T, N, OtherDerived>& other) {
    copyFrom(other);
    return *this;
}

template <typename T, size_t N>
CudaArray<T, N>& CudaArray<T, N>::operator=(const CudaArray& other) {
    _data = other._data;
    Base::setPtrAndSize(_data.data(), other.size());
    return *this;
}

template <typename T, size_t N>
CudaArray<T, N>& CudaArray<T, N>::operator=(CudaArray&& other) {
    swap(other);
    other.clear();
    return *this;
}

}  // namespace jet

#endif  // JET_USE_CUDA

#endif  // INCLUDE_JET_DETAIL_CUDA_ARRAY_INL_H_

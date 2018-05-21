// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_DETAIL_CUDA_TEXTURE_INL_H_
#define INCLUDE_JET_DETAIL_CUDA_TEXTURE_INL_H_

#include <jet/cuda_texture.h>
#include <jet/cuda_utils.h>

namespace jet {

template <typename T, size_t N, typename Derived>
CudaTexture<T, N, Derived>::CudaTexture() {}

template <typename T, size_t N, typename Derived>
CudaTexture<T, N, Derived>::CudaTexture(const CudaArrayView<T, N>& view) {
    set(view);
}

template <typename T, size_t N, typename Derived>
CudaTexture<T, N, Derived>::CudaTexture(const CudaTexture& other) {
    set(static_cast<const Derived&>(other));
}

template <typename T, size_t N, typename Derived>
CudaTexture<T, N, Derived>::CudaTexture(CudaTexture&& other) {
    *this = std::move(other);
}

template <typename T, size_t N, typename Derived>
CudaTexture<T, N, Derived>::~CudaTexture() {
    clear();
}

template <typename T, size_t N, typename Derived>
void CudaTexture<T, N, Derived>::clear() {
    if (_array != nullptr) {
        cudaFreeArray(_array);
        _array = nullptr;
    }

    if (_tex != 0) {
        cudaDestroyTextureObject(_tex);
        _tex = 0;
    }

    _size = Size<N>{};
}

template <typename T, size_t N, typename Derived>
void CudaTexture<T, N, Derived>::set(const CudaArrayView<T, N>& view) {
    static_cast<Derived*>(this)->_set(view);
}

template <typename T, size_t N, typename Derived>
void CudaTexture<T, N, Derived>::set(const Derived& other) {
    static_cast<Derived*>(this)->_set(other);
}

template <typename T, size_t N, typename Derived>
cudaTextureObject_t CudaTexture<T, N, Derived>::textureObject() const {
    return _tex;
}

template <typename T, size_t N, typename Derived>
CudaTexture<T, N, Derived>& CudaTexture<T, N, Derived>::operator=(
    const CudaTexture& other) {
    set(other);
    return *this;
}

template <typename T, size_t N, typename Derived>
cudaTextureObject_t CudaTexture<T, N, Derived>::createTexture(
    cudaArray_t array, cudaTextureFilterMode filterMode,
    bool shouldNormalizeCoords) {
    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = filterMode;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = shouldNormalizeCoords;

    // Create texture object
    cudaTextureObject_t tex = 0;
    checkResult(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

    return tex;
}

//

template <typename T>
CudaTexture1<T>::CudaTexture1() : Base() {}

template <typename T>
CudaTexture1<T>::CudaTexture1(const CudaArrayView<T, 1>& view) : Base(view) {}

template <typename T>
CudaTexture1<T>::CudaTexture1(const CudaTexture1& other) : Base(other) {}

template <typename T>
CudaTexture1<T>::CudaTexture1(CudaTexture1&& other) : Base() {
    *this = std::move(other);
}

template <typename T>
size_t CudaTexture1<T>::size() const {
    return _size[0];
}

template <typename T>
CudaTexture1<T>& CudaTexture1<T>::operator=(CudaTexture1&& other) {
    clear();
    std::swap(_size, other._size);
    std::swap(_array, other._array);
    std::swap(_tex, other._tex);
    return *this;
}

template <typename T>
void CudaTexture1<T>::_set(const CudaArrayView<T, 1>& view) {
    clear();

    size_t len = view.size();
    if (len == 0) {
        return;
    }

    _size[0] = view.size();

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    checkResult(cudaMallocArray(&_array, &channelDesc, len, 1));

    // Copy to device memory to CUDA array
    checkResult(cudaMemcpyToArray(_array, 0, 0, view.data(), sizeof(T) * len,
                                  cudaMemcpyDeviceToDevice));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

template <typename T>
void CudaTexture1<T>::_set(const CudaTexture1& other) {
    clear();

    size_t len = other.size();

    if (len == 0) {
        return;
    }

    _size = other._size;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    checkResult(cudaMallocArray(&_array, &channelDesc, len, 1));

    // Copy to device memory to CUDA array
    checkResult(cudaMemcpyArrayToArray(_array, 0, 0, other._array, 0, 0,
                                       sizeof(T) * len,
                                       cudaMemcpyDeviceToDevice));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

//

template <typename T>
CudaTexture2<T>::CudaTexture2() : Base() {}

template <typename T>
CudaTexture2<T>::CudaTexture2(const CudaArrayView<T, 2>& view) : Base(view) {}

template <typename T>
CudaTexture2<T>::CudaTexture2(const CudaTexture2& other) : Base(other) {}

template <typename T>
CudaTexture2<T>::CudaTexture2(CudaTexture2&& other) : Base() {
    *this = std::move(other);
}

template <typename T>
Size2 CudaTexture2<T>::size() const {
    // TODO: Size2 should be specialization (or alias) of Size<N=2>
    return Size2(_size.x, _size.y);
}

template <typename T>
size_t CudaTexture2<T>::width() const {
    return _size[0];
}

template <typename T>
size_t CudaTexture2<T>::height() const {
    return _size[1];
}

template <typename T>
CudaTexture2<T>& CudaTexture2<T>::operator=(CudaTexture2&& other) {
    clear();
    std::swap(_size, other._size);
    std::swap(_array, other._array);
    std::swap(_tex, other._tex);
    return *this;
}

template <typename T>
void CudaTexture2<T>::_set(const CudaArrayView<T, 2>& view) {
    clear();

    size_t len = view.width() * view.height();
    if (len == 0) {
        return;
    }

    // TODO: Size2 should be specialization (or alias) of Size<N=2>
    _size = Size<2>{view.width(), view.height()};

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    checkResult(
        cudaMallocArray(&_array, &channelDesc, view.width(), view.height()));

    // Copy to device memory to CUDA array
    checkResult(cudaMemcpyToArray(_array, 0, 0, view.data(), sizeof(T) * len,
                                  cudaMemcpyDeviceToDevice));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

template <typename T>
void CudaTexture2<T>::_set(const CudaTexture2& other) {
    clear();

    size_t len = other.width() * other.height();

    if (len == 0) {
        return;
    }

    _size = other._size;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    checkResult(
        cudaMallocArray(&_array, &channelDesc, other.width(), other.height()));

    // Copy to device memory to CUDA array
    checkResult(cudaMemcpyArrayToArray(_array, 0, 0, other._array, 0, 0,
                                       sizeof(T) * len,
                                       cudaMemcpyDeviceToDevice));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

//

template <typename T>
CudaTexture3<T>::CudaTexture3() : Base() {}

template <typename T>
CudaTexture3<T>::CudaTexture3(const CudaArrayView<T, 3>& view) : Base(view) {}

template <typename T>
CudaTexture3<T>::CudaTexture3(const CudaTexture3& other) : Base(other) {}

template <typename T>
CudaTexture3<T>::CudaTexture3(CudaTexture3&& other) : Base() {
    *this = std::move(other);
}

template <typename T>
Size3 CudaTexture3<T>::size() const {
    // TODO: Size3 should be specialization (or alias) of Size<N=3>
    return Size3(_size.x, _size.y, _size.z);
}

template <typename T>
size_t CudaTexture3<T>::width() const {
    return _size[0];
}

template <typename T>
size_t CudaTexture3<T>::height() const {
    return _size[1];
}

template <typename T>
size_t CudaTexture3<T>::depth() const {
    return _size[2];
}

template <typename T>
CudaTexture3<T>& CudaTexture3<T>::operator=(CudaTexture3&& other) {
    clear();
    std::swap(_size, other._size);
    std::swap(_array, other._array);
    std::swap(_tex, other._tex);
    return *this;
}

template <typename T>
void CudaTexture3<T>::_set(const CudaArrayView<T, 3>& view) {
    clear();

    size_t len = view.width() * view.height() * view.depth();
    if (len == 0) {
        return;
    }

    // TODO: Size3 should be specialization (or alias) of Size<N=3>
    _size = Size<3>{view.width(), view.height(), view.depth()};

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    // Note that the first param is not in bytes in this non-linear memory case.
    cudaExtent ext = make_cudaExtent(view.width(), view.height(), view.depth());
    checkResult(cudaMalloc3DArray(&_array, &channelDesc, ext));

    // Copy to device memory to CUDA array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr((void*)view.data(), view.width() * sizeof(T),
                            view.width(), view.height());
    copyParams.dstArray = _array;
    copyParams.extent = ext;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    checkResult(cudaMemcpy3D(&copyParams));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

template <typename T>
void CudaTexture3<T>::_set(const CudaTexture3& other) {
    clear();

    size_t len = other.width() * other.height() * other.depth();

    if (len == 0) {
        return;
    }

    _size = other._size;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    // Note that the first param is not in bytes in this non-linear memory case.
    cudaExtent ext =
        make_cudaExtent(other.width(), other.height(), other.depth());
    checkResult(cudaMalloc3DArray(&_array, &channelDesc, ext));

    // Copy to device memory to CUDA array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcArray = other._array;
    copyParams.dstArray = _array;
    copyParams.extent = ext;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    checkResult(cudaMemcpy3D(&copyParams));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_CUDA_TEXTURE_INL_H_

#endif  // JET_USE_CUDA

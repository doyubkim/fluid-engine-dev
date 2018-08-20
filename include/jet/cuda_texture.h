// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_TEXTURE_H_
#define INCLUDE_JET_CUDA_TEXTURE_H_

#include <jet/cuda_array_view.h>

#include <cuda_runtime.h>

namespace jet {

template <typename T, size_t N, typename Derived>
class CudaTextureBase {
 public:
    static_assert(N >= 1 || N <= 3,
                  "Not implemented - N should be either 1, 2 or 3.");

    virtual ~CudaTextureBase();

    void clear();

    void set(const ArrayView<const T, N>& view);

    void set(const CudaArrayView<const T, N>& view);

    void set(const Derived& other);

    cudaTextureObject_t textureObject() const;

 protected:
    CudaStdArray<size_t, N> _size;
    cudaArray_t _array = nullptr;
    cudaTextureObject_t _tex = 0;

    CudaTextureBase();

    CudaTextureBase(const ArrayView<const T, N>& view);

    CudaTextureBase(const CudaArrayView<const T, N>& view);

    CudaTextureBase(const CudaTextureBase& other);

    CudaTextureBase(CudaTextureBase&& other);

    CudaTextureBase& operator=(const CudaTextureBase& other);

    CudaTextureBase& operator=(CudaTextureBase&& other) = delete;

    static cudaTextureObject_t createTexture(
        cudaArray_t array,
        cudaTextureFilterMode filterMode = cudaFilterModeLinear,
        bool shouldNormalizeCoords = false);
};

template <typename T>
class CudaTexture1 final : public CudaTextureBase<T, 1, CudaTexture1<T>> {
    using Base = CudaTextureBase<T, 1, CudaTexture1<T>>;

    using Base::_array;
    using Base::_size;
    using Base::_tex;
    using Base::createTexture;

 public:
    using Base::clear;
    using Base::textureObject;
    using Base::operator=;

    CudaTexture1();

    CudaTexture1(const ConstArrayView1<T>& view);

    CudaTexture1(const ConstCudaArrayView1<T>& view);

    CudaTexture1(const CudaTexture1& other);

    CudaTexture1(CudaTexture1&& other);

    size_t size() const;

    void resize(const CudaStdArray<size_t, 1>& size);

    template <typename View>
    void set(const View& view, cudaMemcpyKind memcpyKind);

    void set(const CudaTexture1& other);

    CudaTexture1& operator=(CudaTexture1&& other);
};

template <typename T>
class CudaTexture2 final : public CudaTextureBase<T, 2, CudaTexture2<T>> {
    using Base = CudaTextureBase<T, 2, CudaTexture2<T>>;

    using Base::_array;
    using Base::_size;
    using Base::_tex;
    using Base::createTexture;

 public:
    using Base::clear;
    using Base::textureObject;
    using Base::operator=;

    CudaTexture2();

    CudaTexture2(const ConstArrayView2<T>& view);

    CudaTexture2(const ConstCudaArrayView2<T>& view);

    CudaTexture2(const CudaTexture2& other);

    CudaTexture2(CudaTexture2&& other);

    CudaStdArray<size_t, 2> size() const;

    size_t width() const;

    size_t height() const;

    void resize(const CudaStdArray<size_t, 2>& size);

    template <typename View>
    void set(const View& view, cudaMemcpyKind memcpyKind);

    void set(const CudaTexture2& other);

    CudaTexture2& operator=(CudaTexture2&& other);
};

template <typename T>
class CudaTexture3 final : public CudaTextureBase<T, 3, CudaTexture3<T>> {
    using Base = CudaTextureBase<T, 3, CudaTexture3<T>>;

    using Base::_array;
    using Base::_size;
    using Base::_tex;
    using Base::createTexture;

 public:
    using Base::clear;
    using Base::textureObject;
    using Base::operator=;

    CudaTexture3();

    CudaTexture3(const ConstArrayView3<T>& view);

    CudaTexture3(const ConstCudaArrayView3<T>& view);

    CudaTexture3(const CudaTexture3& other);

    CudaTexture3(CudaTexture3&& other);

    CudaStdArray<size_t, 3> size() const;

    size_t width() const;

    size_t height() const;

    size_t depth() const;

    void resize(const CudaStdArray<size_t, 3>& size);

    template <typename View>
    void set(const View& view, cudaMemcpyKind memcpyKind);

    void set(const CudaTexture3& other);

    CudaTexture3& operator=(CudaTexture3&& other);
};

}  // namespace jet

#include "detail/cuda_texture-inl.h"

#endif  // INCLUDE_JET_CUDA_TEXTURE_H_

#endif  // JET_USE_CUDA

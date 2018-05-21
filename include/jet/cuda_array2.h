// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY2_H_
#define INCLUDE_JET_CUDA_ARRAY2_H_

#include <jet/array_view2.h>
#include <jet/cuda_array.h>
#include <jet/macros.h>
#include <jet/size2.h>

#include <thrust/device_vector.h>

#include <vector>

namespace jet {

template <typename T, size_t N>
class CudaArrayView;

template <typename T, size_t N>
class ConstCudaArrayView;

template <typename T>
class CudaArray<T, 2> final {
public:
    typedef T value_type;
    typedef thrust::device_vector<T> ContainerType;
    typedef typename ContainerType::reference reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef thrust::device_ptr<T> iterator;
    typedef iterator const_iterator;

    CudaArray();

    explicit CudaArray(const Size2& size, const T& initVal = T());

    explicit CudaArray(size_t width, size_t height, const T& initVal = T());

    CudaArray(const ConstArrayView<T, 2>& view);

    CudaArray(const ConstCudaArrayView<T, 2>& view);

    CudaArray(const std::initializer_list<std::initializer_list<T>>& lst);

    CudaArray(const CudaArray& other);

    CudaArray(CudaArray&& other);

    void set(const T& value);

    void set(const ConstArrayView<T, 2>& view);

    void set(const ConstCudaArrayView<T, 2>& view);

    void set(const std::initializer_list<std::initializer_list<T>>& lst);

    void set(const CudaArray& other);

    void clear();

    void resize(const Size2& size, const T& initVal = T());

    void swap(CudaArray& other);

    const Size2& size() const;

    size_t width() const;

    size_t height() const;

    pointer data();

    const_pointer data() const;

    iterator begin();

    iterator begin() const;

    iterator end();

    iterator end() const;

    CudaArrayView<T, 2> view();

    ConstCudaArrayView<T, 2> view() const;

    //! Returns the reference to i-th element.
    reference operator[](size_t i);

    value_type operator[](size_t i) const;

    reference operator()(size_t i, size_t j);

    value_type operator()(size_t i, size_t j) const;

    CudaArray& operator=(const T& value);

    CudaArray& operator=(const ArrayView1<T>& view);

    CudaArray& operator=(const CudaArrayView<T, 2>& view);

    CudaArray& operator=(
        const std::initializer_list<std::initializer_list<T>>& lst);

    CudaArray& operator=(const CudaArray& other);

    CudaArray& operator=(CudaArray&& other);

 private:
    ContainerType _data;
    Size2 _size;
};

//! Type alias for 2-D CUDA array.
template <typename T>
using CudaArray2 = CudaArray<T, 2>;

}  // namespace jet

#include "detail/cuda_array2-inl.h"

#endif  // INCLUDE_JET_CUDA_ARRAY2_H_

#endif  // JET_USE_CUDA

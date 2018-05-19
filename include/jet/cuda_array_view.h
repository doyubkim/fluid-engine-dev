// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY_VIEW_H_
#define INCLUDE_JET_CUDA_ARRAY_VIEW_H_

#include <jet/size.h>

namespace jet {

//!
//! \brief Generic N-dimensional CUDA array view class interface.
//!
//! This class provides generic template class for N-dimensional array view
//! where N must be either 1, 2 or 3. This particular class exists to provide
//! generic interface for 1, 2 or 3 dimensional array views using template
//! specialization only, but it cannot create any instance by itself.
//!
//! \tparam T - Real number type.
//! \tparam N - Dimension.
//!
template <typename T, size_t N>
class CudaArrayView final {
 public:
    static_assert(N < 1 || N > 3,
                  "Not implemented - N should be either 1, 2 or 3.");
};

//!
//! \brief Generic N-dimensional const CUDA array view class interface.
//!
//! This class provides generic template class for N-dimensional const array
//! view where N must be either 1, 2 or 3. This particular class exists to
//! provide generic interface for 1, 2 or 3 dimensional array views using
//! template specialization only, but it cannot create any instance by itself.
//!
//! \tparam T - Real number type.
//! \tparam N - Dimension.
//!
template <typename T, size_t N>
class ConstCudaArrayView final {
 public:
    static_assert(N < 1 || N > 3,
                  "Not implemented - N should be either 1, 2 or 3.");
};

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_ARRAY_VIEW_H_

#endif  // JET_USE_CUDA

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SIZE_H_
#define INCLUDE_JET_SIZE_H_

#include <jet/point.h>

namespace jet {

//! \brief N-D size type.
template <size_t N> using Size = Point<size_t, N>;

}  // namespace jet

// #include "detail/size-inl.h"

#endif  // INCLUDE_JET_SIZE_H_


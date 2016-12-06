// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SIZE_H_
#define INCLUDE_JET_SIZE_H_

#include <jet/point.h>

namespace jet {

//! \brief N-D size type.
template <size_t N> using Size = Point<size_t, N>;

}  // namespace jet

// #include "detail/size-inl.h"

#endif  // INCLUDE_JET_SIZE_H_


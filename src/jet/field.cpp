// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/field.h>

namespace jet {

template <size_t N>
Field<N>::Field() {}

template <size_t N>
Field<N>::~Field() {}

template class Field<2>;

template class Field<3>;

}  // namespace jet

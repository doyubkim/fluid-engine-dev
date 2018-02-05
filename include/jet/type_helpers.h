// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_TYPE_HELPERS_H_
#define INCLUDE_JET_TYPE_HELPERS_H_

namespace jet {

//! Returns the type of the value itself.
template <typename T>
struct ScalarType {
    typedef T value;
};

}  // namespace jet

#endif  // INCLUDE_JET_TYPE_HELPERS_H_

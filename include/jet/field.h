// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FIELD_H_
#define INCLUDE_JET_FIELD_H_

#include <memory>

namespace jet {

//! Abstract base class for N-D fields.
template <size_t N>
class Field {
 public:
    Field();

    virtual ~Field();
};

//! 2-D Field type.
using Field2 = Field<2>;

//! 3-D Field type.
using Field3 = Field<3>;

//! Shared pointer type for Field2.
using Field2Ptr = std::shared_ptr<Field2>;

//! Shared pointer type for Field3.
using Field3Ptr = std::shared_ptr<Field3>;

}  // namespace jet

#endif  // INCLUDE_JET_FIELD_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FIELD2_H_
#define INCLUDE_JET_FIELD2_H_

#include <memory>

namespace jet {

//! Abstract base class for 2-D fields.
class Field2 {
 public:
    Field2();

    virtual ~Field2();
};

typedef std::shared_ptr<Field2> Field2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FIELD2_H_

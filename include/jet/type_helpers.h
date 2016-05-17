// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_TYPE_HELPERS_H_
#define INCLUDE_JET_TYPE_HELPERS_H_

namespace jet {

template <typename T>
struct ScalarType {
    typedef T value;
};

}  // namespace jet

#endif // INCLUDE_JET_TYPE_HELPERS_H_
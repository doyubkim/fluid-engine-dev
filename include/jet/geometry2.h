// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GEOMETRY2_H_
#define INCLUDE_JET_GEOMETRY2_H_

#include <memory>

namespace jet {

class Geometry2 {
 public:
    Geometry2();

    virtual ~Geometry2();
};

typedef std::shared_ptr<Geometry2> Geometry2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GEOMETRY2_H_

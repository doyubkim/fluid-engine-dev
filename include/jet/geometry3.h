// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GEOMETRY3_H_
#define INCLUDE_JET_GEOMETRY3_H_

#include <memory>

namespace jet {

class Geometry3 {
 public:
    Geometry3();

    virtual ~Geometry3();
};

typedef std::shared_ptr<Geometry3> Geometry3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GEOMETRY3_H_

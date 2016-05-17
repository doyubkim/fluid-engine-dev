// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER3_H_
#define INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER3_H_

#include <jet/fdm_linear_system3.h>
#include <memory>

namespace jet {

class FdmLinearSystemSolver3 {
 public:
    virtual bool solve(FdmLinearSystem3* system) = 0;
};

typedef std::shared_ptr<FdmLinearSystemSolver3> FdmLinearSystemSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER3_H_

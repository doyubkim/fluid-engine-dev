// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER2_H_
#define INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER2_H_

#include <jet/fdm_linear_system2.h>
#include <memory>

namespace jet {

class FdmLinearSystemSolver2 {
 public:
    virtual bool solve(FdmLinearSystem2* system) = 0;
};

typedef std::shared_ptr<FdmLinearSystemSolver2> FdmLinearSystemSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER2_H_

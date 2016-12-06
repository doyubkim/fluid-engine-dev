// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER2_H_
#define INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER2_H_

#include <jet/fdm_linear_system2.h>
#include <memory>

namespace jet {

//! Abstract base class for 2-D finite difference-type linear system solver.
class FdmLinearSystemSolver2 {
 public:
    //! Solves the given linear system.
    virtual bool solve(FdmLinearSystem2* system) = 0;
};

//! Shared pointer type for the FdmLinearSystemSolver2.
typedef std::shared_ptr<FdmLinearSystemSolver2> FdmLinearSystemSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER2_H_

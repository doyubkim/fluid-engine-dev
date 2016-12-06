// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER3_H_
#define INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER3_H_

#include <jet/fdm_linear_system3.h>
#include <memory>

namespace jet {

//! Abstract base class for 3-D finite difference-type linear system solver.
class FdmLinearSystemSolver3 {
 public:
    //! Solves the given linear system.
    virtual bool solve(FdmLinearSystem3* system) = 0;
};

//! Shared pointer type for the FdmLinearSystemSolver3.
typedef std::shared_ptr<FdmLinearSystemSolver3> FdmLinearSystemSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_LINEAR_SYSTEM_SOLVER3_H_

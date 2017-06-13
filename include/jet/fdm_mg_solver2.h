// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_MG_SOLVER2_H_
#define INCLUDE_JET_FDM_MG_SOLVER2_H_

#include <jet/fdm_linear_system_solver2.h>
#include <jet/fdm_mg_linear_system2.h>
#include <jet/mg.h>

namespace jet {

//! \brief 2-D finite difference-type linear system solver using Multigrid.
class FdmMgSolver2 : public FdmLinearSystemSolver2 {
 public:
    //! Constructs the solver with given parameters.
    FdmMgSolver2(size_t maxNumberOfLevels,
                 unsigned int numberOfRestrictionIter = 10,
                 unsigned int numberOfCorrectionIter = 10,
                 unsigned int numberOfCoarsestIter = 10,
                 unsigned int numberOfFinalIter = 10,
                 double maxTolerance = 1e-9);

    //! Returns the Multigrid parameters.
    const MgParameters<FdmBlas2>& params() const;

    //! No-op. Multigrid-type solvers do not solve FdmLinearSystem2.
    bool solve(FdmLinearSystem2* system) final;

    //! Solves Multigrid linear system.
    virtual bool solve(FdmMgLinearSystem2* system);

 private:
    MgParameters<FdmBlas2> _mgParams;
};

//! Shared pointer type for the FdmMgSolver2.
typedef std::shared_ptr<FdmMgSolver2> FdmMgSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_MG_SOLVER2_H_

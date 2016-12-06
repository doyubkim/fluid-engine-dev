// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_CG_SOLVER3_H_
#define INCLUDE_JET_FDM_CG_SOLVER3_H_

#include <jet/fdm_linear_system_solver3.h>

namespace jet {

//! \brief 3-D finite difference-type linear system solver using conjugate
//!        gradient.
class FdmCgSolver3 final : public FdmLinearSystemSolver3 {
 public:
    //! Constructs the solver with given parameters.
    FdmCgSolver3(unsigned int maxNumberOfIterations, double tolerance);

    //! Solves the given linear system.
    bool solve(FdmLinearSystem3* system) override;

    //! Returns the max number of Jacobi iterations.
    unsigned int maxNumberOfIterations() const;

    //! Returns the last number of Jacobi iterations the solver made.
    unsigned int lastNumberOfIterations() const;

    //! Returns the max residual tolerance for the Jacobi method.
    double tolerance() const;

    //! Returns the last residual after the Jacobi iterations.
    double lastResidual() const;

 private:
    unsigned int _maxNumberOfIterations;
    unsigned int _lastNumberOfIterations;
    double _tolerance;
    double _lastResidual;

    FdmVector3 _r;
    FdmVector3 _d;
    FdmVector3 _q;
    FdmVector3 _s;
};

//! Shared pointer type for the FdmCgSolver3.
typedef std::shared_ptr<FdmCgSolver3> FdmCgSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_CG_SOLVER3_H_

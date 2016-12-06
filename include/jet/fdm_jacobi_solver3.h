// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_JACOBI_SOLVER3_H_
#define INCLUDE_JET_FDM_JACOBI_SOLVER3_H_

#include <jet/fdm_linear_system_solver3.h>

namespace jet {

//! \brief 3-D finite difference-type linear system solver using Jacobi method.
class FdmJacobiSolver3 final : public FdmLinearSystemSolver3 {
 public:
    //! Constructs the solver with given parameters.
    FdmJacobiSolver3(
        unsigned int maxNumberOfIterations,
        unsigned int residualCheckInterval,
        double tolerance);

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
    unsigned int _residualCheckInterval;
    double _tolerance;
    double _lastResidual;

    FdmVector3 _xTemp;
    FdmVector3 _residual;

    void relax(FdmLinearSystem3* system, FdmVector3* xTemp);
};

//! Shared pointer type for the FdmJacobiSolver3.
typedef std::shared_ptr<FdmJacobiSolver3> FdmJacobiSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_JACOBI_SOLVER3_H_

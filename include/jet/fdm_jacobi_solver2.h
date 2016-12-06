// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_JACOBI_SOLVER2_H_
#define INCLUDE_JET_FDM_JACOBI_SOLVER2_H_

#include <jet/fdm_linear_system_solver2.h>

namespace jet {

//! \brief 2-D finite difference-type linear system solver using Jacobi method.
class FdmJacobiSolver2 final : public FdmLinearSystemSolver2 {
 public:
    //! Constructs the solver with given parameters.
    FdmJacobiSolver2(
        unsigned int maxNumberOfIterations,
        unsigned int residualCheckInterval,
        double tolerance);

    //! Solves the given linear system.
    bool solve(FdmLinearSystem2* system) override;

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

    FdmVector2 _xTemp;
    FdmVector2 _residual;

    void relax(FdmLinearSystem2* system, FdmVector2* xTemp);
};

//! Shared pointer type for the FdmJacobiSolver2.
typedef std::shared_ptr<FdmJacobiSolver2> FdmJacobiSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_JACOBI_SOLVER2_H_

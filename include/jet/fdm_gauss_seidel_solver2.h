// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_GAUSS_SEIDEL_SOLVER2_H_
#define INCLUDE_JET_FDM_GAUSS_SEIDEL_SOLVER2_H_

#include <jet/fdm_linear_system_solver2.h>

namespace jet {

//! \brief 2-D finite difference-type linear system solver using Gauss-Seidel
//!        method.
class FdmGaussSeidelSolver2 final : public FdmLinearSystemSolver2 {
 public:
    //! Constructs the solver with given parameters.
    FdmGaussSeidelSolver2(
        unsigned int maxNumberOfIterations,
        unsigned int residualCheckInterval,
        double tolerance);

    //! Solves the given linear system.
    bool solve(FdmLinearSystem2* system) override;

    //! Returns the max number of Gauss-Seidel iterations.
    unsigned int maxNumberOfIterations() const;

    //! Returns the last number of Gauss-Seidel iterations the solver made.
    unsigned int lastNumberOfIterations() const;

    //! Returns the max residual tolerance for the Gauss-Seidel method.
    double tolerance() const;

    //! Returns the last residual after the Gauss-Seidel iterations.
    double lastResidual() const;

 private:
    unsigned int _maxNumberOfIterations;
    unsigned int _lastNumberOfIterations;
    unsigned int _residualCheckInterval;
    double _tolerance;
    double _lastResidual;

    FdmVector2 _residual;

    void relax(FdmLinearSystem2* system);
};

//! Shared pointer type for the FdmGaussSeidelSolver2.
typedef std::shared_ptr<FdmGaussSeidelSolver2> FdmGaussSeidelSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_GAUSS_SEIDEL_SOLVER2_H_

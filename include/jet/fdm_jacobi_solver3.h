// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_JACOBI_SOLVER3_H_
#define INCLUDE_JET_FDM_JACOBI_SOLVER3_H_

#include <jet/fdm_linear_system_solver3.h>

namespace jet {

class FdmJacobiSolver3 final : public FdmLinearSystemSolver3 {
 public:
    FdmJacobiSolver3(
        unsigned int maxNumberOfIterations,
        unsigned int residualCheckInterval,
        double tolerance);

    bool solve(FdmLinearSystem3* system) override;

    unsigned int maxNumberOfIterations() const;
    unsigned int lastNumberOfIterations() const;
    double tolerance() const;
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

typedef std::shared_ptr<FdmJacobiSolver3> FdmJacobiSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_JACOBI_SOLVER3_H_

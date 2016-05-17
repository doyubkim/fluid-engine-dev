// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_JACOBI_SOLVER2_H_
#define INCLUDE_JET_FDM_JACOBI_SOLVER2_H_

#include <jet/fdm_linear_system_solver2.h>

namespace jet {

class FdmJacobiSolver2 final : public FdmLinearSystemSolver2 {
 public:
    FdmJacobiSolver2(
        unsigned int maxNumberOfIterations,
        unsigned int residualCheckInterval,
        double tolerance);

    bool solve(FdmLinearSystem2* system) override;

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

    FdmVector2 _xTemp;
    FdmVector2 _residual;

    void relax(FdmLinearSystem2* system, FdmVector2* xTemp);
};

typedef std::shared_ptr<FdmJacobiSolver2> FdmJacobiSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_JACOBI_SOLVER2_H_

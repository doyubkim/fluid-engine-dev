// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_GAUSS_SEIDEL_SOLVER3_H_
#define INCLUDE_JET_FDM_GAUSS_SEIDEL_SOLVER3_H_

#include <jet/fdm_linear_system_solver3.h>

namespace jet {

class FdmGaussSeidelSolver3 final : public FdmLinearSystemSolver3 {
 public:
    FdmGaussSeidelSolver3(
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

    FdmVector3 _residual;

    void relax(FdmLinearSystem3* system);
};

typedef std::shared_ptr<FdmGaussSeidelSolver3> FdmGaussSeidelSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_GAUSS_SEIDEL_SOLVER3_H_

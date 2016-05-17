// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_CG_SOLVER3_H_
#define INCLUDE_JET_FDM_CG_SOLVER3_H_

#include <jet/fdm_linear_system_solver3.h>

namespace jet {

class FdmCgSolver3 final : public FdmLinearSystemSolver3 {
 public:
    FdmCgSolver3(unsigned int maxNumberOfIterations, double tolerance);

    bool solve(FdmLinearSystem3* system) override;

    unsigned int maxNumberOfIterations() const;
    unsigned int lastNumberOfIterations() const;
    double tolerance() const;
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

typedef std::shared_ptr<FdmCgSolver3> FdmCgSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_CG_SOLVER3_H_

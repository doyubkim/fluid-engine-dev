// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_CG_SOLVER2_H_
#define INCLUDE_JET_FDM_CG_SOLVER2_H_

#include <jet/fdm_linear_system_solver2.h>

namespace jet {

class FdmCgSolver2 final : public FdmLinearSystemSolver2 {
 public:
    FdmCgSolver2(unsigned int maxNumberOfIterations, double tolerance);

    bool solve(FdmLinearSystem2* system) override;

    unsigned int maxNumberOfIterations() const;
    unsigned int lastNumberOfIterations() const;
    double tolerance() const;
    double lastResidual() const;

 private:
    unsigned int _maxNumberOfIterations;
    unsigned int _lastNumberOfIterations;
    double _tolerance;
    double _lastResidual;

    FdmVector2 _r;
    FdmVector2 _d;
    FdmVector2 _q;
    FdmVector2 _s;
};

typedef std::shared_ptr<FdmCgSolver2> FdmCgSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_CG_SOLVER2_H_

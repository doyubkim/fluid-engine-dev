// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_ICCG_SOLVER3_H_
#define INCLUDE_JET_FDM_ICCG_SOLVER3_H_

#include <jet/fdm_cg_solver3.h>

namespace jet {

class FdmIccgSolver3 final : public FdmLinearSystemSolver3 {
 public:
    FdmIccgSolver3(unsigned int maxNumberOfIterations, double tolerance);

    bool solve(FdmLinearSystem3* system) override;

    unsigned int maxNumberOfIterations() const;
    unsigned int lastNumberOfIterations() const;
    double tolerance() const;
    double lastResidual() const;

 private:
    struct Preconditioner final {
        ConstArrayAccessor3<FdmMatrixRow3> A;
        FdmVector3 d;
        FdmVector3 y;

        void build(const FdmMatrix3& matrix);

        void solve(
            const FdmVector3& b,
            FdmVector3* x);
    };

    unsigned int _maxNumberOfIterations;
    unsigned int _lastNumberOfIterations;
    double _tolerance;
    double _lastResidualNorm;

    FdmVector3 _r;
    FdmVector3 _d;
    FdmVector3 _q;
    FdmVector3 _s;
    Preconditioner _precond;
};

typedef std::shared_ptr<FdmIccgSolver3> FdmIccgSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_ICCG_SOLVER3_H_

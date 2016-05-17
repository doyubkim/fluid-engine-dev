// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_ICCG_SOLVER2_H_
#define INCLUDE_JET_FDM_ICCG_SOLVER2_H_

#include <jet/fdm_cg_solver2.h>

namespace jet {

class FdmIccgSolver2 final : public FdmLinearSystemSolver2 {
 public:
    FdmIccgSolver2(unsigned int maxNumberOfIterations, double tolerance);

    bool solve(FdmLinearSystem2* system) override;

    unsigned int maxNumberOfIterations() const;
    unsigned int lastNumberOfIterations() const;
    double tolerance() const;
    double lastResidual() const;

 private:
    struct Preconditioner final {
        ConstArrayAccessor2<FdmMatrixRow2> A;
        FdmVector2 d;
        FdmVector2 y;

        void build(const FdmMatrix2& matrix);

        void solve(
            const FdmVector2& b,
            FdmVector2* x);
    };

    unsigned int _maxNumberOfIterations;
    unsigned int _lastNumberOfIterations;
    double _tolerance;
    double _lastResidualNorm;

    FdmVector2 _r;
    FdmVector2 _d;
    FdmVector2 _q;
    FdmVector2 _s;
    Preconditioner _precond;
};

typedef std::shared_ptr<FdmIccgSolver2> FdmIccgSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_ICCG_SOLVER2_H_

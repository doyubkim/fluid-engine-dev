// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_FDM_ICCG_SOLVER3_H_
#define INCLUDE_JET_FDM_ICCG_SOLVER3_H_

#include <jet/fdm_cg_solver3.h>

namespace jet {

//!
//! \brief 3-D finite difference-type linear system solver using incomplete
//!        Cholesky conjugate gradient (ICCG).
//!
class FdmIccgSolver3 final : public FdmLinearSystemSolver3 {
 public:
    //! Constructs the solver with given parameters.
    FdmIccgSolver3(unsigned int maxNumberOfIterations, double tolerance);

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

//! Shared pointer type for the FdmIccgSolver3.
typedef std::shared_ptr<FdmIccgSolver3> FdmIccgSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_ICCG_SOLVER3_H_

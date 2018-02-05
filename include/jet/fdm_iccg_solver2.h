// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_ICCG_SOLVER2_H_
#define INCLUDE_JET_FDM_ICCG_SOLVER2_H_

#include <jet/fdm_cg_solver2.h>

namespace jet {

//!
//! \brief 2-D finite difference-type linear system solver using incomplete
//!        Cholesky conjugate gradient (ICCG).
//!
class FdmIccgSolver2 final : public FdmLinearSystemSolver2 {
 public:
    //! Constructs the solver with given parameters.
    FdmIccgSolver2(unsigned int maxNumberOfIterations, double tolerance);

    //! Solves the given linear system.
    bool solve(FdmLinearSystem2* system) override;

    //! Solves the given compressed linear system.
    bool solveCompressed(FdmCompressedLinearSystem2* system) override;

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
        ConstArrayAccessor2<FdmMatrixRow2> A;
        FdmVector2 d;
        FdmVector2 y;

        void build(const FdmMatrix2& matrix);

        void solve(const FdmVector2& b, FdmVector2* x);
    };

    struct PreconditionerCompressed final {
        const MatrixCsrD* A;
        VectorND d;
        VectorND y;

        void build(const MatrixCsrD& matrix);

        void solve(const VectorND& b, VectorND* x);
    };

    unsigned int _maxNumberOfIterations;
    unsigned int _lastNumberOfIterations;
    double _tolerance;
    double _lastResidualNorm;

    // Uncompressed vectors and preconditioner
    FdmVector2 _r;
    FdmVector2 _d;
    FdmVector2 _q;
    FdmVector2 _s;
    Preconditioner _precond;

    // Compressed vectors and preconditioner
    VectorND _rComp;
    VectorND _dComp;
    VectorND _qComp;
    VectorND _sComp;
    PreconditionerCompressed _precondComp;

    void clearUncompressedVectors();
    void clearCompressedVectors();
};

//! Shared pointer type for the FdmIccgSolver2.
typedef std::shared_ptr<FdmIccgSolver2> FdmIccgSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_ICCG_SOLVER2_H_

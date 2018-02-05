// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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

    //! Solves the given compressed linear system.
    bool solveCompressed(FdmCompressedLinearSystem3* system) override;

    //! Returns the max number of ICCG iterations.
    unsigned int maxNumberOfIterations() const;

    //! Returns the last number of ICCG iterations the solver made.
    unsigned int lastNumberOfIterations() const;

    //! Returns the max residual tolerance for the ICCG method.
    double tolerance() const;

    //! Returns the last residual after the ICCG iterations.
    double lastResidual() const;

 private:
    struct Preconditioner final {
        ConstArrayAccessor3<FdmMatrixRow3> A;
        FdmVector3 d;
        FdmVector3 y;

        void build(const FdmMatrix3& matrix);

        void solve(const FdmVector3& b, FdmVector3* x);
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
    FdmVector3 _r;
    FdmVector3 _d;
    FdmVector3 _q;
    FdmVector3 _s;
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

//! Shared pointer type for the FdmIccgSolver3.
typedef std::shared_ptr<FdmIccgSolver3> FdmIccgSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_ICCG_SOLVER3_H_

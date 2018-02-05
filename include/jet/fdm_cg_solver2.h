// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_CG_SOLVER2_H_
#define INCLUDE_JET_FDM_CG_SOLVER2_H_

#include <jet/fdm_linear_system_solver2.h>

namespace jet {

//! \brief 2-D finite difference-type linear system solver using conjugate
//!        gradient.
class FdmCgSolver2 final : public FdmLinearSystemSolver2 {
 public:
    //! Constructs the solver with given parameters.
    FdmCgSolver2(unsigned int maxNumberOfIterations, double tolerance);

    //! Solves the given linear system.
    bool solve(FdmLinearSystem2* system) override;

    //! Solves the given compressed linear system.
    bool solveCompressed(FdmCompressedLinearSystem2* system) override;

    //! Returns the max number of CG iterations.
    unsigned int maxNumberOfIterations() const;

    //! Returns the last number of CG iterations the solver made.
    unsigned int lastNumberOfIterations() const;

    //! Returns the max residual tolerance for the CG method.
    double tolerance() const;

    //! Returns the last residual after the CG iterations.
    double lastResidual() const;

 private:
    unsigned int _maxNumberOfIterations;
    unsigned int _lastNumberOfIterations;
    double _tolerance;
    double _lastResidual;

    // Uncompressed vectors
    FdmVector2 _r;
    FdmVector2 _d;
    FdmVector2 _q;
    FdmVector2 _s;

    // Compressed vectors
    VectorND _rComp;
    VectorND _dComp;
    VectorND _qComp;
    VectorND _sComp;

    void clearUncompressedVectors();
    void clearCompressedVectors();
};

//! Shared pointer type for the FdmCgSolver2.
typedef std::shared_ptr<FdmCgSolver2> FdmCgSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_CG_SOLVER2_H_

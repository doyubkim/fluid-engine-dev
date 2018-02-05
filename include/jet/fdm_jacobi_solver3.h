// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_JACOBI_SOLVER3_H_
#define INCLUDE_JET_FDM_JACOBI_SOLVER3_H_

#include <jet/fdm_linear_system_solver3.h>

namespace jet {

//! \brief 3-D finite difference-type linear system solver using Jacobi method.
class FdmJacobiSolver3 final : public FdmLinearSystemSolver3 {
 public:
    //! Constructs the solver with given parameters.
    FdmJacobiSolver3(unsigned int maxNumberOfIterations,
                     unsigned int residualCheckInterval, double tolerance);

    //! Solves the given linear system.
    bool solve(FdmLinearSystem3* system) override;

    //! Solves the given compressed linear system.
    bool solveCompressed(FdmCompressedLinearSystem3* system) override;

    //! Returns the max number of Jacobi iterations.
    unsigned int maxNumberOfIterations() const;

    //! Returns the last number of Jacobi iterations the solver made.
    unsigned int lastNumberOfIterations() const;

    //! Returns the max residual tolerance for the Jacobi method.
    double tolerance() const;

    //! Returns the last residual after the Jacobi iterations.
    double lastResidual() const;

    //! Performs single Jacobi relaxation step.
    static void relax(const FdmMatrix3& A, const FdmVector3& b, FdmVector3* x,
                      FdmVector3* xTemp);

    //! Performs single Jacobi relaxation step for compressed sys.
    static void relax(const MatrixCsrD& A, const VectorND& b, VectorND* x,
                      VectorND* xTemp);

 private:
    unsigned int _maxNumberOfIterations;
    unsigned int _lastNumberOfIterations;
    unsigned int _residualCheckInterval;
    double _tolerance;
    double _lastResidual;

    // Uncompressed vectors
    FdmVector3 _xTemp;
    FdmVector3 _residual;

    // Compressed vectors
    VectorND _xTempComp;
    VectorND _residualComp;

    void clearUncompressedVectors();
    void clearCompressedVectors();
};

//! Shared pointer type for the FdmJacobiSolver3.
typedef std::shared_ptr<FdmJacobiSolver3> FdmJacobiSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_JACOBI_SOLVER3_H_

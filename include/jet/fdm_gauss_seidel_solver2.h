// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_GAUSS_SEIDEL_SOLVER2_H_
#define INCLUDE_JET_FDM_GAUSS_SEIDEL_SOLVER2_H_

#include <jet/fdm_linear_system_solver2.h>

namespace jet {

//! \brief 2-D finite difference-type linear system solver using Gauss-Seidel
//!        method.
class FdmGaussSeidelSolver2 final : public FdmLinearSystemSolver2 {
 public:
    //! Constructs the solver with given parameters.
    FdmGaussSeidelSolver2(unsigned int maxNumberOfIterations,
                          unsigned int residualCheckInterval, double tolerance,
                          double sorFactor = 1.0,
                          bool useRedBlackOrdering = false);

    //! Solves the given linear system.
    bool solve(FdmLinearSystem2* system) override;

    //! Solves the given compressed linear system.
    bool solveCompressed(FdmCompressedLinearSystem2* system) override;

    //! Returns the max number of Gauss-Seidel iterations.
    unsigned int maxNumberOfIterations() const;

    //! Returns the last number of Gauss-Seidel iterations the solver made.
    unsigned int lastNumberOfIterations() const;

    //! Returns the max residual tolerance for the Gauss-Seidel method.
    double tolerance() const;

    //! Returns the last residual after the Gauss-Seidel iterations.
    double lastResidual() const;

    //! Returns the SOR (Successive Over Relaxation) factor.
    double sorFactor() const;

    //! Returns true if red-black ordering is enabled.
    bool useRedBlackOrdering() const;

    //! Performs single natural Gauss-Seidel relaxation step.
    static void relax(const FdmMatrix2& A, const FdmVector2& b,
                      double sorFactor, FdmVector2* x);

    //! \brief Performs single natural Gauss-Seidel relaxation step for
    //!        compressed sys.
    static void relax(const MatrixCsrD& A, const VectorND& b, double sorFactor,
                      VectorND* x);

    //! Performs single Red-Black Gauss-Seidel relaxation step.
    static void relaxRedBlack(const FdmMatrix2& A, const FdmVector2& b,
                              double sorFactor, FdmVector2* x);

 private:
    unsigned int _maxNumberOfIterations;
    unsigned int _lastNumberOfIterations;
    unsigned int _residualCheckInterval;
    double _tolerance;
    double _lastResidual;
    double _sorFactor;
    bool _useRedBlackOrdering;

    // Uncompressed vectors
    FdmVector2 _residual;

    // Compressed vectors
    VectorND _residualComp;

    void clearUncompressedVectors();
    void clearCompressedVectors();
};

//! Shared pointer type for the FdmGaussSeidelSolver2.
typedef std::shared_ptr<FdmGaussSeidelSolver2> FdmGaussSeidelSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_GAUSS_SEIDEL_SOLVER2_H_

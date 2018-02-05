// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_MGPCG_SOLVER3_H_
#define INCLUDE_JET_FDM_MGPCG_SOLVER3_H_

#include <jet/fdm_mg_solver3.h>

namespace jet {

//!
//! \brief 3-D finite difference-type linear system solver using Multigrid
//!        Preconditioned conjugate gradient (MGPCG).
//!
//! \see McAdams, Aleka, Eftychios Sifakis, and Joseph Teran.
//!      "A parallel multigrid Poisson solver for fluids simulation on large
//!      grids." Proceedings of the 2010 ACM SIGGRAPH/Eurographics Symposium on
//!      Computer Animation. Eurographics Association, 2010.
//!
class FdmMgpcgSolver3 final : public FdmMgSolver3 {
 public:
    //!
    //! Constructs the solver with given parameters.
    //!
    //! \param numberOfCgIter - Number of CG iterations.
    //! \param maxNumberOfLevels - Number of maximum MG levels.
    //! \param numberOfRestrictionIter - Number of restriction iterations.
    //! \param numberOfCorrectionIter - Number of correction iterations.
    //! \param numberOfCoarsestIter - Number of iterations at the coarsest grid.
    //! \param numberOfFinalIter - Number of final iterations.
    //! \param maxTolerance - Number of max residual tolerance.
    FdmMgpcgSolver3(unsigned int numberOfCgIter, size_t maxNumberOfLevels,
                    unsigned int numberOfRestrictionIter = 5,
                    unsigned int numberOfCorrectionIter = 5,
                    unsigned int numberOfCoarsestIter = 20,
                    unsigned int numberOfFinalIter = 20,
                    double maxTolerance = 1e-9, double sorFactor = 1.5,
                    bool useRedBlackOrdering = false);

    //! Solves the given linear system.
    bool solve(FdmMgLinearSystem3* system) override;

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
        FdmMgLinearSystem3* system;
        MgParameters<FdmBlas3> mgParams;

        void build(FdmMgLinearSystem3* system, MgParameters<FdmBlas3> mgParams);

        void solve(const FdmVector3& b, FdmVector3* x);
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

//! Shared pointer type for the FdmMgpcgSolver3.
typedef std::shared_ptr<FdmMgpcgSolver3> FdmMgpcgSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FDM_MGPCG_SOLVER3_H_

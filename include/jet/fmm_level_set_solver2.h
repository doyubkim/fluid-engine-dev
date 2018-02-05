// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FMM_LEVEL_SET_SOLVER2_H_
#define INCLUDE_JET_FMM_LEVEL_SET_SOLVER2_H_

#include <jet/level_set_solver2.h>
#include <memory>

namespace jet {

//!
//! \brief Two-dimensional fast marching method (FMM) implementation.
//!
//! This class implements 2-D FMM. First-order upwind-style differencing is used
//! to solve the PDE.
//!
//! \see https://math.berkeley.edu/~sethian/2006/Explanations/fast_marching_explain.html
//! \see Sethian, James A. "A fast marching level set method for monotonically
//!     advancing fronts." Proceedings of the National Academy of Sciences 93.4
//!     (1996): 1591-1595.
//!
class FmmLevelSetSolver2 final : public LevelSetSolver2 {
 public:
    //! Default constructor.
    FmmLevelSetSolver2();

    //!
    //! Reinitializes given scalar field to signed-distance field.
    //!
    //! \param inputSdf Input signed-distance field which can be distorted.
    //! \param maxDistance Max range of reinitialization.
    //! \param outputSdf Output signed-distance field.
    //!
    void reinitialize(
        const ScalarGrid2& inputSdf,
        double maxDistance,
        ScalarGrid2* outputSdf) override;

    //!
    //! Extrapolates given scalar field from negative to positive SDF region.
    //!
    //! \param input Input scalar field to be extrapolated.
    //! \param sdf Reference signed-distance field.
    //! \param maxDistance Max range of extrapolation.
    //! \param output Output scalar field.
    //!
    void extrapolate(
        const ScalarGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        ScalarGrid2* output) override;

    //!
    //! Extrapolates given collocated vector field from negative to positive SDF
    //! region.
    //!
    //! \param input Input collocated vector field to be extrapolated.
    //! \param sdf Reference signed-distance field.
    //! \param maxDistance Max range of extrapolation.
    //! \param output Output collocated vector field.
    //!
    void extrapolate(
        const CollocatedVectorGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        CollocatedVectorGrid2* output) override;

    //!
    //! Extrapolates given face-centered vector field from negative to positive
    //! SDF region.
    //!
    //! \param input Input face-centered field to be extrapolated.
    //! \param sdf Reference signed-distance field.
    //! \param maxDistance Max range of extrapolation.
    //! \param output Output face-centered vector field.
    //!
    void extrapolate(
        const FaceCenteredGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        FaceCenteredGrid2* output) override;

 private:
    void extrapolate(
        const ConstArrayAccessor2<double>& input,
        const ConstArrayAccessor2<double>& sdf,
        const Vector2D& gridSpacing,
        double maxDistance,
        ArrayAccessor2<double> output);
};

//! Shared pointer type for the FmmLevelSetSolver2.
typedef std::shared_ptr<FmmLevelSetSolver2> FmmLevelSetSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FMM_LEVEL_SET_SOLVER2_H_

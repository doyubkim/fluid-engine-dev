// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FMM_LEVEL_SET_SOLVER3_H_
#define INCLUDE_JET_FMM_LEVEL_SET_SOLVER3_H_

#include <jet/level_set_solver3.h>
#include <memory>

namespace jet {

//!
//! \brief Three-dimensional fast marching method (FMM) implementation.
//!
//! This class implements 3-D FMM. First-order upwind-style differencing is used
//! to solve the PDE.
//!
//! \see https://math.berkeley.edu/~sethian/2006/Explanations/fast_marching_explain.html
//! \see Sethian, James A. "A fast marching level set method for monotonically
//!     advancing fronts." Proceedings of the National Academy of Sciences 93.4
//!     (1996): 1591-1595.
//!
class FmmLevelSetSolver3 final : public LevelSetSolver3 {
 public:
    //! Default constructor.
    FmmLevelSetSolver3();

    //!
    //! Reinitializes given scalar field to signed-distance field.
    //!
    //! \param inputSdf Input signed-distance field which can be distorted.
    //! \param maxDistance Max range of reinitialization.
    //! \param outputSdf Output signed-distance field.
    //!
    void reinitialize(
        const ScalarGrid3& inputSdf,
        double maxDistance,
        ScalarGrid3* outputSdf) override;

    //!
    //! Extrapolates given scalar field from negative to positive SDF region.
    //!
    //! \param input Input scalar field to be extrapolated.
    //! \param sdf Reference signed-distance field.
    //! \param maxDistance Max range of extrapolation.
    //! \param output Output scalar field.
    //!
    void extrapolate(
        const ScalarGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        ScalarGrid3* output) override;

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
        const CollocatedVectorGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        CollocatedVectorGrid3* output) override;

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
        const FaceCenteredGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        FaceCenteredGrid3* output) override;

 private:
    void extrapolate(
        const ConstArrayAccessor3<double>& input,
        const ConstArrayAccessor3<double>& sdf,
        const Vector3D& gridSpacing,
        double maxDistance,
        ArrayAccessor3<double> output);
};

//! Shared pointer type for the FmmLevelSetSolver3.
typedef std::shared_ptr<FmmLevelSetSolver3> FmmLevelSetSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_FMM_LEVEL_SET_SOLVER3_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SEMI_LAGRANGIAN2_H_
#define INCLUDE_JET_SEMI_LAGRANGIAN2_H_

#include <jet/advection_solver2.h>
#include <limits>

namespace jet {

//!
//! \brief Implementation of 2-D semi-Lagrangian advection solver.
//!
//! This class implements 2-D semi-Lagrangian advection solver. By default, the
//! class implements 1st-order (linear) algorithm for the spatial interpolation.
//! For the back-tracing, this class uses 2nd-order mid-point rule with adaptive
//! time-stepping (CFL <= 1).
//! To extend the class using higher-order spatial interpolation, the inheriting
//! classes can override SemiLagrangian2::getScalarSamplerFunc and
//! SemiLagrangian2::getVectorSamplerFunc. See CubicSemiLagrangian2 for example.
//!
class SemiLagrangian2 : public AdvectionSolver2 {
 public:
    SemiLagrangian2();

    virtual ~SemiLagrangian2();

    //!
    //! \brief Computes semi-Lagrangian for given scalar grid.
    //!
    //! This function computes semi-Lagrangian method to solve advection
    //! equation for given scalar field \p input and underlying vector field
    //! \p flow that carries the input field. The solution after solving the
    //! equation for given time-step \p dt should be stored in scalar field
    //! \p output. The boundary interface is given by a signed-distance field.
    //! The field is negative inside the boundary. By default, a constant field
    //! with max double value (kMaxD) is used, meaning no boundary.
    //!
    //! \param input Input scalar grid.
    //! \param flow Vector field that advects the input field.
    //! \param dt Time-step for the advection.
    //! \param output Output scalar grid.
    //! \param boundarySdf Boundary interface defined by signed-distance
    //!     field.
    //!
    void advect(const ScalarGrid2& input, const VectorField2& flow, double dt,
                ScalarGrid2* output, const ScalarField2& boundarySdf =
                                         ConstantScalarField2(kMaxD)) final;

    //!
    //! \brief Computes semi-Lagrangian for given collocated vector grid.
    //!
    //! This function computes semi-Lagrangian method to solve advection
    //! equation for given collocated vector grid \p input and underlying vector
    //! field \p flow that carries the input field. The solution after solving
    //! the equation for given time-step \p dt should be stored in scalar field
    //! \p output. The boundary interface is given by a signed-distance field.
    //! The field is negative inside the boundary. By default, a constant field
    //! with max double value (kMaxD) is used, meaning no boundary.
    //!
    //! \param input Input vector grid.
    //! \param flow Vector field that advects the input field.
    //! \param dt Time-step for the advection.
    //! \param output Output vector grid.
    //! \param boundarySdf Boundary interface defined by signed-distance
    //!     field.
    //!
    void advect(
        const CollocatedVectorGrid2& input, const VectorField2& flow, double dt,
        CollocatedVectorGrid2* output,
        const ScalarField2& boundarySdf = ConstantScalarField2(kMaxD)) final;

    //!
    //! \brief Computes semi-Lagrangian for given face-centered vector grid.
    //!
    //! This function computes semi-Lagrangian method to solve advection
    //! equation for given face-centered vector grid \p input and underlying
    //! vector field \p flow that carries the input field. The solution after
    //! solving the equation for given time-step \p dt should be stored in
    //! vector field \p output. The boundary interface is given by a
    //! signed-distance field. The field is negative inside the boundary. By
    //! default, a constant field with max double value (kMaxD) is used, meaning
    //! no boundary.
    //!
    //! \param input Input vector grid.
    //! \param flow Vector field that advects the input field.
    //! \param dt Time-step for the advection.
    //! \param output Output vector grid.
    //! \param boundarySdf Boundary interface defined by signed-distance
    //!     field.
    //!
    void advect(
        const FaceCenteredGrid2& input, const VectorField2& flow, double dt,
        FaceCenteredGrid2* output,
        const ScalarField2& boundarySdf = ConstantScalarField2(kMaxD)) final;

 protected:
    //!
    //! \brief Returns spatial interpolation function object for given scalar
    //! grid.
    //!
    //! This function returns spatial interpolation function (sampler) for given
    //! scalar grid \p input. By default, this function returns linear
    //! interpolation function. Override this function to have custom
    //! interpolation for semi-Lagrangian process.
    //!
    virtual std::function<double(const Vector2D&)> getScalarSamplerFunc(
        const ScalarGrid2& input) const;

    //!
    //! \brief Returns spatial interpolation function object for given
    //! collocated vector grid.
    //!
    //! This function returns spatial interpolation function (sampler) for given
    //! collocated vector grid \p input. By default, this function returns
    //! linear interpolation function. Override this function to have custom
    //! interpolation for semi-Lagrangian process.
    //!
    virtual std::function<Vector2D(const Vector2D&)> getVectorSamplerFunc(
        const CollocatedVectorGrid2& input) const;

    //!
    //! \brief Returns spatial interpolation function object for given
    //! face-centered vector grid.
    //!
    //! This function returns spatial interpolation function (sampler) for given
    //! face-centered vector grid \p input. By default, this function returns
    //! linear interpolation function. Override this function to have custom
    //! interpolation for semi-Lagrangian process.
    //!
    virtual std::function<Vector2D(const Vector2D&)> getVectorSamplerFunc(
        const FaceCenteredGrid2& input) const;

 private:
    Vector2D backTrace(const VectorField2& flow, double dt, double h,
                       const Vector2D& pt0, const ScalarField2& boundarySdf);
};

typedef std::shared_ptr<SemiLagrangian2> SemiLagrangian2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SEMI_LAGRANGIAN2_H_

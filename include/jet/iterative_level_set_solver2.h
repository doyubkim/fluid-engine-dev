// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_ITERATIVE_LEVEL_SET_SOLVER2_H_
#define INCLUDE_JET_ITERATIVE_LEVEL_SET_SOLVER2_H_

#include <jet/level_set_solver2.h>

namespace jet {

//!
//! \brief Abstract base class for 2-D PDE-based iterative level set solver.
//!
//! This class provides infrastructure for 2-D PDE-based iterative level set
//! solver. Internally, the class implements upwind-style wave propagation and
//! the inheriting classes must provide a way to compute the derivatives for
//! given grid points.
//!
//! \see Osher, Stanley, and Ronald Fedkiw. Level set methods and dynamic
//!     implicit surfaces. Vol. 153. Springer Science & Business Media, 2006.
//!
class IterativeLevelSetSolver2 : public LevelSetSolver2 {
 public:
    //! Default constructor.
    IterativeLevelSetSolver2();

    //! Default destructor.
    virtual ~IterativeLevelSetSolver2();

    //!
    //! Reinitializes given scalar field to signed-distance field.
    //!
    //! \param inputSdf Input signed-distance field which can be distorted.
    //! \param maxDistance Max range of reinitialization.
    //! \param outputSdf Output signed-distance field.
    //!
    void reinitialize(const ScalarGrid2& inputSdf, double maxDistance,
                      ScalarGrid2* outputSdf) override;

    //!
    //! Extrapolates given scalar field from negative to positive SDF region.
    //!
    //! \param input Input scalar field to be extrapolated.
    //! \param sdf Reference signed-distance field.
    //! \param maxDistance Max range of extrapolation.
    //! \param output Output scalar field.
    //!
    void extrapolate(const ScalarGrid2& input, const ScalarField2& sdf,
                     double maxDistance, ScalarGrid2* output) override;

    //!
    //! Extrapolates given collocated vector field from negative to positive SDF
    //! region.
    //!
    //! \param input Input collocated vector field to be extrapolated.
    //! \param sdf Reference signed-distance field.
    //! \param maxDistance Max range of extrapolation.
    //! \param output Output collocated vector field.
    //!
    void extrapolate(const CollocatedVectorGrid2& input,
                     const ScalarField2& sdf, double maxDistance,
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
    void extrapolate(const FaceCenteredGrid2& input, const ScalarField2& sdf,
                     double maxDistance, FaceCenteredGrid2* output) override;

    //! Returns the maximum CFL limit.
    double maxCfl() const;

    //!
    //! \brief Sets the maximum CFL limit.
    //!
    //! This function sets the maximum CFL limit for the internal upwind-style
    //! PDE calculation. The negative input will be clamped to 0.
    //!
    void setMaxCfl(double newMaxCfl);

 protected:
    //! Computes the derivatives for given grid point.
    virtual void getDerivatives(ConstArrayAccessor2<double> grid,
                                const Vector2D& gridSpacing, size_t i, size_t j,
                                std::array<double, 2>* dx,
                                std::array<double, 2>* dy) const = 0;

 private:
    double _maxCfl = 0.5;

    void extrapolate(const ConstArrayAccessor2<double>& input,
                     const ConstArrayAccessor2<double>& sdf,
                     const Vector2D& gridSpacing, double maxDistance,
                     ArrayAccessor2<double> output);

    static unsigned int distanceToNumberOfIterations(double distance,
                                                     double dtau);

    static double sign(const ConstArrayAccessor2<double>& sdf,
                       const Vector2D& gridSpacing, size_t i, size_t j);

    double pseudoTimeStep(ConstArrayAccessor2<double> sdf,
                          const Vector2D& gridSpacing);
};

typedef std::shared_ptr<IterativeLevelSetSolver2> IterativeLevelSetSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_ITERATIVE_LEVEL_SET_SOLVER2_H_

// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ITERATIVE_LEVEL_SET_SOLVER2_H_
#define INCLUDE_JET_ITERATIVE_LEVEL_SET_SOLVER2_H_

#include <jet/level_set_solver2.h>

namespace jet {

class IterativeLevelSetSolver2 : public LevelSetSolver2 {
 public:
    IterativeLevelSetSolver2();

    virtual ~IterativeLevelSetSolver2();

    void reinitialize(
        const ScalarGrid2& inputSdf,
        double maxDistance,
        ScalarGrid2* outputSdf) override;

    void extrapolate(
        const ScalarGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        ScalarGrid2* output) override;

    void extrapolate(
        const CollocatedVectorGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        CollocatedVectorGrid2* output) override;

    void extrapolate(
        const FaceCenteredGrid2& input,
        const ScalarField2& sdf,
        double maxDistance,
        FaceCenteredGrid2* output) override;

    double maxCfl() const;

    void setMaxCfl(double newMaxCfl);

 protected:
    void extrapolate(
        const ConstArrayAccessor2<double>& input,
        const ConstArrayAccessor2<double>& sdf,
        const Vector2D& gridSpacing,
        double maxDistance,
        ArrayAccessor2<double> output);

    static unsigned int distanceToNumberOfIterations(
        double distance,
        double dtau);

    static double sign(
        const ConstArrayAccessor2<double>& sdf,
        const Vector2D& gridSpacing,
        size_t i,
        size_t j);

    double pseudoTimeStep(
        ConstArrayAccessor2<double> sdf,
        const Vector2D& gridSpacing);

    virtual void getDerivatives(
        ConstArrayAccessor2<double> grid,
        const Vector2D& gridSpacing,
        size_t i,
        size_t j,
        std::array<double, 2>* dx,
        std::array<double, 2>* dy) const = 0;

 private:
    double _maxCfl = 0.5;
};

}  // namespace jet

#endif  // INCLUDE_JET_ITERATIVE_LEVEL_SET_SOLVER2_H_

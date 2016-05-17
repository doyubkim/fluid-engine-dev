// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ITERATIVE_LEVEL_SET_SOLVER3_H_
#define INCLUDE_JET_ITERATIVE_LEVEL_SET_SOLVER3_H_

#include <jet/level_set_solver3.h>

namespace jet {

class IterativeLevelSetSolver3 : public LevelSetSolver3 {
 public:
    IterativeLevelSetSolver3();

    virtual ~IterativeLevelSetSolver3();

    void reinitialize(
        const ScalarGrid3& inputSdf,
        double maxDistance,
        ScalarGrid3* outputSdf) override;

    void extrapolate(
        const ScalarGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        ScalarGrid3* output) override;

    void extrapolate(
        const CollocatedVectorGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        CollocatedVectorGrid3* output) override;

    void extrapolate(
        const FaceCenteredGrid3& input,
        const ScalarField3& sdf,
        double maxDistance,
        FaceCenteredGrid3* output) override;

    double maxCfl() const;

    void setMaxCfl(double newMaxCfl);

 protected:
    void extrapolate(
        const ConstArrayAccessor3<double>& input,
        const ConstArrayAccessor3<double>& sdf,
        const Vector3D& gridSpacing,
        double maxDistance,
        ArrayAccessor3<double> output);

    static unsigned int distanceToNumberOfIterations(
        double distance,
        double dtau);

    static double sign(
        const ConstArrayAccessor3<double>& sdf,
        const Vector3D& gridSpacing,
        size_t i,
        size_t j,
        size_t k);

    double pseudoTimeStep(
        ConstArrayAccessor3<double> sdf,
        const Vector3D& gridSpacing);

    virtual void getDerivatives(
        ConstArrayAccessor3<double> grid,
        const Vector3D& gridSpacing,
        size_t i,
        size_t j,
        size_t k,
        std::array<double, 2>* dx,
        std::array<double, 2>* dy,
        std::array<double, 2>* dz) const = 0;

 private:
    double _maxCfl = 0.5;
};

}  // namespace jet

#endif  // INCLUDE_JET_ITERATIVE_LEVEL_SET_SOLVER3_H_

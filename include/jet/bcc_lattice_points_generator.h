// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_BCC_LATTICE_POINTS_GENERATOR_H_
#define INCLUDE_JET_BCC_LATTICE_POINTS_GENERATOR_H_

#include <jet/points_generator3.h>

namespace jet {

//! \brief Body-centered lattice points generator.
//!
//! \see http://en.wikipedia.org/wiki/Cubic_crystal_system
//!      http://mathworld.wolfram.com/CubicClosePacking.html
class BccLatticePointsGenerator final : public PointsGenerator3 {
 public:
    void forEachPoint(
        const BoundingBox3D& boundingBox,
        double spacing,
        const std::function<bool(const Vector3D&)>& callback) const override;
};

typedef std::shared_ptr<BccLatticePointsGenerator> BccLatticePointsGeneratorPtr;

}  // namespace jet

#endif  // INCLUDE_JET_BCC_LATTICE_POINTS_GENERATOR_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "sph_solver.h"
#include "pybind11_utils.h"

#include <jet/sph_solver2.h>
#include <jet/sph_solver3.h>

namespace py = pybind11;
using namespace jet;

void addSphSolver2(py::module& m) {
    py::class_<SphSolver2, SphSolver2Ptr, ParticleSystemSolver2>(m,
                                                                 "SphSolver2",
                                                                 R"pbdoc(
        2-D SPH solver.

        This class implements 2-D SPH solver. The main pressure solver is based on
        equation-of-state (EOS).
        - See M{\"u}ller et al., Particle-based fluid simulation for interactive
             applications, SCA 2003.
        - See M. Becker and M. Teschner, Weakly compressible SPH for free surface
             flows, SCA 2007.
        - See Adams and Wicke, Meshless approximation methods and applications in
             physics based modeling and animation, Eurographics tutorials 2009.
        )pbdoc")
        .def(py::init<double, double, double>(),
             R"pbdoc(
             Constructs a solver with target density, spacing, and relative kernel
             radius.
             )pbdoc",
             py::arg("targetDensity") = kWaterDensity,
             py::arg("targetSpacing") = 0.1,
             py::arg("relativeKernelRadius") = 1.8)
        .def_property("eosExponent", &SphSolver2::eosExponent,
                      &SphSolver2::setEosExponent,
                      R"pbdoc(
             The exponent part of the equation-of-state.

             This function sets the exponent part of the equation-of-state.
             The value must be greater than 1.0, and smaller inputs will be clamped.
             Default is 7.
             )pbdoc")
        .def_property("negativePressureScale",
                      &SphSolver2::negativePressureScale,
                      &SphSolver2::setNegativePressureScale,
                      R"pbdoc(
             The negative pressure scale.

             This function sets the negative pressure scale. By setting the number
             between 0 and 1, the solver will scale the effect of negative pressure
             which can prevent the clumping of the particles near the surface. Input
             value outside 0 and 1 will be clamped within the range. Default is 0.
             )pbdoc")
        .def_property("viscosityCoefficient", &SphSolver2::viscosityCoefficient,
                      &SphSolver2::setViscosityCoefficient,
                      R"pbdoc(
             The viscosity coefficient.
             )pbdoc")
        .def_property("pseudoViscosityCoefficient",
                      &SphSolver2::pseudoViscosityCoefficient,
                      &SphSolver2::setPseudoViscosityCoefficient,
                      R"pbdoc(
             The pseudo viscosity coefficient.

             This function sets the pseudo viscosity coefficient which applies
             additional pseudo-physical damping to the system. Default is 10.
             )pbdoc")
        .def_property("speedOfSound", &SphSolver2::speedOfSound,
                      &SphSolver2::setSpeedOfSound,
                      R"pbdoc(
             The speed of sound.

             This function sets the speed of sound which affects the stiffness of the
             EOS and the time-step size. Higher value will make EOS stiffer and the
             time-step smaller. The input value must be higher than 0.0.
             )pbdoc")
        .def_property("timeStepLimitScale", &SphSolver2::timeStepLimitScale,
                      &SphSolver2::setTimeStepLimitScale,
                      R"pbdoc(
             Multiplier that scales the max allowed time-step.

             This function returns the multiplier that scales the max allowed
             time-step. When the scale is 1.0, the time-step is bounded by the speed
             of sound and max acceleration.
             )pbdoc")
        .def_property_readonly("sphSystemData", &SphSolver2::sphSystemData,
                               R"pbdoc(
             The SPH system data.
             )pbdoc");
}

void addSphSolver3(py::module& m) {
    py::class_<SphSolver3, SphSolver3Ptr, ParticleSystemSolver3>(m,
                                                                 "SphSolver3",
                                                                 R"pbdoc(
        3-D SPH solver.

        This class implements 3-D SPH solver. The main pressure solver is based on
        equation-of-state (EOS).
        - See M{\"u}ller et al., Particle-based fluid simulation for interactive
             applications, SCA 2003.
        - See M. Becker and M. Teschner, Weakly compressible SPH for free surface
             flows, SCA 2007.
        - See Adams and Wicke, Meshless approximation methods and applications in
             physics based modeling and animation, Eurographics tutorials 2009.
        )pbdoc")
        .def(py::init<double, double, double>(),
             R"pbdoc(
             Constructs a solver with target density, spacing, and relative kernel
             radius.
             )pbdoc",
             py::arg("targetDensity") = kWaterDensity,
             py::arg("targetSpacing") = 0.1,
             py::arg("relativeKernelRadius") = 1.8)
        .def_property("eosExponent", &SphSolver3::eosExponent,
                      &SphSolver3::setEosExponent,
                      R"pbdoc(
             The exponent part of the equation-of-state.

             This function sets the exponent part of the equation-of-state.
             The value must be greater than 1.0, and smaller inputs will be clamped.
             Default is 7.
             )pbdoc")
        .def_property("negativePressureScale",
                      &SphSolver3::negativePressureScale,
                      &SphSolver3::setNegativePressureScale,
                      R"pbdoc(
             The negative pressure scale.

             This function sets the negative pressure scale. By setting the number
             between 0 and 1, the solver will scale the effect of negative pressure
             which can prevent the clumping of the particles near the surface. Input
             value outside 0 and 1 will be clamped within the range. Default is 0.
             )pbdoc")
        .def_property("viscosityCoefficient", &SphSolver3::viscosityCoefficient,
                      &SphSolver3::setViscosityCoefficient,
                      R"pbdoc(
             The viscosity coefficient.
             )pbdoc")
        .def_property("pseudoViscosityCoefficient",
                      &SphSolver3::pseudoViscosityCoefficient,
                      &SphSolver3::setPseudoViscosityCoefficient,
                      R"pbdoc(
             The pseudo viscosity coefficient.

             This function sets the pseudo viscosity coefficient which applies
             additional pseudo-physical damping to the system. Default is 10.
             )pbdoc")
        .def_property("speedOfSound", &SphSolver3::speedOfSound,
                      &SphSolver3::setSpeedOfSound,
                      R"pbdoc(
             The speed of sound.

             This function sets the speed of sound which affects the stiffness of the
             EOS and the time-step size. Higher value will make EOS stiffer and the
             time-step smaller. The input value must be higher than 0.0.
             )pbdoc")
        .def_property("timeStepLimitScale", &SphSolver3::timeStepLimitScale,
                      &SphSolver3::setTimeStepLimitScale,
                      R"pbdoc(
             Multiplier that scales the max allowed time-step.

             This function returns the multiplier that scales the max allowed
             time-step. When the scale is 1.0, the time-step is bounded by the speed
             of sound and max acceleration.
             )pbdoc")
        .def_property_readonly("sphSystemData", &SphSolver3::sphSystemData,
                               R"pbdoc(
             The SPH system data.
             )pbdoc");
}

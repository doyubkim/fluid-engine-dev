// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "particle_system_solver.h"
#include "pybind11_utils.h"

#include <jet/particle_system_solver2.h>
#include <jet/particle_system_solver3.h>

namespace py = pybind11;
using namespace jet;

void addParticleSystemSolver2(py::module& m) {
    py::class_<ParticleSystemSolver2, ParticleSystemSolver2Ptr,
               PhysicsAnimation>(m, "ParticleSystemSolver2",
                                 R"pbdoc(
        Basic 2-D particle system solver.

        This class implements basic particle system solver. It includes gravity,
        air drag, and collision. But it does not compute particle-to-particle
        interaction. Thus, this solver is suitable for performing simple spray-like
        simulations with low computational cost. This class can be further extend
        to add more sophisticated simulations, such as SPH, to handle
        particle-to-particle intersection.
        )pbdoc")
        .def(py::init<double, double>(),
             R"pbdoc(
             Constructs a solver with particle parameters.

             Parameters
             ----------
             - radius : Radius of a particle in meters (default 1e-2).
             - mass : Mass of a particle in kg (default 1e-2).
             )pbdoc",
             py::arg("radius") = 1e-2, py::arg("mass") = 1e-2)
        .def_property("dragCoefficient",
                      &ParticleSystemSolver2::dragCoefficient,
                      &ParticleSystemSolver2::setDragCoefficient,
                      R"pbdoc(
             The drag coefficient.

             The drag coefficient controls the amount of air-drag. The coefficient
             should be a positive number and 0 means no drag force.
             )pbdoc")
        .def_property("restitutionCoefficient",
                      &ParticleSystemSolver2::restitutionCoefficient,
                      &ParticleSystemSolver2::setRestitutionCoefficient,
                      R"pbdoc(
             The restitution coefficient.

             The restitution coefficient controls the bouncy-ness of a particle when
             it hits a collider surface. The range of the coefficient should be 0 to
             1 -- 0 means no bounce back and 1 means perfect reflection.
             )pbdoc")
        .def_property("gravity", &ParticleSystemSolver2::gravity,
                      &ParticleSystemSolver2::setGravity,
                      R"pbdoc(
             The gravity.
             )pbdoc")
        .def_property_readonly("particleSystemData",
                               &ParticleSystemSolver2::particleSystemData,
                               R"pbdoc(
             The the particle system data.

             This property returns the particle system data. The data is created when
             this solver is constructed and also owned by the solver.
             )pbdoc")
        .def_property("collider", &ParticleSystemSolver2::collider,
                      &ParticleSystemSolver2::setCollider,
                      R"pbdoc(
             The collider.
             )pbdoc")
        .def_property("emitter", &ParticleSystemSolver2::emitter,
                      &ParticleSystemSolver2::setEmitter,
                      R"pbdoc(
             The emitter.
             )pbdoc")
        .def_property("wind", &ParticleSystemSolver2::wind,
                      &ParticleSystemSolver2::setWind,
                      R"pbdoc(
             The wind.

             Wind can be applied to the particle system by setting a vector field to
             the solver.
             )pbdoc");
}

void addParticleSystemSolver3(py::module& m) {
    py::class_<ParticleSystemSolver3, ParticleSystemSolver3Ptr,
               PhysicsAnimation>(m, "ParticleSystemSolver3",
                                 R"pbdoc(
        Basic 3-D particle system solver.

        This class implements basic particle system solver. It includes gravity,
        air drag, and collision. But it does not compute particle-to-particle
        interaction. Thus, this solver is suitable for performing simple spray-like
        simulations with low computational cost. This class can be further extend
        to add more sophisticated simulations, such as SPH, to handle
        particle-to-particle intersection.
        )pbdoc")
        .def(py::init<double, double>(),
             R"pbdoc(
             Constructs a solver with particle parameters.

             Parameters
             ----------
             - radius : Radius of a particle in meters (default 1e-3).
             - mass : Mass of a particle in kg (default 1e-3).
             )pbdoc",
             py::arg("radius") = 1e-3, py::arg("mass") = 1e-3)
        .def_property("dragCoefficient",
                      &ParticleSystemSolver3::dragCoefficient,
                      &ParticleSystemSolver3::setDragCoefficient,
                      R"pbdoc(
             The drag coefficient.

             The drag coefficient controls the amount of air-drag. The coefficient
             should be a positive number and 0 means no drag force.
             )pbdoc")
        .def_property("restitutionCoefficient",
                      &ParticleSystemSolver3::restitutionCoefficient,
                      &ParticleSystemSolver3::setRestitutionCoefficient,
                      R"pbdoc(
             The restitution coefficient.

             The restitution coefficient controls the bouncy-ness of a particle when
             it hits a collider surface. The range of the coefficient should be 0 to
             1 -- 0 means no bounce back and 1 means perfect reflection.
             )pbdoc")
        .def_property("gravity", &ParticleSystemSolver3::gravity,
                      &ParticleSystemSolver3::setGravity,
                      R"pbdoc(
             The gravity.
             )pbdoc")
        .def_property_readonly("particleSystemData",
                               &ParticleSystemSolver3::particleSystemData,
                               R"pbdoc(
             The the particle system data.

             This property returns the particle system data. The data is created when
             this solver is constructed and also owned by the solver.
             )pbdoc")
        .def_property("collider", &ParticleSystemSolver3::collider,
                      &ParticleSystemSolver3::setCollider,
                      R"pbdoc(
             The collider.
             )pbdoc")
        .def_property("emitter", &ParticleSystemSolver3::emitter,
                      &ParticleSystemSolver3::setEmitter,
                      R"pbdoc(
             The emitter.
             )pbdoc")
        .def_property("wind", &ParticleSystemSolver3::wind,
                      &ParticleSystemSolver3::setWind,
                      R"pbdoc(
             The wind.

             Wind can be applied to the particle system by setting a vector field to
             the solver.
             )pbdoc");
}

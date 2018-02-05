// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "particle_system_data.h"
#include "pybind11_utils.h"

#include <jet/particle_system_data2.h>
#include <jet/particle_system_data3.h>

namespace py = pybind11;
using namespace jet;

void addParticleSystemData2(py::module& m) {
    py::class_<ParticleSystemData2, ParticleSystemData2Ptr, Serializable>(
        m, "ParticleSystemData2", R"pbdoc(
        This class is the key data structure for storing particle system data. A
        single particle has position, velocity, and force attributes by default. But
        it can also have additional custom scalar or vector attributes.
        )pbdoc")
        .def(py::init<size_t>(),
             R"pbdoc(
             Constructs particle system data with given number of particles.
             )pbdoc",
             py::arg("numberOfParticles") = 0)
        .def("resize", &ParticleSystemData2::resize,
             R"pbdoc(
             Resizes the number of particles of the container.

             This function will resize internal containers to store newly given
             number of particles including custom data layers. However, this will
             invalidate neighbor searcher and neighbor lists. It is users
             responsibility to call ParticleSystemData2::buildNeighborSearcher and
             ParticleSystemData2::buildNeighborLists to refresh those data.

             Parameters
             ----------
             - newNumberOfParticles : New number of particles.
             )pbdoc",
             py::arg("newNumberOfParticles"))
        .def_property_readonly("numberOfParticles",
                               &ParticleSystemData2::numberOfParticles,
                               R"pbdoc(
             The number of particles.
             )pbdoc")
        .def("addScalarData", &ParticleSystemData2::addScalarData,
             R"pbdoc(
             Adds a scalar data layer and returns its index.

             This function adds a new scalar data layer to the system. It can be used
             for adding a scalar attribute, such as temperature, to the particles.

             Parameters
             ----------
             - initialVal : Initial value of the new scalar data.
             )pbdoc",
             py::arg("initialVal") = 0.0)
        .def("addVectorData",
             [](ParticleSystemData2& instance, py::object object) {
                 return instance.addVectorData(objectToVector2D(object));
             },
             R"pbdoc(
             Adds a vector data layer and returns its index.

             This function adds a new vector data layer to the system. It can be used
             for adding a vector attribute, such as vortex, to the particles.

             Parameters
             ----------
             - initialVal : Initial value of the new vector data.
             )pbdoc",
             py::arg("initialVal") = Vector2D())
        .def_property("radius", &ParticleSystemData2::radius,
                      &ParticleSystemData2::setRadius,
                      R"pbdoc(
             The radius of the particles.
             )pbdoc")
        .def_property("mass", &ParticleSystemData2::mass,
                      &ParticleSystemData2::setMass,
                      R"pbdoc(
             The mass of a particle.
             )pbdoc")
        .def_property_readonly(
            "positions",
            [](ParticleSystemData2& instance) { return instance.positions(); },
            R"pbdoc(
             Returns the position array (mutable).
             )pbdoc")
        .def_property_readonly(
            "velocities",
            [](ParticleSystemData2& instance) { return instance.velocities(); },
            R"pbdoc(
             Returns the velocity array (mutable).
             )pbdoc")
        .def_property_readonly(
            "forces",
            [](ParticleSystemData2& instance) { return instance.forces(); },
            R"pbdoc(
             Returns the force array (mutable).
             )pbdoc")
        .def("scalarDataAt",
             [](ParticleSystemData2& instance, size_t idx) {
                 return instance.scalarDataAt(idx);
             },
             R"pbdoc(
             Returns custom scalar data layer at given index (mutable).
             )pbdoc")
        .def("vectorDataAt",
             [](ParticleSystemData2& instance, size_t idx) {
                 return instance.vectorDataAt(idx);
             },
             R"pbdoc(
             Returns custom vector data layer at given index (mutable).
             )pbdoc")
        .def("addParticle",
             [](ParticleSystemData2& instance, py::object p, py::object v,
                py::object f) {
                 instance.addParticle(objectToVector2D(p), objectToVector2D(v),
                                      objectToVector2D(f));
             },
             R"pbdoc(
             Adds a particle to the data structure.

             This function will add a single particle to the data structure. For
             custom data layers, zeros will be assigned for new particles.
             However, this will invalidate neighbor searcher and neighbor lists. It
             is users responsibility to call
             ParticleSystemData2::buildNeighborSearcher and
             ParticleSystemData2::buildNeighborLists to refresh those data.

             Parameters
             ----------
             - newPosition : The new position.
             - newVelocity : The new velocity.
             - newForce    : The new force.
             )pbdoc",
             py::arg("newPosition"), py::arg("newVelocity") = Vector2D(),
             py::arg("newForce") = Vector2D())
        .def("addParticles",
             [](ParticleSystemData2& instance, py::list ps, py::list vs,
                py::list fs) {
                 if (vs.size() > 0 && vs.size() != ps.size()) {
                     throw std::invalid_argument(
                         "Wrong input size for velocities list.");
                 }
                 if (fs.size() > 0 && fs.size() != ps.size()) {
                     throw std::invalid_argument(
                         "Wrong input size for velocities list.");
                 }

                 Array1<Vector2D> positions;
                 Array1<Vector2D> velocities;
                 Array1<Vector2D> forces;
                 for (size_t i = 0; i < ps.size(); ++i) {
                     positions.append(objectToVector2D(ps[i]));
                     if (vs.size() > 0) {
                         velocities.append(objectToVector2D(vs[i]));
                     }
                     if (fs.size() > 0) {
                         forces.append(objectToVector2D(fs[i]));
                     }
                 }

                 instance.addParticles(positions.constAccessor(),
                                       velocities.constAccessor(),
                                       forces.constAccessor());
             },
             R"pbdoc(
             Adds particles to the data structure.

             This function will add particles to the data structure. For custom data
             layers, zeros will be assigned for new particles. However, this will
             invalidate neighbor searcher and neighbor lists. It is users
             responsibility to call ParticleSystemData2::buildNeighborSearcher and
             ParticleSystemData2::buildNeighborLists to refresh those data.

             Parameters
             ----------
             - newPositions  : The new positions.
             - newVelocities : The new velocities.
             - newForces     : The new forces.
             )pbdoc")
        .def_property("neighborSearcher",
                      &ParticleSystemData2::neighborSearcher,
                      &ParticleSystemData2::setNeighborSearcher,
                      R"pbdoc(
             The neighbor searcher.

             This property returns currently set neighbor searcher object. By
             default, PointParallelHashGridSearcher2 is used.
             )pbdoc")
        .def_property_readonly("neighborLists",
                               &ParticleSystemData2::neighborLists,
                               R"pbdoc(
             The neighbor lists.

             This property returns neighbor lists which is available after calling
             PointParallelHashGridSearcher2::buildNeighborLists. Each list stores
             indices of the neighbors.
             )pbdoc")
        .def("set",
             [](ParticleSystemData2& instance,
                const ParticleSystemData2Ptr& other) { instance.set(*other); },
             R"pbdoc(
             Copies from other particle system data.
             )pbdoc");
}

void addParticleSystemData3(py::module& m) {
    py::class_<ParticleSystemData3, ParticleSystemData3Ptr, Serializable>(
        m, "ParticleSystemData3", R"pbdoc(
        This class is the key data structure for storing particle system data. A
        single particle has position, velocity, and force attributes by default. But
        it can also have additional custom scalar or vector attributes.
        )pbdoc")
        .def(py::init<size_t>(),
             R"pbdoc(
             Constructs particle system data with given number of particles.
             )pbdoc",
             py::arg("numberOfParticles") = 0)
        .def("resize", &ParticleSystemData3::resize,
             R"pbdoc(
             Resizes the number of particles of the container.

             This function will resize internal containers to store newly given
             number of particles including custom data layers. However, this will
             invalidate neighbor searcher and neighbor lists. It is users
             responsibility to call ParticleSystemData3::buildNeighborSearcher and
             ParticleSystemData3::buildNeighborLists to refresh those data.

             Parameters
             ----------
             - newNumberOfParticles : New number of particles.
             )pbdoc",
             py::arg("newNumberOfParticles"))
        .def_property_readonly("numberOfParticles",
                               &ParticleSystemData3::numberOfParticles,
                               R"pbdoc(
             The number of particles.
             )pbdoc")
        .def("addScalarData", &ParticleSystemData3::addScalarData,
             R"pbdoc(
             Adds a scalar data layer and returns its index.

             This function adds a new scalar data layer to the system. It can be used
             for adding a scalar attribute, such as temperature, to the particles.

             Parameters
             ----------
             - initialVal : Initial value of the new scalar data.
             )pbdoc",
             py::arg("initialVal") = 0.0)
        .def("addVectorData",
             [](ParticleSystemData3& instance, py::object object) {
                 return instance.addVectorData(objectToVector3D(object));
             },
             R"pbdoc(
             Adds a vector data layer and returns its index.

             This function adds a new vector data layer to the system. It can be used
             for adding a vector attribute, such as vortex, to the particles.

             Parameters
             ----------
             - initialVal : Initial value of the new vector data.
             )pbdoc",
             py::arg("initialVal") = Vector3D())
        .def_property("radius", &ParticleSystemData3::radius,
                      &ParticleSystemData3::setRadius,
                      R"pbdoc(
             The radius of the particles.
             )pbdoc")
        .def_property("mass", &ParticleSystemData3::mass,
                      &ParticleSystemData3::setMass,
                      R"pbdoc(
             The mass of a particle.
             )pbdoc")
        .def_property_readonly(
            "positions",
            [](ParticleSystemData3& instance) { return instance.positions(); },
            R"pbdoc(
             Returns the position array (mutable).
             )pbdoc")
        .def_property_readonly(
            "velocities",
            [](ParticleSystemData3& instance) { return instance.velocities(); },
            R"pbdoc(
             Returns the velocity array (mutable).
             )pbdoc")
        .def_property_readonly(
            "forces",
            [](ParticleSystemData3& instance) { return instance.forces(); },
            R"pbdoc(
             Returns the force array (mutable).
             )pbdoc")
        .def("scalarDataAt",
             [](ParticleSystemData3& instance, size_t idx) {
                 return instance.scalarDataAt(idx);
             },
             R"pbdoc(
             Returns custom scalar data layer at given index (mutable).
             )pbdoc")
        .def("vectorDataAt",
             [](ParticleSystemData3& instance, size_t idx) {
                 return instance.vectorDataAt(idx);
             },
             R"pbdoc(
             Returns custom vector data layer at given index (mutable).
             )pbdoc")
        .def("addParticle",
             [](ParticleSystemData3& instance, py::object p, py::object v,
                py::object f) {
                 instance.addParticle(objectToVector3D(p), objectToVector3D(v),
                                      objectToVector3D(f));
             },
             R"pbdoc(
             Adds a particle to the data structure.

             This function will add a single particle to the data structure. For
             custom data layers, zeros will be assigned for new particles.
             However, this will invalidate neighbor searcher and neighbor lists. It
             is users responsibility to call
             ParticleSystemData3::buildNeighborSearcher and
             ParticleSystemData3::buildNeighborLists to refresh those data.

             Parameters
             ----------
             - newPosition : The new position.
             - newVelocity : The new velocity.
             - newForce    : The new force.
             )pbdoc",
             py::arg("newPosition"), py::arg("newVelocity") = Vector3D(),
             py::arg("newForce") = Vector3D())
        .def("addParticles",
             [](ParticleSystemData3& instance, py::list ps, py::list vs,
                py::list fs) {
                 if (vs.size() > 0 && vs.size() != ps.size()) {
                     throw std::invalid_argument(
                         "Wrong input size for velocities list.");
                 }
                 if (fs.size() > 0 && fs.size() != ps.size()) {
                     throw std::invalid_argument(
                         "Wrong input size for velocities list.");
                 }

                 Array1<Vector3D> positions;
                 Array1<Vector3D> velocities;
                 Array1<Vector3D> forces;
                 for (size_t i = 0; i < ps.size(); ++i) {
                     positions.append(objectToVector3D(ps[i]));
                     if (vs.size() > 0) {
                         velocities.append(objectToVector3D(vs[i]));
                     }
                     if (fs.size() > 0) {
                         forces.append(objectToVector3D(fs[i]));
                     }
                 }

                 instance.addParticles(positions.constAccessor(),
                                       velocities.constAccessor(),
                                       forces.constAccessor());
             },
             R"pbdoc(
             Adds particles to the data structure.

             This function will add particles to the data structure. For custom data
             layers, zeros will be assigned for new particles. However, this will
             invalidate neighbor searcher and neighbor lists. It is users
             responsibility to call ParticleSystemData3::buildNeighborSearcher and
             ParticleSystemData3::buildNeighborLists to refresh those data.

             Parameters
             ----------
             - newPositions  : The new positions.
             - newVelocities : The new velocities.
             - newForces     : The new forces.
             )pbdoc")
        .def_property("neighborSearcher",
                      &ParticleSystemData3::neighborSearcher,
                      &ParticleSystemData3::setNeighborSearcher,
                      R"pbdoc(
             The neighbor searcher.

             This property returns currently set neighbor searcher object. By
             default, PointParallelHashGridSearcher2 is used.
             )pbdoc")
        .def_property_readonly("neighborLists",
                               &ParticleSystemData3::neighborLists,
                               R"pbdoc(
             The neighbor lists.

             This property returns neighbor lists which is available after calling
             PointParallelHashGridSearcher2::buildNeighborLists. Each list stores
             indices of the neighbors.
             )pbdoc")
        .def("set",
             [](ParticleSystemData3& instance,
                const ParticleSystemData3Ptr& other) { instance.set(*other); },
             R"pbdoc(
             Copies from other particle system data.
             )pbdoc");
}

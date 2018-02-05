// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "sph_system_data.h"
#include "pybind11_utils.h"

#include <jet/sph_system_data2.h>
#include <jet/sph_system_data3.h>

namespace py = pybind11;
using namespace jet;

void addSphSystemData2(py::module& m) {
    py::class_<SphSystemData2, SphSystemData2Ptr, ParticleSystemData2>(
        m, "SphSystemData2", R"pbdoc(
        2-D SPH particle system data.

        This class extends ParticleSystemData2 to specialize the data model for SPH.
        It includes density and pressure array as a default particle attribute, and
        it also contains SPH utilities such as interpolation operator.
        )pbdoc")
        .def(py::init<size_t>(),
             R"pbdoc(
             Constructs SPH system data with given number of particles.
             )pbdoc",
             py::arg("numberOfParticles") = 0)
        .def_property_readonly(
            "densities",
            [](SphSystemData2& instance) { return instance.densities(); },
            R"pbdoc(
             The density array accessor.
             )pbdoc")
        .def_property_readonly(
            "pressures",
            [](SphSystemData2& instance) { return instance.pressures(); },
            R"pbdoc(
             The pressure array accessor.
             )pbdoc")
        .def("updateDensities", &SphSystemData2::updateDensities,
             R"pbdoc(
             Updates the density array with the latest particle positions.
             )pbdoc")
        .def_property("targetDensity", &SphSystemData2::targetDensity,
                      &SphSystemData2::setTargetDensity,
                      R"pbdoc(
             The target density of this particle system.
             )pbdoc")
        .def_property("targetSpacing", &SphSystemData2::targetSpacing,
                      &SphSystemData2::setTargetSpacing,
                      R"pbdoc(
             The target particle spacing in meters.

             Once this property has changed, hash grid and density should be
             updated using updateHashGrid() and updateDensities).
             )pbdoc")
        .def_property("relativeKernelRadius",
                      &SphSystemData2::relativeKernelRadius,
                      &SphSystemData2::setRelativeKernelRadius,
                      R"pbdoc(
             The relative kernel radius.

             The relative kernel radius compared to the target particle
             spacing (i.e. kernel radius / target spacing).
             Once this property has changed, hash grid and density should
             be updated using updateHashGrid() and updateDensities).
             )pbdoc")
        .def_property("kernelRadius", &SphSystemData2::kernelRadius,
                      &SphSystemData2::setKernelRadius,
                      R"pbdoc(
             The kernel radius in meters unit.

             Sets the absolute kernel radius compared to the target particle
             spacing (i.e. relative kernel radius * target spacing).
             Once this function is called, hash grid and density should
             be updated using updateHashGrid() and updateDensities).
             )pbdoc")
        .def("buildNeighborSearcher", &SphSystemData2::buildNeighborSearcher,
             R"pbdoc(
             Builds neighbor searcher with kernel radius.
             )pbdoc")
        .def("buildNeighborLists", &SphSystemData2::buildNeighborLists,
             R"pbdoc(
             Builds neighbor lists with kernel radius.
             )pbdoc")
        .def("set", &SphSystemData2::set,
             R"pbdoc(
             Copies from other SPH system data.
             )pbdoc");
}

void addSphSystemData3(py::module& m) {
    py::class_<SphSystemData3, SphSystemData3Ptr, ParticleSystemData3>(
        m, "SphSystemData3", R"pbdoc(
        3-D SPH particle system data.

        This class extends ParticleSystemData3 to specialize the data model for SPH.
        It includes density and pressure array as a default particle attribute, and
        it also contains SPH utilities such as interpolation operator.
        )pbdoc")
        .def(py::init<size_t>(),
             R"pbdoc(
             Constructs SPH system data with given number of particles.
             )pbdoc",
             py::arg("numberOfParticles") = 0)
        .def_property_readonly(
            "densities",
            [](SphSystemData3& instance) { return instance.densities(); },
            R"pbdoc(
             The density array accessor.
             )pbdoc")
        .def_property_readonly(
            "pressures",
            [](SphSystemData3& instance) { return instance.pressures(); },
            R"pbdoc(
             The pressure array accessor.
             )pbdoc")
        .def("updateDensities", &SphSystemData3::updateDensities,
             R"pbdoc(
             Updates the density array with the latest particle positions.
             )pbdoc")
        .def_property("targetDensity", &SphSystemData3::targetDensity,
                      &SphSystemData3::setTargetDensity,
                      R"pbdoc(
             The target density of this particle system.
             )pbdoc")
        .def_property("targetSpacing", &SphSystemData3::targetSpacing,
                      &SphSystemData3::setTargetSpacing,
                      R"pbdoc(
             The target particle spacing in meters.

             Once this property has changed, hash grid and density should be
             updated using updateHashGrid() and updateDensities).
             )pbdoc")
        .def_property("relativeKernelRadius",
                      &SphSystemData3::relativeKernelRadius,
                      &SphSystemData3::setRelativeKernelRadius,
                      R"pbdoc(
             The relative kernel radius.

             The relative kernel radius compared to the target particle
             spacing (i.e. kernel radius / target spacing).
             Once this property has changed, hash grid and density should
             be updated using updateHashGrid() and updateDensities).
             )pbdoc")
        .def_property("kernelRadius", &SphSystemData3::kernelRadius,
                      &SphSystemData3::setKernelRadius,
                      R"pbdoc(
             The kernel radius in meters unit.

             Sets the absolute kernel radius compared to the target particle
             spacing (i.e. relative kernel radius * target spacing).
             Once this function is called, hash grid and density should
             be updated using updateHashGrid() and updateDensities).
             )pbdoc")
        .def("buildNeighborSearcher", &SphSystemData3::buildNeighborSearcher,
             R"pbdoc(
             Builds neighbor searcher with kernel radius.
             )pbdoc")
        .def("buildNeighborLists", &SphSystemData3::buildNeighborLists,
             R"pbdoc(
             Builds neighbor lists with kernel radius.
             )pbdoc")
        .def("set", &SphSystemData3::set,
             R"pbdoc(
             Copies from other SPH system data.
             )pbdoc");
}

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "surface_set.h"
#include "pybind11_utils.h"

#include <jet/surface_set2.h>
#include <jet/surface_set3.h>

namespace py = pybind11;
using namespace jet;

void addSurfaceSet2(py::module& m) {
    py::class_<SurfaceSet2, SurfaceSet2Ptr, Surface2>(m, "SurfaceSet2",
                                                      R"pbdoc(
         2-D surface set.

         This class represents 2-D surface set which extends Surface2 by overriding
         surface-related queries. This is class can hold a collection of other
         surface instances.
         )pbdoc")
        .def("__init__",
             [](SurfaceSet2& instance, py::list others,
                const Transform2& transform, bool isNormalFlipped) {
                 std::vector<Surface2Ptr> others_;
                 for (size_t i = 0; i < others.size(); ++i) {
                     others_.push_back(others[i].cast<Surface2Ptr>());
                 }
                 new (&instance)
                     SurfaceSet2(others_, transform, isNormalFlipped);
             },
             R"pbdoc(
             Constructs with a list of other surfaces.
             )pbdoc",
             py::arg("others") = py::list(),
             py::arg("transform") = Transform2(),
             py::arg("isNormalFlipped") = false)
        .def("numberOfSurfaces", &SurfaceSet2::numberOfSurfaces,
             R"pbdoc(
             Returns the number of surfaces.
             )pbdoc")
        .def("surfaceAt", &SurfaceSet2::surfaceAt,
             R"pbdoc(
             Returns the `i`-th surface.
             )pbdoc",
             py::arg("i"))
        .def("addSurface", &SurfaceSet2::addSurface,
             R"pbdoc(
             Adds a surface instance.
             )pbdoc",
             py::arg("surface"));
}

void addSurfaceSet3(py::module& m) {
    py::class_<SurfaceSet3, SurfaceSet3Ptr, Surface3>(m, "SurfaceSet3",
                                                      R"pbdoc(
         3-D surface set.

         This class represents 3-D surface set which extends Surface3 by overriding
         surface-related queries. This is class can hold a collection of other
         surface instances.
         )pbdoc")
        .def("__init__",
             [](SurfaceSet3& instance, py::list others,
                const Transform3& transform, bool isNormalFlipped) {
                 std::vector<Surface3Ptr> others_;
                 for (size_t i = 0; i < others.size(); ++i) {
                     others_.push_back(others[i].cast<Surface3Ptr>());
                 }
                 new (&instance)
                     SurfaceSet3(others_, transform, isNormalFlipped);
             },
             R"pbdoc(
             Constructs with a list of other surfaces.
             )pbdoc",
             py::arg("others") = py::list(),
             py::arg("transform") = Transform3(),
             py::arg("isNormalFlipped") = false)
        .def("numberOfSurfaces", &SurfaceSet3::numberOfSurfaces,
             R"pbdoc(
             Returns the number of surfaces.
             )pbdoc")
        .def("surfaceAt", &SurfaceSet3::surfaceAt,
             R"pbdoc(
             Returns the `i`-th surface.
             )pbdoc",
             py::arg("i"))
        .def("addSurface", &SurfaceSet3::addSurface,
             R"pbdoc(
             Adds a surface instance.
             )pbdoc",
             py::arg("surface"));
}

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "implicit_triangle_mesh.h"
#include "pybind11_utils.h"

#include <jet/implicit_triangle_mesh3.h>

namespace py = pybind11;
using namespace jet;

void addImplicitTriangleMesh3(pybind11::module& m) {
    py::class_<ImplicitTriangleMesh3, ImplicitTriangleMesh3Ptr,
               ImplicitSurface3>(m, "ImplicitTriangleMesh3",
                                 R"pbdoc(
         TriangleMesh3 to ImplicitSurface3 converter.

         This class builds signed-distance field for given TriangleMesh3 instance so
         that it can be used as an ImplicitSurface3 instance. The mesh is discretize
         into a regular grid and the signed-distance is measured at each grid point.
         Thus, there is a sampling error and its magnitude depends on the grid
         resolution.
         )pbdoc")
        // CTOR
        .def(py::init<TriangleMesh3Ptr, size_t, double, Transform3, bool>(),
             R"pbdoc(
             Constructs an ImplicitSurface3 with mesh and other grid parameters.
             )pbdoc",
             py::arg("mesh"), py::arg("resolutionX") = 32,
             py::arg("margin") = 0.2, py::arg("transform") = Transform3(),
             py::arg("isNormalFlipped") = false)
        .def_property_readonly("grid", &ImplicitTriangleMesh3::grid,
                               R"pbdoc(The grid data.)pbdoc");
}

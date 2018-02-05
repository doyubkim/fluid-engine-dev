// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "surface_to_implicit.h"
#include "pybind11_utils.h"

#include <jet/surface_to_implicit2.h>
#include <jet/surface_to_implicit3.h>

namespace py = pybind11;
using namespace jet;

void addSurfaceToImplicit2(py::module& m) {
    py::class_<SurfaceToImplicit2, SurfaceToImplicit2Ptr, ImplicitSurface2>(
        m, "SurfaceToImplicit2",
        R"pbdoc(
         2-D implicit surface wrapper for generic Surface2 instance.

         This class represents 2-D implicit surface that converts Surface2 instance
         to an ImplicitSurface2 object. The conversion is made by evaluating closest
         point and normal from a given point for the given (explicit) surface. Thus,
         this conversion won't work for every single surfaces, especially
         TriangleMesh2. To use TriangleMesh2 as an ImplicitSurface2 instance,
         please take a look at ImplicitTriangleMesh2. Use this class only
         for the basic primitives such as Sphere2 or Box2.
         )pbdoc")
        .def(py::init<Surface2Ptr, Transform2, bool>(),
             R"pbdoc(
             Constructs an instance with generic Surface2 instance.
             )pbdoc",
             py::arg("surface"), py::arg("transform") = Transform2(),
             py::arg("isNormalFlipped") = false)
        .def_property_readonly("surface", &SurfaceToImplicit2::surface,
                               R"pbdoc(
             The raw surface instance.
             )pbdoc");
}

void addSurfaceToImplicit3(py::module& m) {
    py::class_<SurfaceToImplicit3, SurfaceToImplicit3Ptr, ImplicitSurface3>(
        m, "SurfaceToImplicit3",
        R"pbdoc(
         3-D implicit surface wrapper for generic Surface3 instance.

         This class represents 3-D implicit surface that converts Surface3 instance
         to an ImplicitSurface3 object. The conversion is made by evaluating closest
         point and normal from a given point for the given (explicit) surface. Thus,
         this conversion won't work for every single surfaces, especially
         TriangleMesh3. To use TriangleMesh3 as an ImplicitSurface3 instance,
         please take a look at ImplicitTriangleMesh3. Use this class only
         for the basic primitives such as Sphere3 or Box3.
         )pbdoc")
        .def(py::init<Surface3Ptr, Transform3, bool>(),
             R"pbdoc(
             Constructs an instance with generic Surface3 instance.
             )pbdoc",
             py::arg("surface"), py::arg("transform") = Transform3(),
             py::arg("isNormalFlipped") = false)
        .def_property_readonly("surface", &SurfaceToImplicit3::surface,
                               R"pbdoc(
             The raw surface instance.
             )pbdoc");
}

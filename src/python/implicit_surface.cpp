// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "implicit_surface.h"
#include "pybind11_utils.h"

#include <jet/implicit_surface2.h>
#include <jet/implicit_surface3.h>

namespace py = pybind11;
using namespace jet;

void addImplicitSurface3(py::module& m) {
    py::class_<ImplicitSurface2, ImplicitSurface2Ptr, Surface2>(m, "ImplicitSurface2");
    py::class_<ImplicitSurface3, ImplicitSurface3Ptr, Surface3>(m, "ImplicitSurface3");
}

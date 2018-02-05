// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "surface.h"
#include "pybind11_utils.h"

#include <jet/surface2.h>
#include <jet/surface3.h>

namespace py = pybind11;
using namespace jet;

void addSurface2(py::module& m) {
    py::class_<Surface2, Surface2Ptr>(m, "Surface2")
            .def_readwrite("transform", &Surface2::transform)
            .def_readwrite("isNormalFlipped", &Surface2::isNormalFlipped);
}

void addSurface3(py::module& m) {
    py::class_<Surface3, Surface3Ptr>(m, "Surface3")
        .def_readwrite("transform", &Surface3::transform)
        .def_readwrite("isNormalFlipped", &Surface3::isNormalFlipped);
}

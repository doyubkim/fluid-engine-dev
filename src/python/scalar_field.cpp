// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "scalar_field.h"
#include "pybind11_utils.h"

#include <jet/scalar_field2.h>
#include <jet/scalar_field3.h>

namespace py = pybind11;
using namespace jet;

void addScalarField2(py::module& m) {
    py::class_<ScalarField2, ScalarField2Ptr, Field2>(
        m, "ScalarField2",
        R"pbdoc(Abstract base class for 2-D scalar field.)pbdoc");
}

void addScalarField3(py::module& m) {
    py::class_<ScalarField3, ScalarField3Ptr, Field3>(
        m, "ScalarField3",
        R"pbdoc(Abstract base class for 3-D scalar field.)pbdoc");
}

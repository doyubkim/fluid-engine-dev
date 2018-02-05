// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "vector_field.h"
#include "pybind11_utils.h"

#include <jet/vector_field2.h>
#include <jet/vector_field3.h>

namespace py = pybind11;
using namespace jet;

void addVectorField2(py::module& m) {
    py::class_<VectorField2, VectorField2Ptr, Field2>(
        m, "VectorField2",
        R"pbdoc(Abstract base class for 2-D vector field.)pbdoc");
}

void addVectorField3(py::module& m) {
    py::class_<VectorField3, VectorField3Ptr, Field3>(
        m, "VectorField3",
        R"pbdoc(Abstract base class for 3-D vector field.)pbdoc");
}

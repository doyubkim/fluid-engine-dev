// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "constant_scalar_field.h"
#include "pybind11_utils.h"

#include <jet/constant_scalar_field2.h>
#include <jet/constant_scalar_field3.h>

namespace py = pybind11;
using namespace jet;

void addConstantScalarField2(py::module& m) {
    py::class_<ConstantScalarField2, ConstantScalarField2Ptr, ScalarField2>(
        m, "ConstantScalarField2",
        R"pbdoc(2-D constant scalar field.)pbdoc")
        .def(
            py::init<double>(),
            R"pbdoc(Constructs a constant scalar field with given `value`.)pbdoc",
            py::arg("value"))
        .def("sample",
             [](const ConstantScalarField2& instance, py::object obj) {
                 return instance.sample(objectToVector2D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("sampler",
             [](const ConstantScalarField2& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}

void addConstantScalarField3(py::module& m) {
    py::class_<ConstantScalarField3, ConstantScalarField3Ptr, ScalarField3>(
        m, "ConstantScalarField3",
        R"pbdoc(3-D constant scalar field.)pbdoc")
        .def(
            py::init<double>(),
            R"pbdoc(Constructs a constant scalar field with given `value`.)pbdoc",
            py::arg("value"))
        .def("sample",
             [](const ConstantScalarField3& instance, py::object obj) {
                 return instance.sample(objectToVector3D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("sampler",
             [](const ConstantScalarField3& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}

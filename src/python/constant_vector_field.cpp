// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "constant_vector_field.h"
#include "pybind11_utils.h"

#include <jet/constant_vector_field2.h>
#include <jet/constant_vector_field3.h>

namespace py = pybind11;
using namespace jet;

void addConstantVectorField2(py::module& m) {
    py::class_<ConstantVectorField2, ConstantVectorField2Ptr, VectorField2>(
        m, "ConstantVectorField2",
        R"pbdoc(2-D constant vector field.)pbdoc")
        .def(
            py::init<Vector2D>(),
            R"pbdoc(Constructs a constant vector field with given `value`.)pbdoc",
            py::arg("value"))
        .def("sample",
             [](const ConstantVectorField2& instance, py::object obj) {
                 return instance.sample(objectToVector2D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("sampler",
             [](const ConstantVectorField2& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}

void addConstantVectorField3(py::module& m) {
    py::class_<ConstantVectorField3, ConstantVectorField3Ptr, VectorField3>(
        m, "ConstantVectorField3",
        R"pbdoc(3-D constant vector field.)pbdoc")
        .def(
            py::init<Vector3D>(),
            R"pbdoc(Constructs a constant vector field with given `value`.)pbdoc",
            py::arg("value"))
        .def("sample",
             [](const ConstantVectorField3& instance, py::object obj) {
                 return instance.sample(objectToVector3D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("sampler",
             [](const ConstantVectorField3& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}

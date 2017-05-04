// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "array_accessor2.h"
#include "pybind11_utils.h"

#include <jet/array_accessor2.h>
#include <jet/vector2.h>
#include <jet/vector3.h>
#include <jet/vector4.h>

namespace py = pybind11;
using namespace jet;

template <typename T>
void addArrayAccessor2Scalar(py::module m, const char* name) {
    py::class_<ArrayAccessor2<T>>(m, name, py::buffer_protocol())
        .def_buffer([](ArrayAccessor2<T>& m) -> py::buffer_info {
            return py::buffer_info(m.data(), sizeof(T),
                                   py::format_descriptor<T>::format(), 2,
                                   {m.height(), m.width()}, {sizeof(T) * m.width(), sizeof(T)});
        });
};

void addArrayAccessor2(py::module& m) {
    addArrayAccessor2Scalar<double>(m, "ArrayAccessor2D");
    addArrayAccessor2Scalar<float>(m, "ArrayAccessor2F");
}

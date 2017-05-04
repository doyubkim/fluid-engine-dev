// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "array_accessor3.h"
#include "pybind11_utils.h"

#include <jet/array_accessor3.h>
#include <jet/vector2.h>
#include <jet/vector3.h>
#include <jet/vector4.h>

namespace py = pybind11;
using namespace jet;

template <typename T>
void addArrayAccessor3Scalar(py::module m, const char* name) {
    py::class_<ArrayAccessor3<T>>(m, name, py::buffer_protocol())
        .def_buffer([](ArrayAccessor3<T>& m) -> py::buffer_info {
            return py::buffer_info(m.data(), sizeof(T),
                                   py::format_descriptor<T>::format(), 3,
                                   {m.depth(), m.height(), m.width()},
                                   {sizeof(T) * m.width() * m.height(),
                                    sizeof(T) * m.width(), sizeof(T)});
        });
};

void addArrayAccessor3(py::module& m) {
    addArrayAccessor3Scalar<double>(m, "ArrayAccessor3D");
    addArrayAccessor3Scalar<float>(m, "ArrayAccessor3F");
}

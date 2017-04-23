// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "array_accessor1.h"
#include "pybind11_utils.h"

#include <jet/array_accessor1.h>
#include <jet/vector2.h>
#include <jet/vector3.h>
#include <jet/vector4.h>

namespace py = pybind11;
using namespace jet;

template <typename T>
void addArrayAccessor1Scalar(py::module m, const char* name) {
    py::class_<ArrayAccessor1<T>>(m, name, py::buffer_protocol())
        .def_buffer([](ArrayAccessor1<T>& m) -> py::buffer_info {
            return py::buffer_info(m.data(), sizeof(T),
                                   py::format_descriptor<T>::format(), 1,
                                   {m.size()}, {sizeof(T)});
        });
};

template <typename T, size_t N>
void addArrayAccessor1Vector(py::module m, const char* name) {
    py::class_<ArrayAccessor1<Vector<T, N>>>(m, name, py::buffer_protocol())
        .def_buffer([](ArrayAccessor1<Vector<T, N>>& m) -> py::buffer_info {
            return py::buffer_info(m.data(), sizeof(T),
                                   py::format_descriptor<T>::format(), 2,
                                   {m.size(), N}, {sizeof(T) * N, sizeof(T)});
        });
};

void addArrayAccessor1(py::module& m) {
    addArrayAccessor1Scalar<double>(m, "ArrayAccessor1D");
    addArrayAccessor1Vector<double, 2>(m, "ArrayAccessor1Vector2D");
    addArrayAccessor1Vector<double, 3>(m, "ArrayAccessor1Vector3D");
    addArrayAccessor1Vector<double, 4>(m, "ArrayAccessor1Vector4D");
    addArrayAccessor1Scalar<float>(m, "ArrayAccessor1F");
    addArrayAccessor1Vector<float, 2>(m, "ArrayAccessor1Vector2F");
    addArrayAccessor1Vector<float, 3>(m, "ArrayAccessor1Vector3F");
    addArrayAccessor1Vector<float, 4>(m, "ArrayAccessor1Vector4F");
}

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "array_view.h"
#include "pybind11_utils.h"

#include <jet/array_view.h>
#include <jet/matrix.h>

namespace py = pybind11;
using namespace jet;

template <typename T>
void addArrayView1Scalar(py::module m, const char* name) {
    py::class_<ArrayView1<T>>(m, name, py::buffer_protocol())
        .def_buffer([](ArrayView1<T>& m) -> py::buffer_info {
            return py::buffer_info(m.data(), sizeof(T),
                                   py::format_descriptor<T>::format(), 1,
                                   {m.length()}, {sizeof(T)});
        });
};

template <typename T, size_t N>
void addArrayView1Vector(py::module m, const char* name) {
    py::class_<ArrayView1<Vector<T, N>>>(m, name, py::buffer_protocol())
        .def_buffer([](ArrayView1<Vector<T, N>>& m) -> py::buffer_info {
            return py::buffer_info(m.data(), sizeof(T),
                                   py::format_descriptor<T>::format(), 2,
                                   {m.length(), N}, {sizeof(T) * N, sizeof(T)});
        });
};

template <typename T>
void addArrayView2Scalar(py::module m, const char* name) {
    py::class_<ArrayView2<T>>(m, name, py::buffer_protocol())
        .def_buffer([](ArrayView2<T>& m) -> py::buffer_info {
            return py::buffer_info(
                m.data(), sizeof(T), py::format_descriptor<T>::format(), 2,
                {m.height(), m.width()}, {sizeof(T) * m.width(), sizeof(T)});
        });
};

template <typename T>
void addArrayView3Scalar(py::module m, const char* name) {
    py::class_<ArrayView3<T>>(m, name, py::buffer_protocol())
        .def_buffer([](ArrayView3<T>& m) -> py::buffer_info {
            return py::buffer_info(m.data(), sizeof(T),
                                   py::format_descriptor<T>::format(), 3,
                                   {m.depth(), m.height(), m.width()},
                                   {sizeof(T) * m.width() * m.height(),
                                    sizeof(T) * m.width(), sizeof(T)});
        });
};

void addArrayView1(py::module& m) {
    addArrayView1Scalar<double>(m, "ArrayView1D");
    addArrayView1Vector<double, 2>(m, "ArrayView1Vector2D");
    addArrayView1Vector<double, 3>(m, "ArrayView1Vector3D");
    addArrayView1Vector<double, 4>(m, "ArrayView1Vector4D");
    addArrayView1Scalar<float>(m, "ArrayView1F");
    addArrayView1Vector<float, 2>(m, "ArrayView1Vector2F");
    addArrayView1Vector<float, 3>(m, "ArrayView1Vector3F");
    addArrayView1Vector<float, 4>(m, "ArrayView1Vector4F");
}

void addArrayView2(py::module& m) {
    addArrayView2Scalar<double>(m, "ArrayView2D");
    addArrayView2Scalar<float>(m, "ArrayView2F");
}

void addArrayView3(py::module& m) {
    addArrayView3Scalar<double>(m, "ArrayView3D");
    addArrayView3Scalar<float>(m, "ArrayView3F");
}

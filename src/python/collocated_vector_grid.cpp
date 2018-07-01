// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "collocated_vector_grid.h"
#include "pybind11_utils.h"

#include <jet/collocated_vector_grid2.h>
#include <jet/collocated_vector_grid3.h>

namespace py = pybind11;
using namespace jet;

void addCollocatedVectorGrid2(py::module& m) {
    py::class_<CollocatedVectorGrid2, CollocatedVectorGrid2Ptr, VectorGrid2>(
        m, "CollocatedVectorGrid2",
        R"pbdoc(Abstract base class for 2-D collocated vector grid structure.)pbdoc")
        .def("__getitem__",
             [](const CollocatedVectorGrid2& instance,
                py::object obj) -> Vector2D {
                 if (py::isinstance<py::tuple>(obj)) {
                     auto tidx = obj.cast<py::tuple>();
                     if (tidx.size() == 2) {
                         return instance(tidx[0].cast<size_t>(),
                                         tidx[1].cast<size_t>());
                     } else {
                         throw std::invalid_argument(
                             "Size of index tuple must be 2.");
                     }
                 } else if (py::isinstance<py::list>(obj)) {
                     auto lidx = obj.cast<py::list>();
                     if (lidx.size() == 2) {
                         return instance(lidx[0].cast<size_t>(),
                                         lidx[1].cast<size_t>());
                     } else {
                         throw std::invalid_argument(
                             "Size of index list must be 2.");
                     }
                 } else {
                     throw std::invalid_argument(
                         "Input type must be tuple or list");
                 }
             },
             R"pbdoc(
             Returns the grid data at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j).
             )pbdoc",
             py::arg("idx"))
        .def("__setitem__",
             [](CollocatedVectorGrid2& instance, py::object obj,
                const Vector2D& val) {
                 if (py::isinstance<py::tuple>(obj)) {
                     auto tidx = obj.cast<py::tuple>();
                     if (tidx.size() == 2) {
                         instance(tidx[0].cast<size_t>(),
                                  tidx[1].cast<size_t>()) = val;
                     } else {
                         throw std::invalid_argument(
                             "Size of index tuple must be 2.");
                     }
                 } else if (py::isinstance<py::list>(obj)) {
                     auto lidx = obj.cast<py::list>();
                     if (lidx.size() == 2) {
                         instance(lidx[0].cast<size_t>(),
                                  lidx[1].cast<size_t>()) = val;
                     } else {
                         throw std::invalid_argument(
                             "Size of index list must be 2.");
                     }
                 } else {
                     throw std::invalid_argument(
                         "Input type must be tuple or list");
                 }
             },
             R"pbdoc(
            Sets the grid data at given data point.

            Parameters
            ----------
            - idx : Data point index (i, j).
            - val : Value to set.
            )pbdoc",
             py::arg("idx"), py::arg("val"))
        .def("divergenceAtDataPoint",
             &CollocatedVectorGrid2::divergenceAtDataPoint,
             R"pbdoc(
             Returns divergence at data point location.

             Parameters
             ----------
             - i : Data point index i.
             - j : Data point index j.
             )pbdoc",
             py::arg("i"), py::arg("j"))
        .def("curlAtDataPoint", &CollocatedVectorGrid2::curlAtDataPoint,
             R"pbdoc(
             Returns curl at data point location.

             Parameters
             ----------
             - i : Data point index i.
             - j : Data point index j.
             )pbdoc",
             py::arg("i"), py::arg("j"))
        .def("dataAccessor", &CollocatedVectorGrid2::dataAccessor,
             R"pbdoc(The data array accessor.)pbdoc")
        .def("dataPosition", &CollocatedVectorGrid2::dataPosition,
             R"pbdoc(The function that maps data point to its position.)pbdoc")
        .def("forEachDataPointIndex",
             [](CollocatedVectorGrid2& instance, py::function func) {
                 instance.forEachDataPointIndex(func);
             },
             R"pbdoc(
             Invokes the given function `func` for each data point.

             This function invokes the given function object `func` for each data
             point in serial manner. The input parameters are i and j indices of a
             data point. The order of execution is i-first, j-last.
             )pbdoc",
             py::arg("func"))
        .def("sample",
             [](const CollocatedVectorGrid2& instance, py::object obj) {
                 return instance.sample(objectToVector2D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("divergence",
             [](const CollocatedVectorGrid2& instance, py::object obj) {
                 return instance.divergence(objectToVector2D(obj));
             },
             R"pbdoc(Returns divergence at given position `x`.)pbdoc",
             py::arg("x"))
        .def("curl",
             [](const CollocatedVectorGrid2& instance, py::object obj) {
                 return instance.curl(objectToVector2D(obj));
             },
             R"pbdoc(Returns curl at given position `x`.)pbdoc", py::arg("x"))
        .def("sampler",
             [](const CollocatedVectorGrid2& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}

void addCollocatedVectorGrid3(py::module& m) {
    py::class_<CollocatedVectorGrid3, CollocatedVectorGrid3Ptr, VectorGrid3>(
        m, "CollocatedVectorGrid3",
        R"pbdoc(Abstract base class for 3-D collocated vector grid structure.)pbdoc")
        .def("__getitem__",
             [](const CollocatedVectorGrid3& instance,
                py::object obj) -> Vector3D {
                 if (py::isinstance<py::tuple>(obj)) {
                     auto tidx = obj.cast<py::tuple>();
                     if (tidx.size() == 3) {
                         return instance(tidx[0].cast<size_t>(),
                                         tidx[1].cast<size_t>(),
                                         tidx[2].cast<size_t>());
                     } else {
                         throw std::invalid_argument(
                             "Size of index tuple must be 3.");
                     }
                 } else if (py::isinstance<py::list>(obj)) {
                     auto lidx = obj.cast<py::list>();
                     if (lidx.size() == 3) {
                         return instance(lidx[0].cast<size_t>(),
                                         lidx[1].cast<size_t>(),
                                         lidx[2].cast<size_t>());
                     } else {
                         throw std::invalid_argument(
                             "Size of index list must be 3.");
                     }
                 } else {
                     throw std::invalid_argument(
                         "Input type must be tuple or list");
                 }
             },
             R"pbdoc(
             Returns the grid data at given data point.

             Parameters
             ----------
             - idx : Data point index (i, j, k).
             )pbdoc",
             py::arg("idx"))
        .def(
            "__setitem__",
            [](CollocatedVectorGrid3& instance, py::object obj,
               const Vector3D& val) {
                if (py::isinstance<py::tuple>(obj)) {
                    auto tidx = obj.cast<py::tuple>();
                    if (tidx.size() == 3) {
                        instance(tidx[0].cast<size_t>(), tidx[1].cast<size_t>(),
                                 tidx[2].cast<size_t>()) = val;
                    } else {
                        throw std::invalid_argument(
                            "Size of index tuple must be 3.");
                    }
                } else if (py::isinstance<py::list>(obj)) {
                    auto lidx = obj.cast<py::list>();
                    if (lidx.size() == 3) {
                        instance(lidx[0].cast<size_t>(), lidx[1].cast<size_t>(),
                                 lidx[2].cast<size_t>()) = val;
                    } else {
                        throw std::invalid_argument(
                            "Size of index list must be 3.");
                    }
                } else {
                    throw std::invalid_argument(
                        "Input type must be tuple or list");
                }
            },
            R"pbdoc(
            Sets the grid data at given data point.

            Parameters
            ----------
            - idx : Data point index (i, j, k).
            - val : Value to set.
            )pbdoc",
            py::arg("idx"), py::arg("val"))
        .def("divergenceAtDataPoint",
             &CollocatedVectorGrid3::divergenceAtDataPoint,
             R"pbdoc(
             Returns divergence at data point location.

             Parameters
             ----------
             - i : Data point index i.
             - j : Data point index j.
             - k : Data point index k.
             )pbdoc",
             py::arg("i"), py::arg("j"), py::arg("k"))
        .def("curlAtDataPoint", &CollocatedVectorGrid3::curlAtDataPoint,
             R"pbdoc(
             Returns curl at data point location.

             Parameters
             ----------
             - i : Data point index i.
             - j : Data point index j.
             - k : Data point index k.
             )pbdoc",
             py::arg("i"), py::arg("j"), py::arg("k"))
        .def("dataAccessor", &CollocatedVectorGrid3::dataAccessor,
             R"pbdoc(The data array accessor.)pbdoc")
        .def("dataPosition", &CollocatedVectorGrid3::dataPosition,
             R"pbdoc(The function that maps data point to its position.)pbdoc")
        .def("forEachDataPointIndex",
             [](CollocatedVectorGrid3& instance, py::function func) {
                 instance.forEachDataPointIndex(func);
             },
             R"pbdoc(
             Invokes the given function `func` for each data point.

             This function invokes the given function object `func` for each data
             point in serial manner. The input parameters are i and j indices of a
             data point. The order of execution is i-first, j-last.
             )pbdoc",
             py::arg("func"))
        .def("sample",
             [](const CollocatedVectorGrid3& instance, py::object obj) {
                 return instance.sample(objectToVector3D(obj));
             },
             R"pbdoc(Returns sampled value at given position `x`.)pbdoc",
             py::arg("x"))
        .def("divergence",
             [](const CollocatedVectorGrid3& instance, py::object obj) {
                 return instance.divergence(objectToVector3D(obj));
             },
             R"pbdoc(Returns divergence at given position `x`.)pbdoc",
             py::arg("x"))
        .def("curl",
             [](const CollocatedVectorGrid3& instance, py::object obj) {
                 return instance.curl(objectToVector3D(obj));
             },
             R"pbdoc(Returns curl at given position `x`.)pbdoc", py::arg("x"))
        .def("sampler",
             [](const CollocatedVectorGrid3& instance) {
                 return instance.sampler();
             },
             R"pbdoc(
             Returns the sampler function.

             This function returns the data sampler function object. The sampling
             function is linear.
             )pbdoc");
}

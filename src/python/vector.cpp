// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "vector.h"
#include "pybind11_utils.h"

#include <jet/vector3.h>

namespace py = pybind11;
using namespace jet;

void addVector2F(pybind11::module& m) {
    py::class_<Vector2F>(m, "Vector2F")
        // CTOR
        .def("__init__",
             [](Vector2F& instance, float x, float y) {
                 new (&instance) Vector2F(x, y);
             },
             R"pbdoc(
             Constructs Vector2F.

             This method constructs float-type 2-D vector with x and y.
             )pbdoc",
             py::arg("x") = 0.0f, py::arg("y") = 0.0f)
        .def_readwrite("x", &Vector2F::x)
        .def_readwrite("y", &Vector2F::y)
        .def("setZero", &Vector2F::setZero)
        .def("normalize", &Vector2F::normalize)
        .def("dot",
             [](const Vector2F& instance, py::object other) {
                 return instance.dot(objectToVector2F(other));
             })
        .def("cross",
             [](const Vector2F& instance, py::object other) {
                 return instance.cross(objectToVector2F(other));
             })
        .def("sum", &Vector2F::sum)
        .def("avg", &Vector2F::avg)
        .def("min", &Vector2F::min)
        .def("max", &Vector2F::max)
        .def("absmin", &Vector2F::absmin)
        .def("absmax", &Vector2F::absmax)
        .def("dominantAxis", &Vector2F::dominantAxis)
        .def("subminantAxis", &Vector2F::subminantAxis)
        .def("normalized", &Vector2F::normalized)
        .def("length", &Vector2F::length)
        .def("lengthSquared", &Vector2F::lengthSquared)
        .def("distanceTo",
             [](const Vector2F& instance, py::object other) {
                 return instance.distanceTo(objectToVector2F(other));
             })
        .def("distanceSquaredTo",
             [](const Vector2F& instance, py::object other) {
                 return instance.distanceSquaredTo(objectToVector2F(other));
             })
        .def("reflected",
             [](const Vector2F& instance, py::object other) {
                 return instance.reflected(objectToVector2F(other));
             })
        .def("projected",
             [](const Vector2F& instance, py::object other) {
                 return instance.projected(objectToVector2F(other));
             })
        .def("tangential", &Vector2F::tangential)
        .def("__getitem__",
             [](const Vector2F& instance, size_t i) -> float {
                 return instance[i];
             })
        .def("__setitem__",
             [](Vector2F& instance, size_t i, float val) { instance[i] = val; })
        .def("__add__",
             [](const Vector2F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.add(object.cast<float>());
                 } else {
                     return instance.add(objectToVector2F(object));
                 }
             })
        .def("__sub__",
             [](const Vector2F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.sub(object.cast<float>());
                 } else {
                     return instance.sub(objectToVector2F(object));
                 }
             })
        .def("__rsub__",
             [](const Vector2F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.rsub(object.cast<float>());
                 } else {
                     return instance.rsub(objectToVector2F(object));
                 }
             })
        .def("__mul__",
             [](const Vector2F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.mul(object.cast<float>());
                 } else {
                     return instance.mul(objectToVector2F(object));
                 }
             })
        .def("__div__",
             [](const Vector2F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.div(object.cast<float>());
                 } else {
                     return instance.div(objectToVector2F(object));
                 }
             })
        .def("__rdiv__",
             [](const Vector2F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.rdiv(object.cast<float>());
                 } else {
                     return instance.rdiv(objectToVector2F(object));
                 }
             })
        .def("__truediv__",
             [](const Vector2F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.div(object.cast<float>());
                 } else {
                     return instance.div(objectToVector2F(object));
                 }
             })
        .def("__len__", [](const Vector2F&) { return 2; })
        .def("__iter__",
             [](const Vector2F& instance) {
                 return py::make_iterator(&instance.x, &instance.y + 1);
             })
        .def("__str__",
             [](const Vector2F& instance) {
                 return std::string(std::to_string(instance.x) + ", " +
                                    std::to_string(instance.y));
             })
        .def("__eq__", [](const Vector2F& instance, py::object obj) {
            Vector2F other = objectToVector2F(obj);
            return instance == other;
        });
    ;
}

void addVector2D(pybind11::module& m) {
    py::class_<Vector2D>(m, "Vector2D")
        // CTOR
        .def("__init__",
             [](Vector2D& instance, double x, double y) {
                 new (&instance) Vector2D(x, y);
             },
             R"pbdoc(
             Constructs Vector2D.

             This method constructs double-type 2-D vector with x and y.
             )pbdoc",
             py::arg("x") = 0.0, py::arg("y") = 0.0)
        .def_readwrite("x", &Vector2D::x)
        .def_readwrite("y", &Vector2D::y)
        .def("setZero", &Vector2D::setZero)
        .def("normalize", &Vector2D::normalize)
        .def("dot",
             [](const Vector2D& instance, py::object other) {
                 return instance.dot(objectToVector2D(other));
             })
        .def("cross",
             [](const Vector2D& instance, py::object other) {
                 return instance.cross(objectToVector2D(other));
             })
        .def("sum", &Vector2D::sum)
        .def("avg", &Vector2D::avg)
        .def("min", &Vector2D::min)
        .def("max", &Vector2D::max)
        .def("absmin", &Vector2D::absmin)
        .def("absmax", &Vector2D::absmax)
        .def("dominantAxis", &Vector2D::dominantAxis)
        .def("subminantAxis", &Vector2D::subminantAxis)
        .def("normalized", &Vector2D::normalized)
        .def("length", &Vector2D::length)
        .def("lengthSquared", &Vector2D::lengthSquared)
        .def("distanceTo",
             [](const Vector2D& instance, py::object other) {
                 return instance.distanceTo(objectToVector2D(other));
             })
        .def("distanceSquaredTo",
             [](const Vector2D& instance, py::object other) {
                 return instance.distanceSquaredTo(objectToVector2D(other));
             })
        .def("reflected",
             [](const Vector2D& instance, py::object other) {
                 return instance.reflected(objectToVector2D(other));
             })
        .def("projected",
             [](const Vector2D& instance, py::object other) {
                 return instance.projected(objectToVector2D(other));
             })
        .def("tangential", &Vector2D::tangential)
        .def("__getitem__",
             [](const Vector2D& instance, size_t i) -> double {
                 return instance[i];
             })
        .def("__setitem__", [](Vector2D& instance, size_t i,
                               double val) { instance[i] = val; })
        .def("__add__",
             [](const Vector2D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.add(object.cast<double>());
                 } else {
                     return instance.add(objectToVector2D(object));
                 }
             })
        .def("__sub__",
             [](const Vector2D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.sub(object.cast<double>());
                 } else {
                     return instance.sub(objectToVector2D(object));
                 }
             })
        .def("__rsub__",
             [](const Vector2D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.rsub(object.cast<double>());
                 } else {
                     return instance.rsub(objectToVector2D(object));
                 }
             })
        .def("__mul__",
             [](const Vector2D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.mul(object.cast<double>());
                 } else {
                     return instance.mul(objectToVector2D(object));
                 }
             })
        .def("__div__",
             [](const Vector2D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.div(object.cast<double>());
                 } else {
                     return instance.div(objectToVector2D(object));
                 }
             })
        .def("__rdiv__",
             [](const Vector2D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.rdiv(object.cast<double>());
                 } else {
                     return instance.rdiv(objectToVector2D(object));
                 }
             })
        .def("__truediv__",
             [](const Vector2D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.div(object.cast<double>());
                 } else {
                     return instance.div(objectToVector2D(object));
                 }
             })
        .def("__len__", [](const Vector2D&) { return 2; })
        .def("__iter__",
             [](const Vector2D& instance) {
                 return py::make_iterator(&instance.x, &instance.y + 1);
             })
        .def("__str__",
             [](const Vector2D& instance) {
                 return std::string(std::to_string(instance.x) + ", " +
                                    std::to_string(instance.y));
             })
        .def("__eq__", [](const Vector2D& instance, py::object obj) {
            Vector2D other = objectToVector2D(obj);
            return instance == other;
        });
}

void addVector3F(pybind11::module& m) {
    py::class_<Vector3F>(m, "Vector3F")
        // CTOR
        .def("__init__",
             [](Vector3F& instance, float x, float y, float z) {
                 new (&instance) Vector3F(x, y, z);
             },
             R"pbdoc(
             Constructs Vector3F.

             This method constructs float-type 3-D vector with x, y, and z.
             )pbdoc",
             py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def_readwrite("x", &Vector3F::x)
        .def_readwrite("y", &Vector3F::y)
        .def_readwrite("z", &Vector3F::z)
        .def("setZero", &Vector3F::setZero)
        .def("normalize", &Vector3F::normalize)
        .def("dot",
             [](const Vector3F& instance, py::object other) {
                 return instance.dot(objectToVector3F(other));
             })
        .def("cross",
             [](const Vector3F& instance, py::object other) {
                 return instance.cross(objectToVector3F(other));
             })
        .def("sum", &Vector3F::sum)
        .def("avg", &Vector3F::avg)
        .def("min", &Vector3F::min)
        .def("max", &Vector3F::max)
        .def("absmin", &Vector3F::absmin)
        .def("absmax", &Vector3F::absmax)
        .def("dominantAxis", &Vector3F::dominantAxis)
        .def("subminantAxis", &Vector3F::subminantAxis)
        .def("normalized", &Vector3F::normalized)
        .def("length", &Vector3F::length)
        .def("lengthSquared", &Vector3F::lengthSquared)
        .def("distanceTo",
             [](const Vector3F& instance, py::object other) {
                 return instance.distanceTo(objectToVector3F(other));
             })
        .def("distanceSquaredTo",
             [](const Vector3F& instance, py::object other) {
                 return instance.distanceSquaredTo(objectToVector3F(other));
             })
        .def("reflected",
             [](const Vector3F& instance, py::object other) {
                 return instance.reflected(objectToVector3F(other));
             })
        .def("projected",
             [](const Vector3F& instance, py::object other) {
                 return instance.projected(objectToVector3F(other));
             })
        .def("tangential", &Vector3F::tangential)
        .def("__getitem__",
             [](const Vector3F& instance, size_t i) -> float {
                 return instance[i];
             })
        .def("__setitem__",
             [](Vector3F& instance, size_t i, float val) { instance[i] = val; })
        .def("__add__",
             [](const Vector3F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.add(object.cast<float>());
                 } else {
                     return instance.add(objectToVector3F(object));
                 }
             })
        .def("__sub__",
             [](const Vector3F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.sub(object.cast<float>());
                 } else {
                     return instance.sub(objectToVector3F(object));
                 }
             })
        .def("__rsub__",
             [](const Vector3F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.rsub(object.cast<float>());
                 } else {
                     return instance.rsub(objectToVector3F(object));
                 }
             })
        .def("__mul__",
             [](const Vector3F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.mul(object.cast<float>());
                 } else {
                     return instance.mul(objectToVector3F(object));
                 }
             })
        .def("__div__",
             [](const Vector3F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.div(object.cast<float>());
                 } else {
                     return instance.div(objectToVector3F(object));
                 }
             })
        .def("__rdiv__",
             [](const Vector3F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.rdiv(object.cast<float>());
                 } else {
                     return instance.rdiv(objectToVector3F(object));
                 }
             })
        .def("__truediv__",
             [](const Vector3F& instance, py::object object) {
                 if (py::isinstance<float>(object)) {
                     return instance.div(object.cast<float>());
                 } else {
                     return instance.div(objectToVector3F(object));
                 }
             })
        .def("__len__", [](const Vector3F&) { return 3; })
        .def("__iter__",
             [](const Vector3F& instance) {
                 return py::make_iterator(&instance.x, &instance.z + 1);
             })
        .def("__str__",
             [](const Vector3F& instance) {
                 return std::string(std::to_string(instance.x) + ", " +
                                    std::to_string(instance.y) + ", " +
                                    std::to_string(instance.z));
             })
        .def("__eq__", [](const Vector3F& instance, py::object obj) {
            Vector3F other = objectToVector3F(obj);
            return instance == other;
        });
    ;
}

void addVector3D(pybind11::module& m) {
    py::class_<Vector3D>(m, "Vector3D")
        // CTOR
        .def("__init__",
             [](Vector3D& instance, double x, double y, double z) {
                 new (&instance) Vector3D(x, y, z);
             },
             R"pbdoc(
             Constructs Vector3D.

             This method constructs double-type 3-D vector with x, y, and z.
             )pbdoc",
             py::arg("x") = 0.0, py::arg("y") = 0.0, py::arg("z") = 0.0)
        .def_readwrite("x", &Vector3D::x)
        .def_readwrite("y", &Vector3D::y)
        .def_readwrite("z", &Vector3D::z)
        .def("setZero", &Vector3D::setZero)
        .def("normalize", &Vector3D::normalize)
        .def("dot",
             [](const Vector3D& instance, py::object other) {
                 return instance.dot(objectToVector3D(other));
             })
        .def("cross",
             [](const Vector3D& instance, py::object other) {
                 return instance.cross(objectToVector3D(other));
             })
        .def("sum", &Vector3D::sum)
        .def("avg", &Vector3D::avg)
        .def("min", &Vector3D::min)
        .def("max", &Vector3D::max)
        .def("absmin", &Vector3D::absmin)
        .def("absmax", &Vector3D::absmax)
        .def("dominantAxis", &Vector3D::dominantAxis)
        .def("subminantAxis", &Vector3D::subminantAxis)
        .def("normalized", &Vector3D::normalized)
        .def("length", &Vector3D::length)
        .def("lengthSquared", &Vector3D::lengthSquared)
        .def("distanceTo",
             [](const Vector3D& instance, py::object other) {
                 return instance.distanceTo(objectToVector3D(other));
             })
        .def("distanceSquaredTo",
             [](const Vector3D& instance, py::object other) {
                 return instance.distanceSquaredTo(objectToVector3D(other));
             })
        .def("reflected",
             [](const Vector3D& instance, py::object other) {
                 return instance.reflected(objectToVector3D(other));
             })
        .def("projected",
             [](const Vector3D& instance, py::object other) {
                 return instance.projected(objectToVector3D(other));
             })
        .def("tangential", &Vector3D::tangential)
        .def("__getitem__",
             [](const Vector3D& instance, size_t i) -> double {
                 return instance[i];
             })
        .def("__setitem__", [](Vector3D& instance, size_t i,
                               double val) { instance[i] = val; })
        .def("__add__",
             [](const Vector3D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.add(object.cast<double>());
                 } else {
                     return instance.add(objectToVector3D(object));
                 }
             })
        .def("__sub__",
             [](const Vector3D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.sub(object.cast<double>());
                 } else {
                     return instance.sub(objectToVector3D(object));
                 }
             })
        .def("__rsub__",
             [](const Vector3D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.rsub(object.cast<double>());
                 } else {
                     return instance.rsub(objectToVector3D(object));
                 }
             })
        .def("__mul__",
             [](const Vector3D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.mul(object.cast<double>());
                 } else {
                     return instance.mul(objectToVector3D(object));
                 }
             })
        .def("__div__",
             [](const Vector3D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.div(object.cast<double>());
                 } else {
                     return instance.div(objectToVector3D(object));
                 }
             })
        .def("__rdiv__",
             [](const Vector3D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.rdiv(object.cast<double>());
                 } else {
                     return instance.rdiv(objectToVector3D(object));
                 }
             })
        .def("__truediv__",
             [](const Vector3D& instance, py::object object) {
                 if (py::isinstance<double>(object)) {
                     return instance.div(object.cast<double>());
                 } else {
                     return instance.div(objectToVector3D(object));
                 }
             })
        .def("__len__", [](const Vector3D&) { return 3; })
        .def("__iter__",
             [](const Vector3D& instance) {
                 return py::make_iterator(&instance.x, &instance.z + 1);
             })
        .def("__str__",
             [](const Vector3D& instance) {
                 return std::string(std::to_string(instance.x) + ", " +
                                    std::to_string(instance.y) + ", " +
                                    std::to_string(instance.z));
             })
        .def("__eq__", [](const Vector3D& instance, py::object obj) {
            Vector3D other = objectToVector3D(obj);
            return instance == other;
        });
}

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "vector.h"
#include "pybind11_utils.h"

#include <jet/matrix.h>

namespace py = pybind11;
using namespace jet;

template <typename T, size_t R>
auto obj2vec(const py::object& obj, const std::string& name) {
    using VectorType = Matrix<T, R, 1>;
    if (pybind11::isinstance<VectorType>(obj)) {
        return obj.cast<VectorType>();
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        return tupleToVector<T, R>(pybind11::tuple(obj));
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        return tupleToVector<T, R>(pybind11::list(obj));
    } else {
        throw std::invalid_argument(
            ("Cannot convert to " + name + ".").c_str());
    }
}

template <typename PyBindClass, typename T, size_t R>
void addVector(PyBindClass& cls, const std::string& name) {
    using VectorType = Matrix<T, R, 1>;

    cls.def("fill", (void (VectorType::*)(const T&)) & VectorType::fill)
        .def("dot",
             [name](const VectorType& instance, py::object other) {
                 return instance.dot(obj2vec<T, R>(other, name));
             })
        .def("cross",
             [name](const VectorType& instance, py::object other) {
                 return instance.cross(obj2vec<T, R>(other, name));
             })
        .def("sum", &VectorType::sum)
        .def("avg", &VectorType::avg)
        .def("min", &VectorType::min)
        .def("max", &VectorType::max)
        .def("absmin", &VectorType::absmin)
        .def("absmax", &VectorType::absmax)
        .def("dominantAxis", &VectorType::dominantAxis)
        .def("subminantAxis", &VectorType::subminantAxis)
        .def("lengthSquared", &VectorType::lengthSquared)
        .def("distanceSquaredTo",
             [name](const VectorType& instance, py::object other) {
                 return instance.distanceSquaredTo(obj2vec<T, R>(other, name));
             })
        .def("__getitem__",
             [name](const VectorType& instance, size_t i) -> T {
                 return instance[i];
             })
        .def("__setitem__", [name](VectorType& instance, size_t i,
                                   T val) { instance[i] = val; })
        .def("__add__",
             [name](const VectorType& instance,
                    py::object object) -> VectorType {
                 if (py::isinstance<T>(object)) {
                     return instance + object.cast<T>();
                 } else {
                     return instance + obj2vec<T, R>(object, name);
                 }
             })
        .def("__sub__",
             [name](const VectorType& instance,
                    py::object object) -> VectorType {
                 if (py::isinstance<T>(object)) {
                     return instance - object.cast<T>();
                 } else {
                     return instance - obj2vec<T, R>(object, name);
                 }
             })
        .def("__rsub__",
             [name](const VectorType& instance,
                    py::object object) -> VectorType {
                 if (py::isinstance<T>(object)) {
                     return object.cast<T>() - instance;
                 } else {
                     return obj2vec<T, R>(object, name) - instance;
                 }
             })
        .def("__mul__",
             [name](const VectorType& instance,
                    py::object object) -> VectorType {
                 if (py::isinstance<T>(object)) {
                     return instance * object.cast<T>();
                 } else {
                     // TODO: Should be deprecated
                     return elemMul(instance, obj2vec<T, R>(object, name));
                 }
             })
        .def("__div__",
             [name](const VectorType& instance,
                    py::object object) -> VectorType {
                 if (py::isinstance<T>(object)) {
                     return instance / object.cast<T>();
                 } else {
                     // TODO: Should be deprecated
                     return elemDiv(instance, obj2vec<T, R>(object, name));
                 }
             })
        .def("__rdiv__",
             [name](const VectorType& instance,
                    py::object object) -> VectorType {
                 if (py::isinstance<T>(object)) {
                     return object.cast<T>() / instance;
                 } else {
                     // TODO: Should be deprecated
                     return elemDiv(obj2vec<T, R>(object, name), instance);
                 }
             })
        .def("__truediv__",
             [name](const VectorType& instance,
                    py::object object) -> VectorType {
                 if (py::isinstance<T>(object)) {
                     return instance / object.cast<T>();
                 } else {
                     // TODO: Should be deprecated
                     return elemDiv(instance, obj2vec<T, R>(object, name));
                 }
             })
        .def("__len__", [](const VectorType&) { return R; })
        .def("__iter__",
             [](const VectorType& instance) {
                 return py::make_iterator(&instance.x, &instance.x + R);
             })
        .def("__str__",
             [](const VectorType& instance) {
                 std::string result;
                 for (size_t i = 0; i < R; ++i) {
                     result += std::to_string(instance.x);
                     if (i < R - 1) {
                         result += ", ";
                     }
                 }
             })
        .def("__eq__", [name](const VectorType& instance, py::object obj) {
            VectorType other = obj2vec<T, R>(obj, name);
            return instance == other;
        });
}

template <typename PyBindClass, typename T, size_t R>
void addFloatVector(PyBindClass& cls, const std::string& name) {
    using VectorType = Matrix<T, R, 1>;

    cls.def("normalize", &VectorType::normalize)
        .def("avg", &VectorType::avg)
        .def("normalized", &VectorType::normalized)
        .def("length", &VectorType::length)
        .def("distanceTo",
             [name](const VectorType& instance, py::object other) {
                 return instance.distanceTo(obj2vec<T, R>(other, name));
             })
        .def("reflected",
             [name](const VectorType& instance, py::object other) {
                 return instance.reflected(obj2vec<T, R>(other, name));
             })
        .def("projected", [name](const VectorType& instance, py::object other) {
            return instance.projected(obj2vec<T, R>(other, name));
        });
}

#define ADD_VECTOR2(NAME, SCALAR)                    \
    py::class_<NAME> cls(m, #NAME);                  \
    cls.def("__init__",                              \
            [](NAME& instance, SCALAR x, SCALAR y) { \
                new (&instance) NAME(x, y);          \
            },                                       \
            "Constructs " #NAME                      \
            ".\n\n"                                  \
            "This method constructs " #SCALAR        \
            "-type 2-D vector with x and y.\n",      \
            py::arg("x") = 0, py::arg("y") = 0)      \
        .def_readwrite("x", &NAME::x)                \
        .def_readwrite("y", &NAME::y);               \
    addVector<py::class_<NAME>, SCALAR, 2>(cls, #NAME);

#define ADD_FLOAT_VECTOR2(NAME, SCALAR)               \
    ADD_VECTOR2(NAME, SCALAR);                        \
    cls.def("tangential", &NAME::tangential<SCALAR>); \
    addFloatVector<py::class_<NAME>, SCALAR, 2>(cls, #NAME);

#define ADD_VECTOR3(NAME, SCALAR)                                 \
    py::class_<NAME> cls(m, #NAME);                               \
    cls.def("__init__",                                           \
            [](NAME& instance, SCALAR x, SCALAR y, SCALAR z) {    \
                new (&instance) NAME(x, y, z);                    \
            },                                                    \
            "Constructs " #NAME                                   \
            ".\n\n"                                               \
            "This method constructs " #SCALAR                     \
            "-type 3-D vector with x, y, and z.\n",               \
            py::arg("x") = 0, py::arg("y") = 0, py::arg("z") = 0) \
        .def_readwrite("x", &NAME::x)                             \
        .def_readwrite("y", &NAME::y)                             \
        .def_readwrite("z", &NAME::z);                            \
    addVector<py::class_<NAME>, SCALAR, 3>(cls, #NAME);

#define ADD_FLOAT_VECTOR3(NAME, SCALAR)                 \
    ADD_VECTOR3(NAME, SCALAR);                          \
    cls.def("tangentials", &NAME::tangentials<SCALAR>); \
    addFloatVector<py::class_<NAME>, SCALAR, 3>(cls, #NAME);

void addVector2F(pybind11::module& m) { ADD_FLOAT_VECTOR2(Vector2F, float); }
void addVector2D(pybind11::module& m) { ADD_FLOAT_VECTOR2(Vector2D, double); }
void addVector2Z(pybind11::module& m) { ADD_VECTOR2(Vector2Z, ssize_t); }
void addVector2UZ(pybind11::module& m) { ADD_VECTOR2(Vector2UZ, size_t); }
void addVector3F(pybind11::module& m) { ADD_FLOAT_VECTOR3(Vector3F, float); }
void addVector3D(pybind11::module& m) { ADD_FLOAT_VECTOR3(Vector3D, double); }
void addVector3Z(pybind11::module& m) { ADD_VECTOR3(Vector3Z, ssize_t); }
void addVector3UZ(pybind11::module& m) { ADD_VECTOR3(Vector3UZ, size_t); }

void addVectors(pybind11::module& m) {
    addVector2F(m);
    addVector2D(m);
    addVector2Z(m);
    addVector2UZ(m);
    addVector3F(m);
    addVector3D(m);
    addVector3Z(m);
    addVector3UZ(m);
}

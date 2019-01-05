// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "collider_set.h"
#include "pybind11_utils.h"

#include <jet/collider_set2.h>
#include <jet/collider_set3.h>

namespace py = pybind11;
using namespace jet;

void addColliderSet2(py::module& m) {
    py::class_<ColliderSet2, ColliderSet2Ptr, Collider2>(m, "ColliderSet2",
                                                         R"pbdoc(
        Collection of 2-D colliders
        )pbdoc")
        .def(py::init([](py::args args) {
                 if (args.size() == 1) {
                     return new ColliderSet2(
                         args[0].cast<std::vector<Collider2Ptr>>());
                 } else if (args.size() == 0) {
                     return new ColliderSet2();
                 }
                 throw std::invalid_argument("Invalid number of arguments.");
             }),
             R"pbdoc(
             Constructs ColliderSet2

             This method constructs ColliderSet2 with other colliders.

             Parameters
             ----------
             - `*args` : List of other colliders. Must be size of 0 or 1.
             )pbdoc")
        .def("addCollider", &ColliderSet2::addCollider,
             R"pbdoc(Adds a collider to the set.)pbdoc")
        .def_property_readonly("numberOfColliders",
                               &ColliderSet2::numberOfColliders,
                               R"pbdoc(Number of colliders.)pbdoc")
        .def("collider", &ColliderSet2::collider,
             R"pbdoc(Returns collider at index i.)pbdoc");
}

void addColliderSet3(py::module& m) {
    py::class_<ColliderSet3, ColliderSet3Ptr, Collider3>(m, "ColliderSet3",
                                                         R"pbdoc(
        Collection of 3-D colliders
        )pbdoc")
        .def(py::init([](py::args args) {
                 if (args.size() == 1) {
                     return new ColliderSet3(
                         args[0].cast<std::vector<Collider3Ptr>>());
                 } else if (args.size() == 0) {
                     return new ColliderSet3();
                 }
                 throw std::invalid_argument("Invalid number of arguments.");
             }),
             R"pbdoc(
             Constructs ColliderSet3

             This method constructs ColliderSet3 with other colliders.

             Parameters
             ----------
             - `*args` : List of other colliders. Must be size of 0 or 1.
             )pbdoc")
        .def("addCollider", &ColliderSet3::addCollider,
             R"pbdoc(Adds a collider to the set.)pbdoc")
        .def_property_readonly("numberOfColliders",
                               &ColliderSet3::numberOfColliders,
                               R"pbdoc(Number of colliders.)pbdoc")
        .def("collider", &ColliderSet3::collider,
             R"pbdoc(Returns collider at index i.)pbdoc");
}

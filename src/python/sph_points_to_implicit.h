// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_PYTHON_SPH_POINTS_TO_IMPLICIT_H_
#define SRC_PYTHON_SPH_POINTS_TO_IMPLICIT_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void addSphPointsToImplicit2(pybind11::module& m);
void addSphPointsToImplicit3(pybind11::module& m);

#endif  // SRC_PYTHON_SPH_POINTS_TO_IMPLICIT_H_

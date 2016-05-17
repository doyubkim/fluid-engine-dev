// Copyright (c) 2016 Doyub Kim

// Function extrapolateToRegion is from http://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/

#ifndef INCLUDE_JET_FDM_UTILS_H_
#define INCLUDE_JET_FDM_UTILS_H_

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/cell_centered_vector_grid3.h>
#include <jet/collocated_vector_grid2.h>
#include <jet/collocated_vector_grid3.h>
#include <jet/face_centered_grid2.h>
#include <jet/face_centered_grid3.h>
#include <jet/scalar_field2.h>
#include <jet/scalar_field3.h>

#include <iostream>

namespace jet {

Vector2D gradient2(
    const ConstArrayAccessor2<double>& data,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j);

std::array<Vector2D, 2> gradient2(
    const ConstArrayAccessor2<Vector2D>& data,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j);

Vector3D gradient3(
    const ConstArrayAccessor3<double>& data,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k);

std::array<Vector3D, 3> gradient3(
    const ConstArrayAccessor3<Vector3D>& data,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k);

double laplacian2(
    const ConstArrayAccessor2<double>& data,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j);

Vector2D laplacian2(
    const ConstArrayAccessor2<Vector2D>& data,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j);

double laplacian3(
    const ConstArrayAccessor3<double>& data,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k);

Vector3D laplacian3(
    const ConstArrayAccessor3<Vector3D>& data,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k);

//!
//! Projects 2-D vector field in the normal direction of signed-distance
//! field.
//!
//! For the region where the input signed-distance field is less than zero,
//! the input vector field will be projected on to the contour of the
//! signed-distance field.
//!
//! \param sdf - reference signed-distance field
//! \param data - collocated-type vector grid data
//!
void projectVectorFieldToSdf(
    const ScalarField2& sdf,
    CollocatedVectorGrid2* data);

//!
//! Projects 3-D vector field in the normal direction of signed-distance
//! field.
//!
//! For the region where the input signed-distance field is less than zero,
//! the input vector field will be projected on to the contour of the
//! signed-distance field.
//!
//! \param sdf - reference signed-distance field
//! \param data - collocated-type vector grid data
//!
void projectVectorFieldToSdf(
    const ScalarField3& sdf,
    CollocatedVectorGrid3* data);

//!
//! Projects 2-D vector field in the normal direction of signed-distance
//! field.
//!
//! For the region where the input signed-distance field is less than zero,
//! the input vector field will be projected on to the contour of the
//! signed-distance field.
//!
//! \param sdf - reference signed-distance field
//! \param data - MAC-type vector grid data
//!
void projectVectorFieldToSdf(const ScalarField2& sdf, FaceCenteredGrid2* data);

//!
//! Projects 3-D vector field in the normal direction of signed-distance
//! field.
//!
//! For the region where the input signed-distance field is less than zero,
//! the input vector field will be projected on to the contour of the
//! signed-distance field.
//!
//! \param sdf - reference signed-distance field
//! \param data - MAC-type vector grid data
//!
void projectVectorFieldToSdf(const ScalarField3& sdf, FaceCenteredGrid3* data);

}  // namespace jet

#endif  // INCLUDE_JET_FDM_UTILS_H_

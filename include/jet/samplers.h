// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SAMPLERS_H_
#define INCLUDE_JET_SAMPLERS_H_

#include <jet/vector2.h>
#include <jet/vector3.h>

namespace jet {

//!
//! \brief      Returns randomly sampled direction within a cone.
//!
//! For a given cone, defined by axis and angle, this function returns a sampled
//! direction vector within the cone.
//!
//! \param[in]  u1    First random sample.
//! \param[in]  u2    Second random sample.
//! \param[in]  axis  The axis of the cone.
//! \param[in]  angle The angle of the cone.
//!
//! \tparam     T     Real number type.
//!
//! \return     Sampled direction vector.
//!
template <typename T>
inline Vector3<T> uniformSampleCone(
    T u1, T u2, const Vector3<T>& axis, T angle);

//!
//! \brief      Returns randomly sampled point within a unit hemisphere.
//!
//! For a given unit hemisphere, defined by center normal vector, this function
//! returns a point within the hemisphere.
//!
//! \param[in]  u1      First random sample.
//! \param[in]  u2      Second random sample.
//! \param[in]  normal  The center normal of the hemisphere.
//!
//! \tparam     T       Real number type.
//!
//! \return     Sampled point.
//!
template <typename T>
inline Vector3<T> uniformSampleHemisphere(
    T u1, T u2, const Vector3<T>& normal);

//!
//! \brief      Returns weighted sampled point on a hemisphere.
//!
//! For a given hemisphere, defined by center normal vector, this function
//! returns a point on the hemisphere, where the probability is
//! consine-weighted.
//!
//! \param[in]  u1      First random sample.
//! \param[in]  u2      Second random sample.
//! \param[in]  normal  The center normal of the hemisphere.
//!
//! \tparam     T       Real number type.
//!
//! \return     Sampled point.
//!
template <typename T>
inline Vector3<T> cosineWeightedSampleHemisphere(
    T u1, T u2, const Vector3<T>& normal);

//!
//! \brief      Returns randomly a point on a sphere.
//!
//! For a given sphere, defined by center normal vector, this function returns a
//! point on the sphere.
//!
//! \param[in]  u1    First random sample.
//! \param[in]  u2    Second random sample.
//! \param[in]  axis  The axis of the cone.
//! \param[in]  angle The angle of the cone.
//!
//! \tparam     T     Real number type.
//!
//! \return     Sampled point.
//!
template <typename T>
inline Vector3<T> uniformSampleSphere(T u1, T u2);

//!
//! \brief      Returns randomly a point on a disk.
//!
//! For a given disk, this function returns a point on the disk.
//!
//! \param[in]  u1    First random sample.
//! \param[in]  u2    Second random sample.
//!
//! \tparam     T     Real number type.
//!
//! \return     Sampled point.
//!
template <typename T>
inline Vector2<T> uniformSampleDisk(T u1, T u2);

}  // namespace jet

#include "detail/samplers-inl.h"

#endif  // INCLUDE_JET_SAMPLERS_H_

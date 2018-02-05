// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_LEVEL_SET_UTILS_H_
#define INCLUDE_JET_LEVEL_SET_UTILS_H_

#include <jet/macros.h>

namespace jet {

//!
//! \brief      Returns true if \p phi is inside the implicit surface (< 0).
//!
//! \param[in]  phi   The level set value.
//!
//! \tparam     T     Value type.
//!
//! \return     True if inside the implicit surface, false otherwise.
//!
template <typename T>
bool isInsideSdf(T phi);

//!
//! \brief      Returns smeared Heaviside function.
//!
//! This function returns smeared (or smooth) Heaviside (or step) function
//! between 0 and 1. If \p phi is less than -1.5, it will return 0. If \p phi
//! is greater than 1.5, it will return 1. Between -1.5 and 1.5, the function
//! will return smooth profile between 0 and 1. Derivative of this function is
//! smearedDeltaSdf.
//!
//! \param[in]  phi   The level set value.
//!
//! \tparam     T     Value type.
//!
//! \return     Smeared Heaviside function.
//!
template <typename T>
T smearedHeavisideSdf(T phi);

//!
//! \brief      Returns smeared delta function.
//!
//! This function returns smeared (or smooth) delta function between 0 and 1.
//! If \p phi is less than -1.5, it will return 0. If \p phi is greater than
//! 1.5, it will also return 0. Between -1.5 and 1.5, the function will return
//! smooth delta function. Integral of this function is smearedHeavisideSdf.
//!
//! \param[in]  phi   The level set value.
//!
//! \tparam     T     Value type.
//!
//! \return     Smeared delta function.
//!
template <typename T>
T smearedDeltaSdf(T phi);

//!
//! \brief      Returns the fraction occupied by the implicit surface.
//!
//! The input parameters, \p phi0 and \p phi1, are the level set values,
//! measured from two nearby points. This function computes how much the
//! implicit surface occupies the line between two points. For example, if both
//! \p phi0 and \p phi1 are negative, it means the points are both inside the
//! surface, thus the function will return 1. If both are positive, it will
//! return 0 because both are outside the surface. If the signs are different,
//! then only one of the points is inside the surface and the function will
//! return a value between 0 and 1.
//!
//! \param[in]  phi0  The level set value from the first point.
//! \param[in]  phi1  The level set value from the second point.
//!
//! \tparam     T     Value type.
//!
//! \return     The fraction occupied by the implicit surface.
//!
template <typename T>
T fractionInsideSdf(T phi0, T phi1);

//!
//! \brief      Returns the fraction occupied by the implicit surface.
//!
//! Given four signed distance values (square corners), determine what fraction
//! of the square is "inside". The original implementation can be found from
//! Christopher Batty's variational fluid code at
//! https://github.com/christopherbatty/Fluid3D.
//!
//! \tparam T               Value type.
//!
//! \param phiBottomLeft    The level set value on the bottom-left corner.
//! \param phiBottomRight   The level set value on the bottom-right corner.
//! \param phiTopLeft       The level set value on the top-left corner.
//! \param phiTopRight      The level set value on the top-right corner.
//!
//! \return                 The fraction occupied by the implicit surface.
//!
template <typename T>
T fractionInside(T phiBottomLeft, T phiBottomRight, T phiTopLeft,
                 T phiTopRight);

}  // namespace jet

#include "detail/level_set_utils-inl.h"

#endif  // INCLUDE_JET_LEVEL_SET_UTILS_H_

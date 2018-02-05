// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_MG_LINEAR_SYSTEM3_H_
#define INCLUDE_JET_FDM_MG_LINEAR_SYSTEM3_H_

#include <jet/face_centered_grid3.h>
#include <jet/fdm_linear_system3.h>
#include <jet/mg.h>

namespace jet {

//! Multigrid-style 3-D FDM matrix.
typedef MgMatrix<FdmBlas3> FdmMgMatrix3;

//! Multigrid-style 3-D FDM vector.
typedef MgVector<FdmBlas3> FdmMgVector3;

//! Multigrid-syle 3-D linear system.
struct FdmMgLinearSystem3 {
    //! The system matrix.
    FdmMgMatrix3 A;

    //! The solution vector.
    FdmMgVector3 x;

    //! The RHS vector.
    FdmMgVector3 b;

    //! Clears the linear system.
    void clear();

    //! Returns the number of multigrid levels.
    size_t numberOfLevels() const;

    //! Resizes the system with the coarsest resolution and number of levels.
    void resizeWithCoarsest(const Size3 &coarsestResolution,
                            size_t numberOfLevels);

    //!
    //! \brief Resizes the system with the finest resolution and max number of
    //! levels.
    //!
    //! This function resizes the system with multiple levels until the
    //! resolution is divisible with 2^(level-1).
    //!
    //! \param finestResolution - The finest grid resolution.
    //! \param maxNumberOfLevels - Maximum number of multigrid levels.
    //!
    void resizeWithFinest(const Size3 &finestResolution,
                          size_t maxNumberOfLevels);
};

//! Multigrid utilities for 2-D FDM system.
class FdmMgUtils3 {
 public:
    //! Restricts given finer grid to the coarser grid.
    static void restrict(const FdmVector3 &finer, FdmVector3 *coarser);

    //! Corrects given coarser grid to the finer grid.
    static void correct(const FdmVector3 &coarser, FdmVector3 *finer);

    //! Resizes the array with the coarsest resolution and number of levels.
    template <typename T>
    static void resizeArrayWithCoarsest(const Size3 &coarsestResolution,
                                        size_t numberOfLevels,
                                        std::vector<Array3<T>> *levels);

    //!
    //! \brief Resizes the array with the finest resolution and max number of
    //! levels.
    //!
    //! This function resizes the system with multiple levels until the
    //! resolution is divisible with 2^(level-1).
    //!
    //! \param finestResolution - The finest grid resolution.
    //! \param maxNumberOfLevels - Maximum number of multigrid levels.
    //!
    template <typename T>
    static void resizeArrayWithFinest(const Size3 &finestResolution,
                                      size_t maxNumberOfLevels,
                                      std::vector<Array3<T>> *levels);
};

}  // namespace jet

#include "detail/fdm_mg_linear_system3-inl.h"

#endif  // INCLUDE_JET_FDM_MG_LINEAR_SYSTEM3_H_

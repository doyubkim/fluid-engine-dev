// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_MG_LINEAR_SYSTEM2_H_
#define INCLUDE_JET_FDM_MG_LINEAR_SYSTEM2_H_

#include <jet/face_centered_grid2.h>
#include <jet/fdm_linear_system2.h>
#include <jet/mg.h>

namespace jet {

//! Multigrid-style 2-D FDM matrix.
typedef MgMatrix<FdmBlas2> FdmMgMatrix2;

//! Multigrid-style 2-D FDM vector.
typedef MgVector<FdmBlas2> FdmMgVector2;

//! Multigrid-syle 2-D linear system.
struct FdmMgLinearSystem2 {
    //! The system matrix.
    FdmMgMatrix2 A;

    //! The solution vector.
    FdmMgVector2 x;

    //! The RHS vector.
    FdmMgVector2 b;

    //! Clears the linear system.
    void clear();

    //! Returns the number of multigrid levels.
    size_t numberOfLevels() const;

    //! Resizes the system with the coarsest resolution and number of levels.
    void resizeWithCoarsest(const Size2 &coarsestResolution,
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
    void resizeWithFinest(const Size2 &finestResolution,
                          size_t maxNumberOfLevels);
};

//! Multigrid utilities for 2-D FDM system.
class FdmMgUtils2 {
 public:
    //! Restricts given finer grid to the coarser grid.
    static void restrict(const FdmVector2 &finer, FdmVector2 *coarser);

    //! Corrects given coarser grid to the finer grid.
    static void correct(const FdmVector2 &coarser, FdmVector2 *finer);

    //! Resizes the array with the coarsest resolution and number of levels.
    template <typename T>
    static void resizeArrayWithCoarsest(const Size2 &coarsestResolution,
                                        size_t numberOfLevels,
                                        std::vector<Array2<T>> *levels);

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
    static void resizeArrayWithFinest(const Size2 &finestResolution,
                                      size_t maxNumberOfLevels,
                                      std::vector<Array2<T>> *levels);
};

}  // namespace jet

#include "detail/fdm_mg_linear_system2-inl.h"

#endif  // INCLUDE_JET_FDM_MG_LINEAR_SYSTEM2_H_

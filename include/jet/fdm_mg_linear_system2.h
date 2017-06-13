// Copyright (c) 2017 Doyub Kim
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

typedef MgMatrix<FdmBlas2> FdmMgMatrix2;
typedef MgVector<FdmBlas2> FdmMgVector2;

struct FdmMgLinearSystem2 {
    FdmMgMatrix2 A;
    FdmMgVector2 x, b;

    void clear();

    size_t numberOfLevels() const;

    void resizeWithCoarsest(const Size2 &coarsestResolution,
                            size_t numberOfLevels);
    void resizeWithFinest(const Size2 &finestResolution,
                          size_t maxNumberOfLevels);
};

class FdmMgUtils2 {
 public:
    static void restrict(const FdmVector2 &finer, FdmVector2 *coarser);

    static void correct(const FdmVector2 &coarser, FdmVector2 *finer);

    template <typename T>
    static void resizeArrayWithCoarsest(const Size2 &coarsestResolution,
                                        size_t numberOfLevels,
                                        std::vector<Array2<T>> *levels);

    template <typename T>
    static void resizeArrayWithFinest(const Size2 &finestResolution,
                                      size_t maxNumberOfLevels,
                                      std::vector<Array2<T>> *levels);

    static void jacobi(const FdmMatrix2 &A, const FdmVector2 &b,
                       unsigned int numberOfIterations, double maxTolerance,
                       FdmVector2 *x, FdmVector2 *xTemp);
};

}  // namespace jet

#include "detail/fdm_mg_linear_system2-inl.h"

#endif  // INCLUDE_JET_FDM_MG_LINEAR_SYSTEM2_H_

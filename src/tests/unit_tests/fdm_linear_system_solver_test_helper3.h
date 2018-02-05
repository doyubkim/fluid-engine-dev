// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_linear_system_solver3.h>

namespace jet {

class FdmLinearSystemSolverTestHelper3 {
 public:
    static void buildTestLinearSystem(FdmLinearSystem3* system,
                                      const Size3& size) {
        system->A.resize(size);
        system->x.resize(size);
        system->b.resize(size);

        system->A.forEachIndex([&](size_t i, size_t j, size_t k) {
            if (i > 0) {
                system->A(i, j, k).center += 1.0;
            }
            if (i < system->A.width() - 1) {
                system->A(i, j, k).center += 1.0;
                system->A(i, j, k).right -= 1.0;
            }

            if (j > 0) {
                system->A(i, j, k).center += 1.0;
            } else {
                system->b(i, j, k) += 1.0;
            }

            if (j < system->A.height() - 1) {
                system->A(i, j, k).center += 1.0;
                system->A(i, j, k).up -= 1.0;
            } else {
                system->b(i, j, k) -= 1.0;
            }

            if (k > 0) {
                system->A(i, j, k).center += 1.0;
            }
            if (k < system->A.depth() - 1) {
                system->A(i, j, k).center += 1.0;
                system->A(i, j, k).front -= 1.0;
            }
        });
    }

    static void buildTestCompressedLinearSystem(
        FdmCompressedLinearSystem3* system, const Size3& size) {
        Array3<size_t> coordToIndex(size);
        const auto acc = coordToIndex.constAccessor();

        coordToIndex.forEachIndex([&](size_t i, size_t j, size_t k) {
            const size_t cIdx = acc.index(i, j, k);

            coordToIndex[cIdx] = system->b.size();
            double bijk = 0.0;

            std::vector<double> row(1, 0.0);
            std::vector<size_t> colIdx(1, cIdx);

            if (i > 0) {
                const size_t lIdx = acc.index(i - 1, j, k);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(lIdx);
            }
            if (i < size.x - 1) {
                const size_t rIdx = acc.index(i + 1, j, k);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(rIdx);
            }

            if (j > 0) {
                const size_t dIdx = acc.index(i, j - 1, k);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(dIdx);
            } else {
                bijk += 1.0;
            }

            if (j < size.y - 1) {
                const size_t uIdx = acc.index(i, j + 1, k);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(uIdx);
            } else {
                bijk -= 1.0;
            }

            if (k > 0) {
                const size_t bIdx = acc.index(i, j, k - 1);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(bIdx);
            } else {
                bijk += 1.0;
            }

            if (k < size.z - 1) {
                const size_t fIdx = acc.index(i, j, k + 1);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(fIdx);
            } else {
                bijk -= 1.0;
            }

            system->A.addRow(row, colIdx);
            system->b.append(bijk);
        });

        system->x.resize(system->b.size(), 0.0);
    }
};

}  // namespace jet

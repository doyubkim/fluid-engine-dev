// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_linear_system_solver2.h>

namespace jet {

class FdmLinearSystemSolverTestHelper2 {
 public:
    static void buildTestLinearSystem(FdmLinearSystem2* system,
                                      const Size2& size) {
        system->A.resize(size);
        system->x.resize(size);
        system->b.resize(size);

        system->A.forEachIndex([&](size_t i, size_t j) {
            if (i > 0) {
                system->A(i, j).center += 1.0;
            }
            if (i < system->A.width() - 1) {
                system->A(i, j).center += 1.0;
                system->A(i, j).right -= 1.0;
            }

            if (j > 0) {
                system->A(i, j).center += 1.0;
            } else {
                system->b(i, j) += 1.0;
            }

            if (j < system->A.height() - 1) {
                system->A(i, j).center += 1.0;
                system->A(i, j).up -= 1.0;
            } else {
                system->b(i, j) -= 1.0;
            }
        });
    }

    static void buildTestCompressedLinearSystem(
        FdmCompressedLinearSystem2* system, const Size2& size) {
        Array2<size_t> coordToIndex(size);
        const auto acc = coordToIndex.constAccessor();

        coordToIndex.forEachIndex([&](size_t i, size_t j) {
            const size_t cIdx = acc.index(i, j);

            coordToIndex[cIdx] = system->b.size();
            double bij = 0.0;

            std::vector<double> row(1, 0.0);
            std::vector<size_t> colIdx(1, cIdx);

            if (i > 0) {
                const size_t lIdx = acc.index(i - 1, j);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(lIdx);
            }
            if (i < size.x - 1) {
                const size_t rIdx = acc.index(i + 1, j);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(rIdx);
            }

            if (j > 0) {
                const size_t dIdx = acc.index(i, j - 1);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(dIdx);
            } else {
                bij += 1.0;
            }

            if (j < size.y - 1) {
                const size_t uIdx = acc.index(i, j + 1);
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(uIdx);
            } else {
                bij -= 1.0;
            }

            system->A.addRow(row, colIdx);
            system->b.append(bij);
        });

        system->x.resize(system->b.size(), 0.0);
    }
};

}  // namespace jet

// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/parallel.h>

#include <thread>

static unsigned int sMaxNumberOfThreads = std::thread::hardware_concurrency();

namespace jet {

void setMaxNumberOfThreads(unsigned int numThreads) {
    sMaxNumberOfThreads = std::max(numThreads, 1u);
}

unsigned int maxNUmberOfThreads() {
    return sMaxNumberOfThreads;
}

}  // namespace jet

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/logging.h>

#include <benchmark/benchmark.h>

#include <fstream>

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    std::ofstream logFile("mem_perf_tests.log");
    if (logFile) {
        jet::Logging::setAllStream(&logFile);
    }

    ::benchmark::RunSpecifiedBenchmarks();
}

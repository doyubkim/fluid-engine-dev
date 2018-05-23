// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "mem_perf_tests.h"

#include <iostream>
#include <string>
#include <utility>

void printMemReport(double memUsage, const std::string& memMessage) {
    std::cout << "Mem usage: " << memUsage << ' ' << memMessage << '\n';
}

std::pair<double, std::string> makeReadableByteSize(size_t bytes) {
    double s = static_cast<double>(bytes);
    std::string unit = "B";

    if (s > 1024) {
        s /= 1024;
        unit = "kB";
    }

    if (s > 1024) {
        s /= 1024;
        unit = "MB";
    }

    if (s > 1024) {
        s /= 1024;
        unit = "GB";
    }

    if (s > 1024) {
        s /= 1024;
        unit = "TB";
    }

    return std::make_pair(s, unit);
}

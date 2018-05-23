// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_TESTS_PERF_TESTS_PERF_TESTS_H_
#define SRC_TESTS_PERF_TESTS_PERF_TESTS_H_

#include <string>
#include <utility>
#include <vector>

void printMemReport(double memUsage, const std::string& memMessage);

size_t getCurrentRSS();

std::pair<double, std::string> makeReadableByteSize(size_t bytes);

#endif  // SRC_TESTS_PERF_TESTS_PERF_TESTS_H_

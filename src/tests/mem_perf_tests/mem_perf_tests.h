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

// gtest hack
namespace testing {

namespace internal {

enum GTestColor {
    COLOR_DEFAULT,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_YELLOW
};

extern void ColoredPrintf(GTestColor color, const char* fmt, ...);

}  // namespace internal

}  // namespace testing

size_t getCurrentRSS();

std::pair<double, std::string> makeReadableByteSize(size_t bytes);

#define JET_PRINT_INFO(fmt, ...) \
    testing::internal::ColoredPrintf( \
        testing::internal::COLOR_YELLOW,  "[   STAT   ] "); \
    testing::internal::ColoredPrintf( \
        testing::internal::COLOR_YELLOW, fmt, __VA_ARGS__); \


#endif  // SRC_TESTS_PERF_TESTS_PERF_TESTS_H_

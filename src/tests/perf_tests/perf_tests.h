// Copyright (c) 2016 Doyub Kim

#ifndef SRC_TESTS_PERF_TESTS_PERF_TESTS_H_
#define SRC_TESTS_PERF_TESTS_PERF_TESTS_H_

#include <chrono>
#include <string>
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

#define JET_PRINT_INFO(fmt, ...) \
    testing::internal::ColoredPrintf( \
        testing::internal::COLOR_YELLOW,  "[----------] "); \
    testing::internal::ColoredPrintf( \
        testing::internal::COLOR_YELLOW, fmt, __VA_ARGS__); \


#endif  // SRC_TESTS_PERF_TESTS_PERF_TESTS_H_

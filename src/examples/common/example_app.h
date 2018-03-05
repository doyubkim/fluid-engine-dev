// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_COMMON_EXAMPLE_APP_H_
#define SRC_EXAMPLES_COMMON_EXAMPLE_APP_H_

#include "example.h"

#include <string>

class ExampleApp {
public:
    ExampleApp() = delete;

    static void initialize(const std::string& appName, int windowWidth, int windowHeight);

    static void finalize();

    template <typename ExampleType, typename... Args>
    static void addExample(Args... args) {
        addExample(std::make_shared<ExampleType>(args...));
    };

    static void addExample(const ExamplePtr& example);

    static void run();
};

#endif  // SRC_EXAMPLES_COMMON_EXAMPLE_APP_H_

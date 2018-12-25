// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_EXAMPLE_UTILS_GFX_EXAMPLE_APP_H_
#define SRC_EXAMPLES_EXAMPLE_UTILS_GFX_EXAMPLE_APP_H_

#include "gfx_example.h"

#include <string>

class GfxExampleManager {
 public:
    GfxExampleManager() = delete;

    static void initialize(const jet::gfx::WindowPtr& window);

    template <typename ExampleType, typename... Args>
    static void addExample(Args... args) {
        addExample(std::make_shared<ExampleType>(args...));
    };

    static void addExample(const GfxExamplePtr& example);
};

#endif  // SRC_EXAMPLES_EXAMPLE_UTILS_GFX_EXAMPLE_APP_H_

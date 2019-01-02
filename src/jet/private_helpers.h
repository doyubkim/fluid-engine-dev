// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_JET_PRIVATE_HELPERS_H_
#define SRC_JET_PRIVATE_HELPERS_H_

#include <jet/macros.h>

#ifndef UNUSED_VARIABLE
#   define UNUSED_VARIABLE(x) ((void)x)
#endif

#ifndef CLONE_W_CUSTOM_DELETER
#   define CLONE_W_CUSTOM_DELETER(ClassName) \
        std::shared_ptr<ClassName>( \
            new ClassName(*this), \
            [] (ClassName* obj) { \
                delete obj; \
            });
#endif

#endif  // SRC_JET_PRIVATE_HELPERS_H_

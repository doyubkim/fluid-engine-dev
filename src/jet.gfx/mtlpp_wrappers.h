// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_JET_VIZ_METAL_WRAPPERS_H_
#define SRC_JET_VIZ_METAL_WRAPPERS_H_

#import "3rdparty/mtlpp/mtlpp.hpp"

namespace jet {
namespace gfx {

#define MTLPP_WRAPPER_CLASS(Type)                                         \
    class MetalPrivate##Type final {                                      \
     public:                                                              \
        mtlpp::Type value;                                                \
        MetalPrivate##Type(const mtlpp::Type& val) : value(val) {}        \
        MetalPrivate##Type(mtlpp::Type&& val) { value = std::move(val); } \
    };

MTLPP_WRAPPER_CLASS(Buffer)
MTLPP_WRAPPER_CLASS(CommandQueue)
MTLPP_WRAPPER_CLASS(Device)
MTLPP_WRAPPER_CLASS(Function)
MTLPP_WRAPPER_CLASS(Library)
MTLPP_WRAPPER_CLASS(RenderPipelineState)

}  // namespace gfx
}  // namespace jet

#endif  // SRC_JET_VIZ_METAL_WRAPPERS_H_

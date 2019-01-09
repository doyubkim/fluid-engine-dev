// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#import "3rdparty/mtlpp/mtlpp.hpp"

namespace jet {
namespace gfx {

class MetalPrivateWindow : public ns::Object {
 public:
    MetalPrivateWindow() {}
    MetalPrivateWindow(const ns::Handle &handle) : ns::Object(handle) {}
};

class MetalPrivateView : public ns::Object {
 public:
    MetalPrivateView() {}
    MetalPrivateView(const ns::Handle &handle) : ns::Object(handle) {}
};

}  // namespace gfx
}  // namespace jet

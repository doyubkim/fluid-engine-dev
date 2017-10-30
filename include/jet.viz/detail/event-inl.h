// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_DETAIL_EVENT_INL_H_
#define INCLUDE_JET_VIZ_DETAIL_EVENT_INL_H_

namespace jet { namespace viz {

template <typename... EventArgTypes>
bool Event<EventArgTypes...>::operator()(EventArgTypes... args) {
    bool handled = false;
    for (auto& callback : _callbacks) {
        handled |= callback.second(args...);
    }
    return handled;
}

template <typename... EventArgTypes>
EventToken Event<EventArgTypes...>::operator+=(const CallbackType& callback) {
    _callbacks[++_lastToken] = callback;
    return _lastToken;
}

template <typename... EventArgTypes>
void Event<EventArgTypes...>::operator-=(EventToken token) {
    _callbacks.erase(token);
}

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_DETAIL_EVENT_INL_H_

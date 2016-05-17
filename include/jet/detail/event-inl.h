// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_DETAIL_EVENT_INL_H_
#define INCLUDE_JET_DETAIL_EVENT_INL_H_

namespace jet {

template <typename ...EventArgTypes>
void Event<EventArgTypes...>::operator()(EventArgTypes... args) {
    for (auto& callback : _callbacks) {
        callback.second(args...);
    }
}

template <typename ...EventArgTypes>
EventToken Event<EventArgTypes...>::operator+=(const CallbackType& callback) {
    _callbacks[++_lastToken] = callback;
    return _lastToken;
}

template <typename ...EventArgTypes>
void Event<EventArgTypes...>::operator-=(EventToken token) {
    _callbacks.erase(token);
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_EVENT_INL_H_

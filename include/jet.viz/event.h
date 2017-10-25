// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_EVENT_H_
#define INCLUDE_JET_VIZ_EVENT_H_

#include <functional>
#include <map>

namespace jet { namespace viz {
typedef std::size_t EventToken;

static const EventToken kEmptyEventToken = 0;

template <typename... EventArgTypes>
class Event {
 public:
    typedef std::function<void(EventArgTypes...)> CallbackType;

    void operator()(EventArgTypes... args);

    EventToken operator+=(const CallbackType& callback);

    void operator-=(EventToken token);

 private:
    EventToken _lastToken = 0;
    std::map<EventToken, CallbackType> _callbacks;
};

} }  // namespace jet::viz

#include "detail/event-inl.h"

#endif  // INCLUDE_JET_VIZ_EVENT_H_

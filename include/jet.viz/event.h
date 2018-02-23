// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_EVENT_H_
#define INCLUDE_JET_VIZ_EVENT_H_

#include <functional>
#include <map>

namespace jet {
namespace viz {

//! Event token (handle) type.
typedef size_t EventToken;

//! Empty event token.
static const EventToken kEmptyEventToken = 0;

//!
//! \brief Generic event type.
//!
//! This class represents a generic event source that can send signals to
//! multiple callback functions.
//!
template <typename... EventArgTypes>
class Event {
 public:
    //! \brief Callback function type.
    //! The callback function type that can be attached to the event source.
    //! Returns true if an event is handled.
    typedef std::function<bool(EventArgTypes...)> CallbackType;

    //!
    //! Invokes callback functions with given arguments.
    //!
    //! \param args Arguments to pass to the attached callback functions.
    //! \return True if one ore more callback functions handled the event.
    //!
    bool operator()(EventArgTypes... args);

    //!
    //! Attaches given callback function.
    //!
    //! \param callback Callback function to attach.
    //! \return Event token (handle) for the attached callback.
    //!
    EventToken operator+=(const CallbackType& callback);

    //!
    //! Detaches the callback function.
    //!
    //! \param token Event token for the callback function to detach.
    //!
    void operator-=(EventToken token);

 private:
    EventToken _lastToken = 0;
    std::map<EventToken, CallbackType> _callbacks;
};

}  // namespace viz
}  // namespace jet

#include "detail/event-inl.h"

#endif  // INCLUDE_JET_VIZ_EVENT_H_

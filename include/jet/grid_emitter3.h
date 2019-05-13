// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_EMITTER3_H_
#define INCLUDE_JET_GRID_EMITTER3_H_

#include <jet/animation.h>
#include <jet/implicit_surface3.h>
#include <jet/scalar_grid3.h>

#include <utility>
#include <vector>

namespace jet {

//!
//! \brief Abstract base class for 3-D grid-based emitters.
//!
class GridEmitter3 {
 public:
    //!
    //! \brief Callback function type for update calls.
    //!
    //! This type of callback function will take the current time and time
    //! interval in seconds.
    //!
    typedef std::function<void(double, double)> OnBeginUpdateCallback;

    //! Constructs an emitter.
    GridEmitter3();

    //! Destructor.
    virtual ~GridEmitter3();

    //! Updates the emitter state from \p currentTimeInSeconds to the following
    //! time-step.
    void update(double currentTimeInSeconds, double timeIntervalInSeconds);

    //! Returns true if the emitter is enabled.
    bool isEnabled() const;

    //! Sets true/false to enable/disable the emitter.
    void setIsEnabled(bool enabled);

    //!
    //! \brief      Sets the callback function to be called when
    //!             GridEmitter3::update function is invoked.
    //!
    //! The callback function takes current simulation time in seconds unit. Use
    //! this callback to track any motion or state changes related to this
    //! emitter.
    //!
    //! \param[in]  callback The callback function.
    //!
    void setOnBeginUpdateCallback(const OnBeginUpdateCallback& callback);

 protected:
    virtual void onUpdate(double currentTimeInSeconds,
                          double timeIntervalInSeconds) = 0;

 private:
    bool _isEnabled = true;
    OnBeginUpdateCallback _onBeginUpdateCallback;
};

//! Shared pointer type for the GridEmitter3.
typedef std::shared_ptr<GridEmitter3> GridEmitter3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_EMITTER3_H_

// Copyright (c) 2010 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PARTICLE_EMITTER2_H_
#define INCLUDE_JET_PARTICLE_EMITTER2_H_

#include <jet/animation.h>
#include <jet/particle_system_data2.h>

namespace jet {

//!
//! \brief Abstract base class for 2-D particle emitter.
//!
class ParticleEmitter2 {
 public:
    //!
    //! \brief Callback function type for update calls.
    //!
    //! This type of callback function will take the emitter pointer, current
    //! time, and time interval in seconds.
    //!
    typedef std::function<void(ParticleEmitter2*, double, double)>
        OnBeginUpdateCallback;

    //! Default constructor.
    ParticleEmitter2();

    //! Destructor.
    virtual ~ParticleEmitter2();

    //! Updates the emitter state from \p currentTimeInSeconds to the following
    //! time-step.
    void update(double currentTimeInSeconds, double timeIntervalInSeconds);

    //! Returns the target particle system to emit.
    const ParticleSystemData2Ptr& target() const;

    //! Sets the target particle system to emit.
    void setTarget(const ParticleSystemData2Ptr& particles);

    //! Returns true if the emitter is enabled.
    bool isEnabled() const;

    //! Sets true/false to enable/disable the emitter.
    void setIsEnabled(bool enabled);

    //!
    //! \brief      Sets the callback function to be called when
    //!             ParticleEmitter2::update function is invoked.
    //!
    //! The callback function takes current simulation time in seconds unit. Use
    //! this callback to track any motion or state changes related to this
    //! emitter.
    //!
    //! \param[in]  callback The callback function.
    //!
    void setOnBeginUpdateCallback(const OnBeginUpdateCallback& callback);

 protected:
    //! Called when ParticleEmitter3::setTarget is executed.
    virtual void onSetTarget(const ParticleSystemData2Ptr& particles);

    //! Called when ParticleEmitter3::update is executed.
    virtual void onUpdate(
        double currentTimeInSeconds,
        double timeIntervalInSeconds) = 0;

 private:
    bool _isEnabled = true;
    ParticleSystemData2Ptr _particles;
    OnBeginUpdateCallback _onBeginUpdateCallback;
};

//! Shared pointer for the ParticleEmitter2 type.
typedef std::shared_ptr<ParticleEmitter2> ParticleEmitter2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_EMITTER2_H_

// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PARTICLE_EMITTER3_H_
#define INCLUDE_JET_PARTICLE_EMITTER3_H_

#include <jet/animation.h>
#include <jet/particle_system_data3.h>

namespace jet {

//!
//! \brief Abstract base class for 3-D particle emitter.
//!
class ParticleEmitter3 {
 public:
    //!
    //! \brief Callback function type for update calls.
    //!
    //! This type of callback function will take the emitter pointer, current
    //! time, and time interval in seconds.
    //!
    typedef std::function<void(ParticleEmitter3*, double, double)>
        OnBeginUpdateCallback;

    //! Default constructor.
    ParticleEmitter3();

    //! Destructor.
    virtual ~ParticleEmitter3();

    //! Updates the emitter state from \p currentTimeInSeconds to the following
    //! time-step.
    void update(double currentTimeInSeconds, double timeIntervalInSeconds);

    //! Returns the target particle system to emit.
    const ParticleSystemData3Ptr& target() const;

    //! Sets the target particle system to emit.
    void setTarget(const ParticleSystemData3Ptr& particles);

    //!
    //! \brief      Sets the callback function to be called when
    //!             ParticleEmitter3::update function is invoked.
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
    virtual void onSetTarget(const ParticleSystemData3Ptr& particles);

    //! Called when ParticleEmitter3::update is executed.
    virtual void onUpdate(
        double currentTimeInSeconds,
        double timeIntervalInSeconds) = 0;

 private:
    ParticleSystemData3Ptr _particles;
    OnBeginUpdateCallback _onBeginUpdateCallback;
};

//! Shared pointer for the ParticleEmitter3 type.
typedef std::shared_ptr<ParticleEmitter3> ParticleEmitter3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_EMITTER3_H_

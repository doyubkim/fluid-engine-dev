// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PARTICLE_EMITTER_H_
#define INCLUDE_JET_PARTICLE_EMITTER_H_

#include <jet/animation.h>
#include <jet/particle_system_data.h>

namespace jet {

//!
//! \brief Abstract base class for N-D particle emitter.
//!
template <size_t N>
class ParticleEmitter {
 public:
    //!
    //! \brief Callback function type for update calls.
    //!
    //! This type of callback function will take the emitter pointer, current
    //! time, and time interval in seconds.
    //!
    typedef std::function<void(ParticleEmitter*, double, double)>
        OnBeginUpdateCallback;

    //! Default constructor.
    ParticleEmitter();

    //! Destructor.
    virtual ~ParticleEmitter();

    //! Updates the emitter state from \p currentTimeInSeconds to the following
    //! time-step.
    void update(double currentTimeInSeconds, double timeIntervalInSeconds);

    //! Returns the target particle system to emit.
    const ParticleSystemDataPtr<N>& target() const;

    //! Sets the target particle system to emit.
    void setTarget(const ParticleSystemDataPtr<N>& particles);

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
    virtual void onSetTarget(const ParticleSystemDataPtr<N>& particles);

    //! Called when ParticleEmitter3::update is executed.
    virtual void onUpdate(double currentTimeInSeconds,
                          double timeIntervalInSeconds) = 0;

 private:
    ParticleSystemDataPtr<N> _particles;
    OnBeginUpdateCallback _onBeginUpdateCallback;
};

//! 2-D ParticleEmitter type.
using ParticleEmitter2 = ParticleEmitter<2>;

//! 3-D ParticleEmitter type.
using ParticleEmitter3 = ParticleEmitter<3>;

//! N-D shared pointer type of ParticleEmitter.
template <size_t N>
using ParticleEmitterPtr = std::shared_ptr<ParticleEmitter<N>>;

//! Shared pointer type of ParticleEmitter2.
using ParticleEmitter2Ptr = ParticleEmitterPtr<2>;

//! Shared pointer type of ParticleEmitter3.
using ParticleEmitter3Ptr = ParticleEmitterPtr<3>;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_EMITTER_H_

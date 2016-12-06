// Copyright (c) 2016 Doyub Kim

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
    //! Default constructor.
    ParticleEmitter2();

    //! Destructor.
    virtual ~ParticleEmitter2();

    //!
    //! \brief      Emits particles to the particle system data.
    //!
    //! \param[in]  frame     Current animation frame.
    //! \param[in]  particles The particle system data.
    //!
    virtual void emit(
        const Frame& frame,
        const ParticleSystemData2Ptr& particles) = 0;
};

typedef std::shared_ptr<ParticleEmitter2> ParticleEmitter2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_EMITTER2_H_

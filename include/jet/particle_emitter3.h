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
    //! Default constructor.
    ParticleEmitter3();

    //! Destructor.
    virtual ~ParticleEmitter3();

    //!
    //! \brief      Emits particles to the particle system data.
    //!
    //! \param[in]  frame     Current animation frame.
    //! \param[in]  particles The particle system data.
    //!
    virtual void emit(
        const Frame& frame,
        const ParticleSystemData3Ptr& particles) = 0;
};

//! Shared pointer for the ParticleEmitter3 type.
typedef std::shared_ptr<ParticleEmitter3> ParticleEmitter3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_EMITTER3_H_

// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PARTICLE_EMITTER3_H_
#define INCLUDE_JET_PARTICLE_EMITTER3_H_

#include <jet/animation.h>
#include <jet/particle_system_data3.h>

#include <limits>
#include <random>

namespace jet {

class ParticleEmitter3 {
 public:
    ParticleEmitter3();

    virtual ~ParticleEmitter3();

    virtual void emit(
        const Frame& frame,
        const ParticleSystemData3Ptr& particles) = 0;
};

typedef std::shared_ptr<ParticleEmitter3> ParticleEmitter3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_EMITTER3_H_

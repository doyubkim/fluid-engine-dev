// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PARTICLE_EMITTER2_H_
#define INCLUDE_JET_PARTICLE_EMITTER2_H_

#include <jet/animation.h>
#include <jet/particle_system_data2.h>

#include <limits>
#include <random>

namespace jet {

class ParticleEmitter2 {
 public:
    ParticleEmitter2();

    virtual ~ParticleEmitter2();

    virtual void emit(
        const Frame& frame,
        const ParticleSystemData2Ptr& particles) = 0;
};

typedef std::shared_ptr<ParticleEmitter2> ParticleEmitter2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_EMITTER2_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PARTICLE_EMITTER_SET2_H_
#define INCLUDE_JET_PARTICLE_EMITTER_SET2_H_

#include <jet/particle_emitter2.h>
#include <tuple>
#include <vector>

namespace jet {

//!
//! \brief 2-D particle-based emitter set.
//!
class ParticleEmitterSet2 final : public ParticleEmitter2 {
 public:
    class Builder;

    //! Constructs an emitter.
    ParticleEmitterSet2();

    //! Constructs an emitter with sub-emitters.
    explicit ParticleEmitterSet2(
        const std::vector<ParticleEmitter2Ptr>& emitters);

    //! Destructor.
    virtual ~ParticleEmitterSet2();

    //! Adds sub-emitter.
    void addEmitter(const ParticleEmitter2Ptr& emitter);

    //! Returns builder fox ParticleEmitterSet2.
    static Builder builder();

 private:
    std::vector<ParticleEmitter2Ptr> _emitters;

    void onSetTarget(const ParticleSystemData2Ptr& particles) override;

    void onUpdate(
        double currentTimeInSeconds,
        double timeIntervalInSecond) override;
};

//! Shared pointer type for the ParticleEmitterSet2.
typedef std::shared_ptr<ParticleEmitterSet2> ParticleEmitterSet2Ptr;


//!
//! \brief Front-end to create ParticleEmitterSet2 objects step by step.
//!
class ParticleEmitterSet2::Builder final {
 public:
    //! Returns builder with list of sub-emitters.
    Builder& withEmitters(const std::vector<ParticleEmitter2Ptr>& emitters);

    //! Builds ParticleEmitterSet2.
    ParticleEmitterSet2 build() const;

    //! Builds shared pointer of ParticleEmitterSet2 instance.
    ParticleEmitterSet2Ptr makeShared() const;

 private:
    std::vector<ParticleEmitter2Ptr> _emitters;
};

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_EMITTER_SET2_H_

// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_EMITTER_SET3_H_
#define INCLUDE_JET_GRID_EMITTER_SET3_H_

#include <jet/grid_emitter3.h>
#include <tuple>
#include <vector>

namespace jet {

//!
//! \brief 3-D grid-based emitter set.
//!
class GridEmitterSet3 final : public GridEmitter3 {
 public:
    class Builder;

    //! Constructs an emitter.
    GridEmitterSet3();

    //! Constructs an emitter with sub-emitters.
    explicit GridEmitterSet3(const std::vector<GridEmitter3Ptr>& emitters);

    //! Destructor.
    virtual ~GridEmitterSet3();

    //! Adds sub-emitter.
    void addEmitter(const GridEmitter3Ptr& emitter);

    //! Returns builder fox GridEmitterSet3.
    static Builder builder();

 private:
    std::vector<GridEmitter3Ptr> _emitters;

    void onUpdate(
        double currentTimeInSeconds,
        double timeIntervalInSeconds) override;
};

//! Shared pointer type for the GridEmitterSet3.
typedef std::shared_ptr<GridEmitterSet3> GridEmitterSet3Ptr;


//!
//! \brief Front-end to create GridEmitterSet3 objects step by step.
//!
class GridEmitterSet3::Builder final {
 public:
    //! Returns builder with list of sub-emitters.
    Builder& withEmitters(const std::vector<GridEmitter3Ptr>& emitters);

    //! Builds GridEmitterSet3.
    GridEmitterSet3 build() const;

    //! Builds shared pointer of GridEmitterSet3 instance.
    GridEmitterSet3Ptr makeShared() const;

 private:
    std::vector<GridEmitter3Ptr> _emitters;
};

}  // namespace jet

#endif  // INCLUDE_JET_GRID_EMITTER_SET3_H_

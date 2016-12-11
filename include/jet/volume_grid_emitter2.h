// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VOLUME_GRID_EMITTER2_H_
#define INCLUDE_JET_VOLUME_GRID_EMITTER2_H_

#include <jet/grid_emitter2.h>

#include <tuple>
#include <vector>

namespace jet {

//!
//! \brief 2-D grid-based volumetric emitter.
//!
class VolumeGridEmitter2 final : public GridEmitter2 {
 public:
    class Builder;

    //! For given signed-distance and location, maps to a scalar value.
    typedef std::function<double(double, const Vector2D&)> ScalarMapper;

    //! For given signed-distance and location, maps to a vector value.
    typedef std::function<Vector2D(double, const Vector2D&)> VectorMapper;

    //! For given old and new scalar values, returns a blended result.
    typedef std::function<double(double, double)> ScalarBlender;

    //! For given old and new vector values, returns a blended result.
    typedef std::function<Vector2D(Vector2D, Vector2D)> VectorBlender;

    //! Constructs an emitter with a source and is-one-shot flag.
    explicit VolumeGridEmitter2(
        const ImplicitSurface2Ptr& sourceRegion,
        bool isOneShot = true);

    //! Destructor.
    virtual ~VolumeGridEmitter2();

    //! Adds signed-distance target to the scalar grid.
    void addSignedDistanceTarget(const ScalarGrid2Ptr& scalarGridTarget);

    //!
    //! \brief      Adds step function target to the scalar grid.
    //!
    //! \param[in]  scalarGridTarget The scalar grid target.
    //! \param[in]  minValue         The minimum value of the step function.
    //! \param[in]  maxValue         The maximum value of the step function.
    //!
    void addStepFunctionTarget(
        const ScalarGrid2Ptr& scalarGridTarget,
        double minValue,
        double maxValue);

    //!
    //! \brief      Adds a scalar grid target.
    //!
    //! This function adds a custom target to the emitter. The first parameter
    //! defines which grid should it write to. The second parameter,
    //! \p customMapper, defines how to map sigend-distance field from the
    //! volume geometry and location of the point to the final scalar value that
    //! is going to be written to the target grid. The third parameter defines
    //! how to blend the old value from the target grid and the new value from
    //! the mapper function.
    //!
    //! \param[in]  scalarGridTarget The scalar grid target
    //! \param[in]  customMapper     The custom mapper.
    //! \param[in]  blender          The blender function.
    //!
    void addTarget(
        const ScalarGrid2Ptr& scalarGridTarget,
        const ScalarMapper& customMapper,
        const ScalarBlender& blender);

    //!
    //! \brief      Adds a vector grid target.
    //!
    //! This function adds a custom target to the emitter. The first parameter
    //! defines which grid should it write to. The second parameter,
    //! \p customMapper, defines how to map sigend-distance field from the
    //! volume geometry and location of the point to the final vector value that
    //! is going to be written to the target grid. The third parameter defines
    //! how to blend the old value from the target grid and the new value from
    //! the mapper function.
    //!
    //! \param[in]  scalarGridTarget The vector grid target
    //! \param[in]  customMapper     The custom mapper.
    //! \param[in]  blender          The blender function.
    //!
    void addTarget(
        const VectorGrid2Ptr& vectorGridTarget,
        const VectorMapper& customMapper,
        const VectorBlender& blender);

    //! Returns implicit surface which defines the source region.
    const ImplicitSurface2Ptr& sourceRegion() const;

    //! Returns true if this emits only once.
    bool isOneShot() const;

    //! Returns builder fox VolumeGridEmitter2.
    static Builder builder();

 private:
    typedef std::tuple<ScalarGrid2Ptr, ScalarMapper, ScalarBlender>
        ScalarTarget;
    typedef std::tuple<VectorGrid2Ptr, VectorMapper, VectorBlender>
        VectorTarget;

    ImplicitSurface2Ptr _sourceRegion;
    bool _isOneShot = true;
    bool _hasEmitted = false;
    std::vector<ScalarTarget> _customScalarTargets;
    std::vector<VectorTarget> _customVectorTargets;

    void onUpdate(
        double currentTimeInSeconds,
        double timeIntervalInSeconds) override;

    void emit();
};

//! Shared pointer type for the VolumeGridEmitter2.
typedef std::shared_ptr<VolumeGridEmitter2> VolumeGridEmitter2Ptr;


//!
//! \brief Front-end to create VolumeGridEmitter2 objects step by step.
//!
class VolumeGridEmitter2::Builder final {
 public:
    //! Returns builder with surface defining source region.
    Builder& withSourceRegion(const Surface2Ptr& sourceRegion);

    //! Returns builder with one-shot flag.
    Builder& withIsOneShot(bool isOneShot);

    //! Builds VolumeGridEmitter2.
    VolumeGridEmitter2 build() const;

    //! Builds shared pointer of VolumeGridEmitter2 instance.
    VolumeGridEmitter2Ptr makeShared() const {
        return std::make_shared<VolumeGridEmitter2>(
            _sourceRegion,
            _isOneShot);
    }

 private:
    ImplicitSurface2Ptr _sourceRegion;
    bool _isOneShot = true;
};

}  // namespace jet

#endif  // INCLUDE_JET_VOLUME_GRID_EMITTER2_H_

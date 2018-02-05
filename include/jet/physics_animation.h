// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PHYSICS_ANIMATION_H_
#define INCLUDE_JET_PHYSICS_ANIMATION_H_

#include <jet/animation.h>

namespace jet {

//!
//! \brief      Abstract base class for physics-based animation.
//!
//! This class represents physics-based animation by adding time-integration
//! specific functions to Animation class.
//!
class PhysicsAnimation : public Animation {
 public:
    //! Default constructor.
    PhysicsAnimation();

    //! Destructor.
    virtual ~PhysicsAnimation();

    //!
    //! \brief      Returns true if fixed sub-timestepping is used.
    //!
    //! When performing a time-integration, it is often required to take
    //! sub-timestepping for better results. The sub-stepping can be either
    //! fixed rate or adaptive, and this function returns which feature is
    //! currently selected.
    //!
    //! \return     True if using fixed sub time steps, false otherwise.
    //!
    bool isUsingFixedSubTimeSteps() const;

    //!
    //! \brief      Sets true if fixed sub-timestepping is used.
    //!
    //! When performing a time-integration, it is often required to take
    //! sub-timestepping for better results. The sub-stepping can be either
    //! fixed rate or adaptive, and this function sets which feature should be
    //! selected.
    //!
    //! \param[in]   isUsing True to enable fixed sub-stepping.
    //!
    void setIsUsingFixedSubTimeSteps(bool isUsing);

    //!
    //! \brief      Returns the number of fixed sub-timesteps.
    //!
    //! When performing a time-integration, it is often required to take
    //! sub-timestepping for better results. The sub-stepping can be either
    //! fixed rate or adaptive, and this function returns the number of fixed
    //! sub-steps.
    //!
    //! \return     The number of fixed sub-timesteps.
    //!
    unsigned int numberOfFixedSubTimeSteps() const;

    //!
    //! \brief      Sets the number of fixed sub-timesteps.
    //!
    //! When performing a time-integration, it is often required to take
    //! sub-timestepping for better results. The sub-stepping can be either
    //! fixed rate or adaptive, and this function sets the number of fixed
    //! sub-steps.
    //!
    //! \param[in]  numberOfSteps The number of fixed sub-timesteps.
    //!
    void setNumberOfFixedSubTimeSteps(unsigned int numberOfSteps);

    //! Advances a single frame.
    void advanceSingleFrame();

    //!
    //! \brief      Returns current frame.
    //!
    Frame currentFrame() const;

    //!
    //! \brief      Sets current frame cursor (but do not invoke update()).
    //!
    void setCurrentFrame(const Frame& frame);

    //!
    //! \brief      Returns current time in seconds.
    //!
    //! This function returns the current time which is calculated by adding
    //! current frame + sub-timesteps it passed.
    //!
    double currentTimeInSeconds() const;

 protected:
    //!
    //! \brief      Called when a single time-step should be advanced.
    //!
    //! When Animation::update function is called, this class will internally
    //! subdivide a frame into sub-steps if needed. Each sub-step, or time-step,
    //! is then taken to move forward in time. This function is called for each
    //! time-step, and a subclass that inherits PhysicsAnimation class should
    //! implement this function for its own physics model.
    //!
    //! \param[in]  timeIntervalInSeconds The time interval in seconds
    //!
    virtual void onAdvanceTimeStep(double timeIntervalInSeconds) = 0;

    //!
    //! \brief      Returns the required number of sub-timesteps for given time
    //!             interval.
    //!
    //! The required number of sub-timestep can be different depending on the
    //! physics model behind the implementation. Override this function to
    //! implement own logic for model specific sub-timestepping for given
    //! time interval.
    //!
    //! \param[in]  timeIntervalInSeconds The time interval in seconds.
    //!
    //! \return     The required number of sub-timesteps.
    //!
    virtual unsigned int numberOfSubTimeSteps(
        double timeIntervalInSeconds) const;

    //!
    //! \brief      Called at frame 0 to initialize the physics state.
    //!
    //! Inheriting classes can override this function to setup initial condition
    //! for the simulation.
    //!
    virtual void onInitialize();

 private:
    Frame _currentFrame;
    bool _isUsingFixedSubTimeSteps = true;
    unsigned int _numberOfFixedSubTimeSteps = 1;
    double _currentTime = 0.0;

    void onUpdate(const Frame& frame) final;

    void advanceTimeStep(double timeIntervalInSeconds);

    void initialize();
};

typedef std::shared_ptr<PhysicsAnimation> PhysicsAnimationPtr;

}  // namespace jet

#endif  // INCLUDE_JET_PHYSICS_ANIMATION_H_

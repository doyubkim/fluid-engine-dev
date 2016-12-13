# Introduction

To test and demonstrate the features of Jet, the SDK comes with a manual test application. The tests are generally more complicated than the unit tests that involves longer run time and collection of API calls. The manual tests are closer to the real-world use cases, so it is useful to take them as SDK examples. The tests normally output data files which can often be rendered into images or even movie clips

To list the entire test cases from Mac OS X or Linux, run

```
bin/list_manual_tests
```

or for Windows, run

```
bin\list_manual_tests.bat
```

Similar to the unit test, run the following command to run entire tests for Mac OS X and Linux:

```
bin/manual_tests <name_of_the_test>
```

For Windows, run

```
bin\manual_tests.bat <name_of_the_test>
```

You can run the entire tests by not specifying the name of the test. However, it will cost more than an hour of execution time since the manual test includes quite intensive tests such as running multiple fluid simulations. Thus, it is recommended to run specific tests for the fast debugging, and then run the entire tests for final validation. Similar to the unit test, you can also use patterns for specifying the tests such as:

```
bin/manual_tests AnimationTests.*
```

Again, replace `bin/manual_tests` with the `.bat` command for Windows.


The test results will be located at `manual_tests_output/TestName/CaseName/file`. To validate the results, you need [Matplotlib](http://matplotlib.org/). The recommended way of installing the latest version of the library is to use `pip` such as:

```
pip install matplotlib
```

The modern Python versions (2.7.9 and above) comes with `pip` by default. Once Matplotlib is installed, run the following:

```
bin/render_manual_tests_output
```

Once renderered, the rendered image will be stored at the same directory where the test output files are located (`manual_tests_output/TestName/CaseName/file`). Also, to render the animations as mpeg movie files, [ffmpeg](https://www.ffmpeg.org/) is required for Mac OS X and Windows. For Linux, [mencoder](http://www.mplayerhq.hu/) is needed. For Mac OS X, ffmpeg can be installed via Homebrew. For Windows, the executable can be downloaded from the [website](https://www.ffmpeg.org/). For Ubuntu, you can use `apt-get`.

For example, after running the following commands:

```
bin/manual_tests AnimationTests.OnUpdateSine
bin/render_manual_tests_output
```

you can find

```
manual_tests_output/Animation/OnUpdateSine/data.#line2.mp4
```

is generated.

# List of the Tests

Below is the list of the tests and their expected results.

* SemiLagrangian2Tests.
    * Boundary

        Before and after advection:

        ![Before Advection](img/manual_tests/SemiLagrangian2/Boundary/before.png "Before Advection")
        ![After Advection](img/manual_tests/SemiLagrangian2/Boundary/after.png "After Advection")

    * Zalesak

        Before and after advection:

        ![Before Advection](img/manual_tests/SemiLagrangian2/Zalesak/before.png "Before Advection")
        ![After Advection](img/manual_tests/SemiLagrangian2/Zalesak/after_628.png "After Advection")

* CubicSemiLagrangian2Tests.
    * Zalesak

        Before and after advection:

        ![Before Advection](img/manual_tests/CubicSemiLagrangian2/Zalesak/before.png "Before Advection")
        ![After Advection](img/manual_tests/CubicSemiLagrangian2/Zalesak/after_628.png "After Advection")

* AnimationTests.
    * OnUpdateSine

        ![Data](img/manual_tests/Animation/OnUpdateSine/data.gif "Data")

    * OnUpdateSineWithDecay

        ![Data](img/manual_tests/Animation/OnUpdateSineWithDecay/data.gif "Data")

* ArrayUtilsTests.
    * ExtralateToRegion2

* ScalarField3Tests.
    * Sample

        ![Data](img/manual_tests/ScalarField3/Sample/data.png "Data")

    * Gradient

        ![Data](img/manual_tests/ScalarField3/Gradient/data.png "Data")

    * Laplacian

        ![Data](img/manual_tests/ScalarField3/Laplacian/data.png "Data")

* VectorField3Tests.
    * Sample

        ![Data](img/manual_tests/VectorField3/Sample/data.png "Data")

    * Divergence

        ![Data](img/manual_tests/VectorField3/Divergence/data.png "Data")

    * Curl

        ![Data](img/manual_tests/VectorField3/Curl/data.png "Data")

    * Sample2

        Testing sample function with different field.

        ![Data](img/manual_tests/VectorField3/Sample2/data.png "Data")

* FlipSolver2Tests.
    * Empty

    * SteadyState

        ![SteadyState](img/manual_tests/FlipSolver2/SteadyState/data.gif "SteadyState")

    * DamBreaking

        ![DamBreaking](img/manual_tests/FlipSolver2/DamBreaking/data.gif "DamBreaking")

    * DamBreakingWithCollider

        ![DamBreakingWithCollider](img/manual_tests/FlipSolver2/DamBreakingWithCollider/data.gif "DamBreakingWithCollider")

* FlipSolver3Tests.
    * WaterDrop

        ![WaterDrop](img/manual_tests/FlipSolver3/WaterDrop/data.gif "WaterDrop")

    * DamBreakingWithCollider

        ![DamBreakingWithCollider](img/manual_tests/FlipSolver3/DamBreakingWithCollider/data.gif "DamBreakingWithCollider")

* FmmLevelSetSolver2Tests.
    * ReinitializeSmall

        2-D FMM test on low-resolution grid.

        Reinitializing constant field (should result the same constant field), before and after:

        ![ConstantBefore](img/manual_tests/FmmLevelSetSolver2/ReinitializeSmall/constant0.png "ConstantBefore")
        ![ConstantAfter](img/manual_tests/FmmLevelSetSolver2/ReinitializeSmall/constant1.png "ConstantAfter")

        Reinitializing SDF (should result nearly the same SDF), before and after:

        ![SDFBefore](img/manual_tests/FmmLevelSetSolver2/ReinitializeSmall/sdf0.png "SDFBefore")
        ![SDFAfter](img/manual_tests/FmmLevelSetSolver2/ReinitializeSmall/sdf1.png "SDFAfter")

        Reinitializing 2x scaled SDF (should result 1x scale SDF), before and after:

        ![ScaledBefore](img/manual_tests/FmmLevelSetSolver2/ReinitializeSmall/scaled0.png "ScaledBefore")
        ![ScaledAfter](img/manual_tests/FmmLevelSetSolver2/ReinitializeSmall/scaled1.png "ScaledAfter")

        Reinitializing unit step function (should result 1x scale SDF), before and after:

        ![UnitStepBefore](img/manual_tests/FmmLevelSetSolver2/ReinitializeSmall/unit_step0.png "UnitStepBefore")
        ![UnitStepAfter](img/manual_tests/FmmLevelSetSolver2/ReinitializeSmall/unit_step1.png "UnitStepAfter")

    * Reinitialize

        2-D FMM test on high-resolution grid.

        Reinitializing constant field (should result the same constant field), before and after:

        ![ConstantBefore](img/manual_tests/FmmLevelSetSolver2/Reinitialize/constant0.png "ConstantBefore")
        ![ConstantAfter](img/manual_tests/FmmLevelSetSolver2/ReinitializeSmall/constant1.png "ConstantAfter")

        Reinitializing SDF (should result nearly the same SDF), before and after:

        ![SDFBefore](img/manual_tests/FmmLevelSetSolver2/Reinitialize/sdf0.png "SDFBefore")
        ![SDFAfter](img/manual_tests/FmmLevelSetSolver2/Reinitialize/sdf1.png "SDFAfter")

        Reinitializing 2x scaled SDF (should result 1x scale SDF), before and after:

        ![ScaledBefore](img/manual_tests/FmmLevelSetSolver2/Reinitialize/scaled0.png "ScaledBefore")
        ![ScaledAfter](img/manual_tests/FmmLevelSetSolver2/Reinitialize/scaled1.png "ScaledAfter")

        Reinitializing unit step function (should result 1x scale SDF), before and after:

        ![UnitStepBefore](img/manual_tests/FmmLevelSetSolver2/Reinitialize/unit_step0.png "UnitStepBefore")
        ![UnitStepAfter](img/manual_tests/FmmLevelSetSolver2/Reinitialize/unit_step1.png "UnitStepAfter")

    * Extrapolate

        Input and output field:

        ![Input](img/manual_tests/FmmLevelSetSolver2/Extrapolate/input.png "Input")
        ![Output](img/manual_tests/FmmLevelSetSolver2/Extrapolate/output.png "Output")

        Background signed-distance field:

        ![SDF](img/manual_tests/FmmLevelSetSolver2/Extrapolate/sdf.png "SDF")

* FmmLevelSetSolver3Tests.
    * ReinitializeSmall

        Input and output field (cross-sectional view):

        ![Input](img/manual_tests/FmmLevelSetSolver3/ReinitializeSmall/input.png "Input")
        ![Output](img/manual_tests/FmmLevelSetSolver3/ReinitializeSmall/output.png "Output")

    * ExtrapolateSmall

        Input and output field (cross-sectional view):

        ![Input](img/manual_tests/FmmLevelSetSolver3/ExtrapolateSmall/input.png "Input")
        ![Output](img/manual_tests/FmmLevelSetSolver3/ExtrapolateSmall/output.png "Output")

* GridBlockedBoundaryConditionSolver2Tests.
    * ConstrainVelocitySmall

        Constrained velocity and boundary marker:

        ![Data](img/manual_tests/GridBlockedBoundaryConditionSolver2/ConstrainVelocitySmall/data.png "Data")
        ![Marker](img/manual_tests/GridBlockedBoundaryConditionSolver2/ConstrainVelocitySmall/marker.png "Marker")

    * ConstrainVelocity

        Constrained velocity and boundary marker:

        ![Data](img/manual_tests/GridBlockedBoundaryConditionSolver2/ConstrainVelocity/data.png "Data")
        ![Marker](img/manual_tests/GridBlockedBoundaryConditionSolver2/ConstrainVelocity/marker.png "Marker")

    * ConstrainVelocityWithFriction

        Constrained velocity and boundary marker:

        ![Data](img/manual_tests/GridBlockedBoundaryConditionSolver2/ConstrainVelocityWithFriction/data.png "Data")
        ![Marker](img/manual_tests/GridBlockedBoundaryConditionSolver2/ConstrainVelocityWithFriction/marker.png "Marker")

* GridFractionalBoundaryConditionSolver2Tests.
    * ConstrainVelocity

        Constrained velocity:

        ![Data](img/manual_tests/GridFractionalBoundaryConditionSolver2/ConstrainVelocity/data.png "Data")

* GridForwardEulerDiffusionSolver3Tests.
    * Solve

        Diffusion applied to the entire domain with small diffusion coefficient.

        ![Input](img/manual_tests/GridForwardEulerDiffusionSolver3/Solve/input.png "Input")
        ![Output](img/manual_tests/GridForwardEulerDiffusionSolver3/Solve/output.png "Output")

    * Unstable

        Diffusion applied to the entire domain with large diffusion coefficient.

        ![Input](img/manual_tests/GridForwardEulerDiffusionSolver3/Solve/input.png "Input")
        ![Output](img/manual_tests/GridForwardEulerDiffusionSolver3/Solve/output.png "Output")

* GridBackwardEulerDiffusionSolver2Tests.
    * Solve

        Diffusion applied to the half of the domain with very large diffusion coefficient.

        Input and output field:

        ![Input](img/manual_tests/GridBackwardEulerDiffusionSolver2/Solve/input.png "Input")
        ![Output](img/manual_tests/GridBackwardEulerDiffusionSolver2/Solve/output.png "Output")

* GridBackwardEulerDiffusionSolver3Tests.
    * Solve

        Diffusion applied to the entire domain with small diffusion coefficient.

        Input and output field:

        ![Input](img/manual_tests/GridBackwardEulerDiffusionSolver3/Solve/input.png "Input")
        ![Output](img/manual_tests/GridBackwardEulerDiffusionSolver3/Solve/output.png "Output")

    * Stable

        Diffusion applied to the entire domain with large diffusion coefficient.

        Input and output field:

        ![Input](img/manual_tests/GridBackwardEulerDiffusionSolver3/Solve/input.png "Input")
        ![Output](img/manual_tests/GridBackwardEulerDiffusionSolver3/Solve/output.png "Output")

    * SolveWithBoundaryDirichlet

        Diffusion applied to the half of the domain using Dirichlet boundary condition with very large diffusion coefficient.

        Input and output field:

        ![Input](img/manual_tests/GridBackwardEulerDiffusionSolver3/SolveWithBoundaryDirichlet/input.png "Input")
        ![Output](img/manual_tests/GridBackwardEulerDiffusionSolver3/SolveWithBoundaryDirichlet/output.png "Output")

    * SolveWithBoundaryNeumann

        Diffusion applied to the half of the domain using Neumann boundary condition with very large diffusion coefficient.

        Input and output field:

        ![Input](img/manual_tests/GridBackwardEulerDiffusionSolver3/SolveWithBoundaryNeumann/input.png "Input")
        ![Output](img/manual_tests/GridBackwardEulerDiffusionSolver3/SolveWithBoundaryNeumann/output.png "Output")

* GridFluidSolver2Tests.
    * ApplyBoundaryConditionWithPressure

        When right-facing velocity field is applied, solve the incompressible flow. In this test case, use closed boundary. Zero velocity field expected.

        Velocity (the arrows may be scaled even if their absolute magnitude is very small. See divergence for better analysis):

        ![Data](img/manual_tests/GridFluidSolver2/ApplyBoundaryConditionWithPressure/data.png "Data")

        Divergence:

        ![Div](img/manual_tests/GridFluidSolver2/ApplyBoundaryConditionWithPressure/div.png "Div")

        Pressure:

        ![Pressure](img/manual_tests/GridFluidSolver2/ApplyBoundaryConditionWithPressure/pressure.png "Pressure")

    * ApplyBoundaryConditionWithVariationalPressure

        When right-facing velocity field is applied, solve the incompressible flow. In this test case, use closed boundary. To solve boundary condition, the fractional (or often called variational) method is used. Div-free flow around the boundary is expected.

        Velocity (the arrows may be scaled even if their absolute magnitude is very small. See divergence for better analysis):

        ![Data](img/manual_tests/GridFluidSolver2/ApplyBoundaryConditionWithVariationalPressure/data.png "Data")

        Divergence:

        ![Div](img/manual_tests/GridFluidSolver2/ApplyBoundaryConditionWithVariationalPressure/div.png "Div")

        Pressure:

        ![Pressure](img/manual_tests/GridFluidSolver2/ApplyBoundaryConditionWithVariationalPressure/pressure.png "Pressure")

    * ApplyBoundaryConditionWithPressureOpen

        When right-facing velocity field is applied, solve the incompressible flow. In this test case, use open (left and right) boundary. Div-free flow around the boundary is expected.

        Velocity (the arrows may be scaled even if their absolute magnitude is very small. See divergence for better analysis):

        ![Data](img/manual_tests/GridFluidSolver2/ApplyBoundaryConditionWithPressureOpen/data.png "Data")

        Divergence:

        ![Div](img/manual_tests/GridFluidSolver2/ApplyBoundaryConditionWithPressureOpen/div.png "Div")

        Pressure:

        ![Pressure](img/manual_tests/GridFluidSolver2/ApplyBoundaryConditionWithPressureOpen/pressure.png "Pressure")

* GridSmokeSolver2Tests.
    * MovingEmitterWithCollider

        ![Data](img/manual_tests/GridSmokeSolver2/MovingEmitterWithCollider/data.gif "Data")

    * Rising

        ![Data](img/manual_tests/GridSmokeSolver2/Rising/data.gif "Data")

    * RisingWithCollider

        ![Data](img/manual_tests/GridSmokeSolver2/RisingWithCollider/data.gif "Data")

    * RisingWithColliderNonVariational

        ![Data](img/manual_tests/GridSmokeSolver2/RisingWithColliderNonVariational/data.gif "Data")

    * RisingWithColliderAndDiffusion

        ![Data](img/manual_tests/GridSmokeSolver2/RisingWithColliderAndDiffusion/data.gif "Data")

* GridSmokeSolver3Tests.
    * Rising

        ![Data](img/manual_tests/GridSmokeSolver3/Rising/data.gif "Data")

    * RisingWithCollider

        ![Data](img/manual_tests/GridSmokeSolver3/RisingWithCollider/data.gif "Data")

    * RisingWithColliderLinear

        ![Data](img/manual_tests/GridSmokeSolver3/RisingWithColliderLinear/data.gif "Data")

* HelloFluidSimTests.
    * Run

        ![Data](img/manual_tests/HelloFluidSim/Run/data.gif "Data")

* LevelSetSolver2Tests.
    * Reinitialize

        Input and output level set field when advection and reinitialization combined:

        ![Input](img/manual_tests/LevelSetSolver2/Reinitialize/input.png "Input")
        ![Output](img/manual_tests/LevelSetSolver2/Reinitialize/output.png "Output")

        Background flow field:

        ![Flow](img/manual_tests/LevelSetSolver2/Reinitialize/flow.png "Flow")

    * NoReinitialize

        Input and output level set field with advection but without reinitialization:

        ![Output](img/manual_tests/LevelSetSolver2/NoReinitialize/output.png "Output")

* UpwindLevelSetSolver2Tests.
    * ReinitializeSmall

        2-D iterative upwind-based level set solver test on low-resolution grid.

        Reinitializing constant field (should result the same constant field), before and after:

        ![ConstantBefore](img/manual_tests/UpwindLevelSetSolver2/ReinitializeSmall/constant0.png "ConstantBefore")
        ![ConstantAfter](img/manual_tests/UpwindLevelSetSolver2/ReinitializeSmall/constant1.png "ConstantAfter")

        Reinitializing SDF (should result nearly the same SDF), before and after:

        ![SDFBefore](img/manual_tests/UpwindLevelSetSolver2/ReinitializeSmall/sdf0.png "SDFBefore")
        ![SDFAfter](img/manual_tests/UpwindLevelSetSolver2/ReinitializeSmall/sdf1.png "SDFAfter")

        Reinitializing 2x scaled SDF (should result 1x scale SDF), before and after:

        ![ScaledBefore](img/manual_tests/UpwindLevelSetSolver2/ReinitializeSmall/scaled0.png "ScaledBefore")
        ![ScaledAfter](img/manual_tests/UpwindLevelSetSolver2/ReinitializeSmall/scaled1.png "ScaledAfter")

        Reinitializing unit step function (should result 1x scale SDF), before and after:

        ![UnitStepBefore](img/manual_tests/UpwindLevelSetSolver2/ReinitializeSmall/unit_step0.png "UnitStepBefore")
        ![UnitStepAfter](img/manual_tests/UpwindLevelSetSolver2/ReinitializeSmall/unit_step1.png "UnitStepAfter")

    * Reinitialize

        2-D iterative upwind-based level set solver test on high-resolution grid.

        Reinitializing constant field (should result the same constant field), before and after:

        ![ConstantBefore](img/manual_tests/UpwindLevelSetSolver2/Reinitialize/constant0.png "ConstantBefore")
        ![ConstantAfter](img/manual_tests/UpwindLevelSetSolver2/Reinitialize/constant1.png "ConstantAfter")

        Reinitializing SDF (should result nearly the same SDF), before and after:

        ![SDFBefore](img/manual_tests/UpwindLevelSetSolver2/Reinitialize/sdf0.png "SDFBefore")
        ![SDFAfter](img/manual_tests/UpwindLevelSetSolver2/Reinitialize/sdf1.png "SDFAfter")

        Reinitializing 2x scaled SDF (should result 1x scale SDF), before and after:

        ![ScaledBefore](img/manual_tests/UpwindLevelSetSolver2/Reinitialize/scaled0.png "ScaledBefore")
        ![ScaledAfter](img/manual_tests/UpwindLevelSetSolver2/Reinitialize/scaled1.png "ScaledAfter")

        Reinitializing unit step function (should result 1x scale SDF), before and after:

        ![UnitStepBefore](img/manual_tests/UpwindLevelSetSolver2/Reinitialize/unit_step0.png "UnitStepBefore")
        ![UnitStepAfter](img/manual_tests/UpwindLevelSetSolver2/Reinitialize/unit_step1.png "UnitStepAfter")

    * Extrapolate

        Input and output field:

        ![Input](img/manual_tests/UpwindLevelSetSolver2/Extrapolate/input.png "Input")
        ![Output](img/manual_tests/UpwindLevelSetSolver2/Extrapolate/output.png "Output")

        Background signed-distance field:

        ![SDF](img/manual_tests/UpwindLevelSetSolver2/Extrapolate/sdf.png "SDF")

* UpwindLevelSetSolver3Tests.
    * ReinitializeSmall

        Input and output field (cross-sectional view):

        ![Input](img/manual_tests/UpwindLevelSetSolver3/ReinitializeSmall/input.png "Input")
        ![Output](img/manual_tests/UpwindLevelSetSolver3/ReinitializeSmall/output.png "Output")

    * ExtrapolateSmall

        Input and output field (cross-sectional view):

        ![Input](img/manual_tests/UpwindLevelSetSolver3/ExtrapolateSmall/input.png "Input")
        ![Output](img/manual_tests/UpwindLevelSetSolver3/ExtrapolateSmall/output.png "Output")

* EnoLevelSetSolver2Tests.
    * ReinitializeSmall

        2-D iterative ENO-based level set solver test on low-resolution grid.

        Reinitializing constant field (should result the same constant field), before and after:

        ![ConstantBefore](img/manual_tests/EnoLevelSetSolver2/ReinitializeSmall/constant0.png "ConstantBefore")
        ![ConstantAfter](img/manual_tests/EnoLevelSetSolver2/ReinitializeSmall/constant1.png "ConstantAfter")

        Reinitializing SDF (should result nearly the same SDF), before and after:

        ![SDFBefore](img/manual_tests/EnoLevelSetSolver2/ReinitializeSmall/sdf0.png "SDFBefore")
        ![SDFAfter](img/manual_tests/EnoLevelSetSolver2/ReinitializeSmall/sdf1.png "SDFAfter")

        Reinitializing 2x scaled SDF (should result 1x scale SDF), before and after:

        ![ScaledBefore](img/manual_tests/EnoLevelSetSolver2/ReinitializeSmall/scaled0.png "ScaledBefore")
        ![ScaledAfter](img/manual_tests/EnoLevelSetSolver2/ReinitializeSmall/scaled1.png "ScaledAfter")

        Reinitializing unit step function (should result 1x scale SDF), before and after:

        ![UnitStepBefore](img/manual_tests/EnoLevelSetSolver2/ReinitializeSmall/unit_step0.png "UnitStepBefore")
        ![UnitStepAfter](img/manual_tests/EnoLevelSetSolver2/ReinitializeSmall/unit_step1.png "UnitStepAfter")

    * Reinitialize

        2-D iterative ENO-based level set solver test on high-resolution grid.

        Reinitializing constant field (should result the same constant field), before and after:

        ![ConstantBefore](img/manual_tests/EnoLevelSetSolver2/Reinitialize/constant0.png "ConstantBefore")
        ![ConstantAfter](img/manual_tests/EnoLevelSetSolver2/Reinitialize/constant1.png "ConstantAfter")

        Reinitializing SDF (should result nearly the same SDF), before and after:

        ![SDFBefore](img/manual_tests/EnoLevelSetSolver2/Reinitialize/sdf0.png "SDFBefore")
        ![SDFAfter](img/manual_tests/EnoLevelSetSolver2/Reinitialize/sdf1.png "SDFAfter")

        Reinitializing 2x scaled SDF (should result 1x scale SDF), before and after:

        ![ScaledBefore](img/manual_tests/EnoLevelSetSolver2/Reinitialize/scaled0.png "ScaledBefore")
        ![ScaledAfter](img/manual_tests/EnoLevelSetSolver2/Reinitialize/scaled1.png "ScaledAfter")

        Reinitializing unit step function (should result 1x scale SDF), before and after:

        ![UnitStepBefore](img/manual_tests/EnoLevelSetSolver2/Reinitialize/unit_step0.png "UnitStepBefore")
        ![UnitStepAfter](img/manual_tests/EnoLevelSetSolver2/Reinitialize/unit_step1.png "UnitStepAfter")

    * Extrapolate

        Input and output field:

        ![Input](img/manual_tests/EnoLevelSetSolver2/Extrapolate/input.png "Input")
        ![Output](img/manual_tests/EnoLevelSetSolver2/Extrapolate/output.png "Output")

        Background signed-distance field:

        ![SDF](img/manual_tests/EnoLevelSetSolver2/Extrapolate/sdf.png "SDF")

* EnoLevelSetSolver3Tests.
    * ReinitializeSmall

        Input and output field (cross-sectional view):

        ![Input](img/manual_tests/EnoLevelSetSolver3/ReinitializeSmall/input.png "Input")
        ![Output](img/manual_tests/EnoLevelSetSolver3/ReinitializeSmall/output.png "Output")

    * ExtrapolateSmall

        Input and output field (cross-sectional view):

        ![Input](img/manual_tests/EnoLevelSetSolver3/ExtrapolateSmall/input.png "Input")
        ![Output](img/manual_tests/EnoLevelSetSolver3/ExtrapolateSmall/output.png "Output")

* LevelSetLiquidSolver2Tests.
    * Drop

        ![Data](img/manual_tests/LevelSetLiquidSolver2/Drop/data.gif "Data")

    * DropHighRes

        ![Data](img/manual_tests/LevelSetLiquidSolver2/DropHighRes/data.gif "Data")

    * DropWithCollider

        ![Data](img/manual_tests/LevelSetLiquidSolver2/DropWithCollider/data.gif "Data")

    * DropVariational

        ![Data](img/manual_tests/LevelSetLiquidSolver2/DropVariational/data.gif "Data")

    * DropWithColliderVariational

        ![Data](img/manual_tests/LevelSetLiquidSolver2/DropWithColliderVariational/data.gif "Data")

    * ViscousDropVariational

        ![Data](img/manual_tests/LevelSetLiquidSolver2/ViscousDropVariational/data.gif "Data")

    * DropWithoutGlobalComp

        ![Data](img/manual_tests/LevelSetLiquidSolver2/DropWithoutGlobalComp/data.gif "Data")

    * DropWithGlobalComp

        ![Data](img/manual_tests/LevelSetLiquidSolver2/DropWithGlobalComp/data.gif "Data")

* LevelSetLiquidSolver3Tests.
    * SubtleSloshing

        The simulation should correctly generate sloshing animation even with subtle slope of the initial geometry.

        ![Data](img/manual_tests/LevelSetLiquidSolver3/SubtleSloshing/data.gif "Data")

* MarchingCubesTests.
    * SingleCube

        ![SingleCube](img/manual_tests/MarchingCubes/SingleCube/single_cube.png "SingleCube")

    * FourCubes

        ![FourCubes](img/manual_tests/MarchingCubes/FourCubes/four_cubes.png "FourCubes")

    * Sphere

        ![Sphere](img/manual_tests/MarchingCubes/Sphere/sphere.png "Sphere")

* ParticleSystemSolver2Tests.
    * Update

        ![Data](img/manual_tests/ParticleSystemSolver2/Update/data.gif "Data")

* ParticleSystemSolver3Tests.
    * PerfectBounce

        ![Data](img/manual_tests/ParticleSystemSolver3/PerfectBounce/data.gif "Data")

    * HalfBounce

        ![Data](img/manual_tests/ParticleSystemSolver3/HalfBounce/data.gif "Data")

    * HalfBounceWithFriction

        ![Data](img/manual_tests/ParticleSystemSolver3/HalfBounceWithFriction/data.gif "Data")

    * NoBounce

        ![Data](img/manual_tests/ParticleSystemSolver3/NoBounce/data.gif "Data")

    * Update

        ![Data](img/manual_tests/ParticleSystemSolver3/Update/data.gif "Data")

* PciSphSolver2Tests.
    * SteadyState

        ![Data](img/manual_tests/PciSphSolver2/SteadyState/data.gif "Data")

    * WaterDrop

        ![Data](img/manual_tests/PciSphSolver2/WaterDrop/data.gif "Data")

* PciSphSolver3Tests.
    * SteadyState

        ![Data](img/manual_tests/PciSphSolver3/SteadyState/data.gif "Data")

    * WaterDrop

        ![Data](img/manual_tests/PciSphSolver3/WaterDrop/data.gif "Data")

* PhysicsAnimationTests.
    * SimpleMassSpringAnimation

        ![Data](img/manual_tests/PhysicsAnimation/SimpleMassSpringAnimation/data.gif "Data")

* PicSolver2Tests.
    * Empty

    * SteadyState

        ![Data](img/manual_tests/PicSolver2/SteadyState/data.gif "Data")

    * DamBreaking

        ![Data](img/manual_tests/PicSolver2/DamBreaking/data.gif "Data")

    * DamBreakingWithCollider

        ![Data](img/manual_tests/PicSolver2/DamBreakingWithCollider/data.gif "Data")

* PicSolver3Tests.
    * DamBreakingWithCollider

        ![Data](img/manual_tests/PicSolver3/DamBreakingWithCollider/data.gif "Data")

    * WaterDrop

        ![Data](img/manual_tests/PicSolver3/WaterDrop/data.gif "Data")

* PointHashGridSearcher2Tests.
    * Build

        ![Data](img/manual_tests/PointHashGridSearcher2/Build/data.png "Data")

* PointHashGridSearcher3Tests.
    * Build

        ![Data](img/manual_tests/PointHashGridSearcher3/Build/data.png "Data")

* PointParallelHashGridSearcher2Tests.
    * Build

        ![Data](img/manual_tests/PointParallelHashGridSearcher2/Build/data.png "Data")

* PointParallelHashGridSearcher3Tests.
    * Build

        ![Data](img/manual_tests/PointParallelHashGridSearcher3/Build/data.png "Data")

* SphStdKernel3Tests.
    * Operator

        Kernel radius h = 1, 1.2, 1.5

        ![Kernel1](img/manual_tests/SphStdKernel3/Operator/kernel1.png "Kernel1")
        ![Kernel2](img/manual_tests/SphStdKernel3/Operator/kernel2.png "Kernel2")
        ![Kernel3](img/manual_tests/SphStdKernel3/Operator/kernel3.png "Kernel3")

* SphSpikyKernel3Tests.
    * Derivatives

        Standard kernel and its 1st and 2nd derivatives:

        ![Standard](img/manual_tests/SphSpikyKernel3/Derivatives/std.png "Standard")
        ![1st derivative](img/manual_tests/SphSpikyKernel3/Derivatives/std_1st.png "1st derivative")
        ![2nd derivative](img/manual_tests/SphSpikyKernel3/Derivatives/std_2nd.png "2nd derivative")

        Spiky kernel and its 1st and 2nd derivatives:

        ![Spiky](img/manual_tests/SphSpikyKernel3/Derivatives/spiky.png "Spiky")
        ![1st derivative](img/manual_tests/SphSpikyKernel3/Derivatives/spiky_1st.png "1st derivative")
        ![2nd derivative](img/manual_tests/SphSpikyKernel3/Derivatives/spiky_2nd.png "2nd derivative")

* SphSolver2Tests.
    * SteadyState

        ![Data](img/manual_tests/SphSolver2/SteadyState/data.gif "Data")

    * WaterDrop

        ![Data](img/manual_tests/SphSolver2/WaterDrop/data.gif "Data")

    * WaterDropLargeDt

        Should give unstable result:

        ![Data](img/manual_tests/SphSolver2/WaterDropLargeDt/data.gif "Data")

* SphSolver3Tests.
    * SteadyState

        ![Data](img/manual_tests/SphSolver3/SteadyState/data.gif "Data")

    * WaterDrop

        ![Data](img/manual_tests/SphSolver3/WaterDrop/data.gif "Data")

* SphSystemData2Tests.
    * Interpolate

        ![Data](img/manual_tests/SphSystemData2/Interpolate/data.png "Data")

    * Gradient

        ![Data](img/manual_tests/SphSystemData2/Gradient/data.png "Data")
        ![Gradient](img/manual_tests/SphSystemData2/Gradient/gradient.png "Gradient")

    * Laplacian

        ![Data](img/manual_tests/SphSystemData2/Laplacian/data.png "Data")
        ![Laplacian](img/manual_tests/SphSystemData2/Laplacian/laplacian.png "Laplacian")

* SphSystemData3Tests.
    * Interpolate

        ![Data](img/manual_tests/SphSystemData3/Interpolate/data.png "Data")

    * Gradient

        ![Data](img/manual_tests/SphSystemData3/Gradient/data.png "Data")
        ![Gradient](img/manual_tests/SphSystemData3/Gradient/gradient.png "Gradient")

    * Laplacian

        ![Data](img/manual_tests/SphSystemData3/Laplacian/data.png "Data")
        ![Laplacian](img/manual_tests/SphSystemData3/Laplacian/laplacian.png "Laplacian")

* TriangleMesh3Tests.
    * PointsOnlyGeometries
    * PointsAndNormalGeometries
    * BasicIO

* TriangleMeshToSdfTests.
    * Cube

        ![SDF](img/manual_tests/TriangleMeshToSdf/Cube/sdf.png "SDF")

    * Bunny
    * Dragon

* VolumeParticleEmitter2Tests.
    * EmitContinuousNonOverlapping

        ![Data](img/manual_tests/VolumeParticleEmitter2/EmitContinuousNonOverlapping/data.gif "Data")

* VolumeParticleEmitter3Tests.
    * EmitContinuousNonOverlapping

        ![Data](img/manual_tests/VolumeParticleEmitter3/EmitContinuousNonOverlapping/data.gif "Data")

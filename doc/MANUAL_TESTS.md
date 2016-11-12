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

* SemiLagrangian2Tests.
    * Boundary
    * Zalesak

* CubicSemiLagrangian2Tests.
    * Zalesak

* AnimationTests.
    * OnUpdateSine
    * OnUpdateSineWithDecay

* ArrayUtilsTests.
    * ExtralateToRegion2

* ScalarField3Tests.
    * Sample
    * Gradient
    * Laplacian

* VectorField3Tests.
    * Sample
    * Divergence
    * Curl
    * Sample2

* FlipSolver2Tests.
    * Empty
    * SteadyState
    * DamBreaking
    * DamBreakingWithCollider

* FlipSolver3Tests.
    * WaterDrop
    * DamBreakingWithCollider

* FmmLevelSetSolver2Tests.
    * ReinitializeSmall
    * Reinitialize
    * Extrapolate

* FmmLevelSetSolver3Tests.
    * ReinitializeSmall
    * ExtrapolateSmall

* GridBlockedBoundaryConditionSolver2Tests.
    * ConstrainVelocitySmall
    * ConstrainVelocity
    * ConstrainVelocityWithFriction

* GridFractionalBoundaryConditionSolver2Tests.
    * ConstrainVelocity

* GridForwardEulerDiffusionSolver3Tests.
    * Solve
    * Unstable

* GridBackwardEulerDiffusionSolver2Tests.
    * Solve

* GridBackwardEulerDiffusionSolver3Tests.
    * Solve
    * Stable
    * SolveWithBoundaryDirichlet
    * SolveWithBoundaryNeumann

* GridFluidSolver2Tests.
    * ApplyBoundaryConditionWithPressure
    * ApplyBoundaryConditionWithVariationalPressure
    * ApplyBoundaryConditionWithPressureOpen

* GridSmokeSolver2Tests.
    * Rising
    * RisingWithCollider
    * RisingWithColliderVariational
    * RisingWithColliderAndDiffusion

* GridSmokeSolver3Tests.
    * Rising
    * RisingWithCollider

* HelloFluidSimTests.
    * Run

* LevelSetSolver2Tests.
    * Reinitialize
    * NoReinitialize

* UpwindLevelSetSolver2Tests.
    * ReinitializeSmall
    * Reinitialize
    * Extrapolate

* UpwindLevelSetSolver3Tests.
    * ReinitializeSmall
    * ExtrapolateSmall

* EnoLevelSetSolver2Tests.
    * ReinitializeSmall
    * Reinitialize
    * Extrapolate

* EnoLevelSetSolver3Tests.
    * ReinitializeSmall
    * ExtrapolateSmall

* LevelSetLiquidSolver2Tests.
    * Drop
    * DropHighRes
    * DropWithCollider
    * DropVariational
    * DropWithColliderVariational
    * ViscousDropVariational
    * DropWithoutGlobalComp
    * DropWithGlobalComp

* MarchingCubesTests.
    * SingleCube
    * FourCubes
    * Sphere

* ParticleSystemSolver2Tests.
    * Update

* ParticleSystemSolver3Tests.
    * PerfectBounce
    * HalfBounce
    * HalfBounceWithFriction
    * NoBounce
    * Update

* PciSphSolver2Tests.
    * SteadyState
    * WaterDrop

* PciSphSolver3Tests.
    * SteadyState
    * WaterDrop

* PhysicsAnimationTests.
    * SimpleMassSpringAnimation

* PicSolver2Tests.
    * Empty
    * SteadyState
    * DamBreaking
    * DamBreakingWithCollider

* PicSolver3Tests.
    * DamBreakingWithCollider

* PointHashGridSearcher2Tests.
    * Build

* PointHashGridSearcher3Tests.
    * Build

* PointParallelHashGridSearcher2Tests.
    * Build

* PointParallelHashGridSearcher3Tests.
    * Build

* SphStdKernel3Tests.
    * Operator

* SphSpikyKernel3Tests.
    * Derivatives

* SphSolver2Tests.
    * SteadyState
    * WaterDrop
    * WaterDropLargeDt

* SphSolver3Tests.
    * SteadyState
    * WaterDrop

* SphSystemData2Tests.
    * Interpolate
    * Gradient
    * Laplacian

* SphSystemData3Tests.
    * Interpolate
    * Gradient
    * Laplacian

* TriangleMesh3Tests.
    * PointsOnlyGeometries
    * PointsAndNormalGeometries
    * BasicIO

* TriangleMeshToSdfTests.
    * Cube
    * Bunny
    * Dragon

* VolumeParticleEmitter2Tests.
    * EmitContinuousNonOverlapping

* VolumeParticleEmitter3Tests.
    * EmitContinuousNonOverlapping

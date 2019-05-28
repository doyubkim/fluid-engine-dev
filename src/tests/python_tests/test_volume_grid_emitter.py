"""
Copyright (c) 2019 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import numpy as np
import pyjet
from pytest_utils import *


def test_volume_grid_emitter2():
    # Basic ctor test
    sphere = pyjet.Sphere2(center=(0.5, 0.5), radius=0.15)
    emitter = pyjet.VolumeGridEmitter2(sphere, False)

    assert emitter.sourceRegion
    assert not emitter.isOneShot
    assert emitter.isEnabled

    # Another basic ctor test
    emitter2 = pyjet.VolumeGridEmitter2(sourceRegion=sphere, isOneShot=False)

    assert emitter2.sourceRegion
    assert not emitter2.isOneShot
    assert emitter2.isEnabled

    # One-shot emitter
    emitter3 = pyjet.VolumeGridEmitter2(sourceRegion=sphere, isOneShot=True)

    assert emitter3.isOneShot

    frame = pyjet.Frame()
    solver = pyjet.GridSmokeSolver2(resolution=(32, 32), domainSizeX=1.0)
    solver.emitter = emitter3
    emitter3.addStepFunctionTarget(solver.smokeDensity, 0.0, 1.0)
    emitter3.addStepFunctionTarget(solver.temperature, 0.0, 1.0)

    # Emit some smoke
    old_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    solver.update(frame)
    frame.advance()
    new_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    diff = np.linalg.norm(old_den - new_den)
    assert diff > 0.0
    assert not emitter3.isEnabled

    # Should not emit more smoke
    old_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    emitter3.update(0, 0)
    new_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    diff = np.linalg.norm(old_den - new_den)
    assert diff < 1e-20

    # Re-enabling the emitter should make it emit one more time
    emitter3.isEnabled = True
    old_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    solver.update(frame)
    frame.advance()
    new_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    diff = np.linalg.norm(old_den - new_den)
    assert diff > 0.0
    assert not emitter3.isEnabled

    # ...and gets disabled again
    old_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    emitter3.update(0, 0)
    new_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    diff = np.linalg.norm(old_den - new_den)
    assert diff < 1e-20


def test_volume_grid_emitter3():
    # Basic ctor test
    sphere = pyjet.Sphere3(center=(0.5, 0.5, 0.5), radius=0.15)
    emitter = pyjet.VolumeGridEmitter3(sphere, False)

    assert emitter.sourceRegion
    assert not emitter.isOneShot
    assert emitter.isEnabled

    # Another basic ctor test
    emitter2 = pyjet.VolumeGridEmitter3(sourceRegion=sphere, isOneShot=False)

    assert emitter2.sourceRegion
    assert not emitter2.isOneShot
    assert emitter2.isEnabled

    # One-shot emitter
    emitter3 = pyjet.VolumeGridEmitter3(sourceRegion=sphere, isOneShot=True)

    assert emitter3.isOneShot

    frame = pyjet.Frame()
    solver = pyjet.GridSmokeSolver3(resolution=(32, 32, 32), domainSizeX=1.0)
    solver.emitter = emitter3
    emitter3.addStepFunctionTarget(solver.smokeDensity, 0.0, 1.0)
    emitter3.addStepFunctionTarget(solver.temperature, 0.0, 1.0)

    # Emit some smoke
    old_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    solver.update(frame)
    frame.advance()
    new_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    diff = np.linalg.norm(old_den - new_den)
    assert diff > 0.0
    assert not emitter3.isEnabled

    # Should not emit more smoke
    old_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    emitter3.update(0, 0)
    new_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    diff = np.linalg.norm(old_den - new_den)
    assert diff < 1e-20

    # Re-enabling the emitter should make it emit one more time
    emitter3.isEnabled = True
    old_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    solver.update(frame)
    frame.advance()
    new_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    diff = np.linalg.norm(old_den - new_den)
    assert diff > 0.0
    assert not emitter3.isEnabled

    # ...and gets disabled again
    old_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    emitter3.update(0, 0)
    new_den = np.array(solver.smokeDensity.dataAccessor(), copy=True)
    diff = np.linalg.norm(old_den - new_den)
    assert diff < 1e-20

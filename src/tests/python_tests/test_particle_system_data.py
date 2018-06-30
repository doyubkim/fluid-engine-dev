"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import numpy as np
import pyjet


def test_init2():
    ps = pyjet.ParticleSystemData2()
    assert ps.numberOfParticles == 0

    ps2 = pyjet.ParticleSystemData2(100)
    assert ps2.numberOfParticles == 100


def test_resize2():
    ps = pyjet.ParticleSystemData2()
    ps.resize(12)
    assert ps.numberOfParticles == 12


def test_add_scalar_data2():
    ps = pyjet.ParticleSystemData2()
    ps.resize(12)

    a0 = ps.addScalarData(2.0)
    a1 = ps.addScalarData(9.0)
    assert ps.numberOfParticles == 12
    assert a0 == 0
    assert a1 == 1

    as0 = np.array(ps.scalarDataAt(a0))
    for val in as0:
        assert val == 2.0

    as1 = np.array(ps.scalarDataAt(a1))
    for val in as1:
        assert val == 9.0


def test_add_vector_data2():
    ps = pyjet.ParticleSystemData2()
    ps.resize(12)

    a0 = ps.addVectorData((2.0, 4.0))
    a1 = ps.addVectorData((9.0, -2.0))
    assert ps.numberOfParticles == 12
    assert a0 == 3
    assert a1 == 4

    as0 = np.array(ps.vectorDataAt(a0))
    for val in as0:
        assert val.tolist() == [2.0, 4.0]

    as1 = np.array(ps.vectorDataAt(a1))
    for val in as1:
        assert val.tolist() == [9.0, -2.0]


def test_add_particles2():
    ps = pyjet.ParticleSystemData2()
    ps.resize(12)

    ps.addParticles([(1.0, 2.0), (4.0, 5.0)],
                    [(7.0, 8.0), (8.0, 7.0)],
                    [(5.0, 4.0), (2.0, 1.0)])

    assert ps.numberOfParticles == 14
    p = np.array(ps.positions)
    v = np.array(ps.velocities)
    f = np.array(ps.forces)

    assert [1.0, 2.0] == p[12].tolist()
    assert [4.0, 5.0] == p[13].tolist()
    assert [7.0, 8.0] == v[12].tolist()
    assert [8.0, 7.0] == v[13].tolist()
    assert [5.0, 4.0] == f[12].tolist()
    assert [2.0, 1.0] == f[13].tolist()


# ------------------------------------------------------------------------------

def test_init3():
    ps = pyjet.ParticleSystemData3()
    assert ps.numberOfParticles == 0

    ps2 = pyjet.ParticleSystemData3(100)
    assert ps2.numberOfParticles == 100


def test_resize3():
    ps = pyjet.ParticleSystemData3()
    ps.resize(12)
    assert ps.numberOfParticles == 12


def test_add_scalar_data3():
    ps = pyjet.ParticleSystemData3()
    ps.resize(12)

    a0 = ps.addScalarData(2.0)
    a1 = ps.addScalarData(9.0)
    assert ps.numberOfParticles == 12
    assert a0 == 0
    assert a1 == 1

    as0 = np.array(ps.scalarDataAt(a0))
    for val in as0:
        assert val == 2.0

    as1 = np.array(ps.scalarDataAt(a1))
    for val in as1:
        assert val == 9.0


def test_add_vector_data3():
    ps = pyjet.ParticleSystemData3()
    ps.resize(12)

    a0 = ps.addVectorData((2.0, 4.0, -1.0))
    a1 = ps.addVectorData((9.0, -2.0, 5.0))
    assert ps.numberOfParticles == 12
    assert a0 == 3
    assert a1 == 4

    as0 = np.array(ps.vectorDataAt(a0))
    for val in as0:
        assert val.tolist() == [2.0, 4.0, -1.0]

    as1 = np.array(ps.vectorDataAt(a1))
    for val in as1:
        assert val.tolist() == [9.0, -2.0, 5.0]


def test_add_particles3():
    ps = pyjet.ParticleSystemData3()
    ps.resize(12)

    ps.addParticles([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
                    [(7.0, 8.0, 9.0), (8.0, 7.0, 6.0)],
                    [(5.0, 4.0, 3.0), (2.0, 1.0, 3.0)])

    assert ps.numberOfParticles == 14
    p = np.array(ps.positions)
    v = np.array(ps.velocities)
    f = np.array(ps.forces)

    assert [1.0, 2.0, 3.0] == p[12].tolist()
    assert [4.0, 5.0, 6.0] == p[13].tolist()
    assert [7.0, 8.0, 9.0] == v[12].tolist()
    assert [8.0, 7.0, 6.0] == v[13].tolist()
    assert [5.0, 4.0, 3.0] == f[12].tolist()
    assert [2.0, 1.0, 3.0] == f[13].tolist()

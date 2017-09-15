"""
Copyright (c) 2017 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
import unittest
import numpy as np


class ParticleSystemData2Tests(unittest.TestCase):
    def testInit(self):
        ps = pyjet.ParticleSystemData2()
        self.assertEqual(ps.numberOfParticles, 0)

        ps2 = pyjet.ParticleSystemData2(100)
        self.assertEqual(ps2.numberOfParticles, 100)

    def testResize(self):
        ps = pyjet.ParticleSystemData2()
        ps.resize(12)
        self.assertEqual(ps.numberOfParticles, 12)

    def testAddScalarData(self):
        ps = pyjet.ParticleSystemData2()
        ps.resize(12)

        a0 = ps.addScalarData(2.0)
        a1 = ps.addScalarData(9.0)
        self.assertEqual(ps.numberOfParticles, 12)
        self.assertEqual(a0, 0)
        self.assertEqual(a1, 1)

        as0 = np.array(ps.scalarDataAt(a0))
        for val in as0:
            self.assertEqual(val, 2.0)

        as1 = np.array(ps.scalarDataAt(a1))
        for val in as1:
            self.assertEqual(val, 9.0)

    def testAddVectorData(self):
        ps = pyjet.ParticleSystemData2()
        ps.resize(12)

        a0 = ps.addVectorData((2.0, 4.0))
        a1 = ps.addVectorData((9.0, -2.0))
        self.assertEqual(ps.numberOfParticles, 12)
        self.assertEqual(a0, 3)
        self.assertEqual(a1, 4)

        as0 = np.array(ps.vectorDataAt(a0))
        for val in as0:
            self.assertEqual(val.tolist(), [2.0, 4.0])

        as1 = np.array(ps.vectorDataAt(a1))
        for val in as1:
            self.assertEqual(val.tolist(), [9.0, -2.0])

    def testAddParticles(self):
        ps = pyjet.ParticleSystemData2()
        ps.resize(12)

        ps.addParticles([(1.0, 2.0), (4.0, 5.0)],
                        [(7.0, 8.0), (8.0, 7.0)],
                        [(5.0, 4.0), (2.0, 1.0)])

        self.assertEqual(ps.numberOfParticles, 14)
        p = np.array(ps.positions)
        v = np.array(ps.velocities)
        f = np.array(ps.forces)

        self.assertEqual([1.0, 2.0], p[12].tolist())
        self.assertEqual([4.0, 5.0], p[13].tolist())
        self.assertEqual([7.0, 8.0], v[12].tolist())
        self.assertEqual([8.0, 7.0], v[13].tolist())
        self.assertEqual([5.0, 4.0], f[12].tolist())
        self.assertEqual([2.0, 1.0], f[13].tolist())



class ParticleSystemData3Tests(unittest.TestCase):
    def testInit(self):
        ps = pyjet.ParticleSystemData3()
        self.assertEqual(ps.numberOfParticles, 0)

        ps2 = pyjet.ParticleSystemData3(100)
        self.assertEqual(ps2.numberOfParticles, 100)

    def testResize(self):
        ps = pyjet.ParticleSystemData3()
        ps.resize(12)
        self.assertEqual(ps.numberOfParticles, 12)

    def testAddScalarData(self):
        ps = pyjet.ParticleSystemData3()
        ps.resize(12)

        a0 = ps.addScalarData(2.0)
        a1 = ps.addScalarData(9.0)
        self.assertEqual(ps.numberOfParticles, 12)
        self.assertEqual(a0, 0)
        self.assertEqual(a1, 1)

        as0 = np.array(ps.scalarDataAt(a0))
        for val in as0:
            self.assertEqual(val, 2.0)

        as1 = np.array(ps.scalarDataAt(a1))
        for val in as1:
            self.assertEqual(val, 9.0)

    def testAddVectorData(self):
        ps = pyjet.ParticleSystemData3()
        ps.resize(12)

        a0 = ps.addVectorData((2.0, 4.0, -1.0))
        a1 = ps.addVectorData((9.0, -2.0, 5.0))
        self.assertEqual(ps.numberOfParticles, 12)
        self.assertEqual(a0, 3)
        self.assertEqual(a1, 4)

        as0 = np.array(ps.vectorDataAt(a0))
        for val in as0:
            self.assertEqual(val.tolist(), [2.0, 4.0, -1.0])

        as1 = np.array(ps.vectorDataAt(a1))
        for val in as1:
            self.assertEqual(val.tolist(), [9.0, -2.0, 5.0])

    def testAddParticles(self):
        ps = pyjet.ParticleSystemData3()
        ps.resize(12)

        ps.addParticles([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
                        [(7.0, 8.0, 9.0), (8.0, 7.0, 6.0)],
                        [(5.0, 4.0, 3.0), (2.0, 1.0, 3.0)])

        self.assertEqual(ps.numberOfParticles, 14)
        p = np.array(ps.positions)
        v = np.array(ps.velocities)
        f = np.array(ps.forces)

        self.assertEqual([1.0, 2.0, 3.0], p[12].tolist())
        self.assertEqual([4.0, 5.0, 6.0], p[13].tolist())
        self.assertEqual([7.0, 8.0, 9.0], v[12].tolist())
        self.assertEqual([8.0, 7.0, 6.0], v[13].tolist())
        self.assertEqual([5.0, 4.0, 3.0], f[12].tolist())
        self.assertEqual([2.0, 1.0, 3.0], f[13].tolist())


def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()

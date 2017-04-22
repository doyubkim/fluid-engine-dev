import pyjet
import unittest

class BoundingBox3DTests(unittest.TestCase):
    def testInit(self):
        a = pyjet.BoundingBox3D()
        self.assertGreater(a.lowerCorner.x, a.upperCorner.x)
        self.assertGreater(a.lowerCorner.y, a.upperCorner.y)
        self.assertGreater(a.lowerCorner.z, a.upperCorner.z)
        b = pyjet.BoundingBox3D((-1, -2, -3), (4, 2, 5))
        self.assertEqual(b.lowerCorner.x, -1.0)
        self.assertEqual(b.lowerCorner.y, -2.0)
        self.assertEqual(b.lowerCorner.z, -3.0)
        self.assertEqual(b.upperCorner.x, 4.0)
        self.assertEqual(b.upperCorner.y, 2.0)
        self.assertEqual(b.upperCorner.z, 5.0)
        l, c = pyjet.Vector3D(-1, -2, -3), pyjet.Vector3D(4, 2, 5)
        c = pyjet.BoundingBox3D(l, c)
        self.assertEqual(c.lowerCorner.x, -1.0)
        self.assertEqual(c.lowerCorner.y, -2.0)
        self.assertEqual(c.lowerCorner.z, -3.0)
        self.assertEqual(c.upperCorner.x, 4.0)
        self.assertEqual(c.upperCorner.y, 2.0)
        self.assertEqual(c.upperCorner.z, 5.0)

class FlipSolver3Tests(unittest.TestCase):
    def testInit(self):
        a = pyjet.FlipSolver3()
        self.assertEqual(a.resolution, (1, 1, 1))
        self.assertEqual(a.gridSpacing, (1.0, 1.0, 1.0))
        self.assertEqual(a.gridOrigin, (0.0, 0.0, 0.0))

        b = pyjet.FlipSolver3((2, 3, 4), (5, 6, 7), (8, 9, 10))
        self.assertEqual(b.resolution, (2, 3, 4))
        self.assertEqual(b.gridSpacing, (5.0, 6.0, 7.0))
        self.assertEqual(b.gridOrigin, (8.0, 9.0, 10.0))

        c = pyjet.FlipSolver3(resolution=(2, 3, 4), gridSpacing=(5, 6, 7), gridOrigin=(8, 9, 10))
        self.assertEqual(c.resolution, (2, 3, 4))
        self.assertEqual(c.gridSpacing, (5.0, 6.0, 7.0))
        self.assertEqual(c.gridOrigin, (8.0, 9.0, 10.0))

    def testAnimation(self):
        a = pyjet.FlipSolver3()
        f = pyjet.Frame()
        a.update(f)
        self.assertEqual(a.currentFrame.index, 0)
        f.advance()
        a.update(f)
        self.assertEqual(a.currentFrame.index, 1)

    def testPhysicsAnimation(self):
        a = pyjet.FlipSolver3()
        a.isUsingFixedSubTimeSteps = False
        self.assertFalse(a.isUsingFixedSubTimeSteps)
        a.isUsingFixedSubTimeSteps = True
        self.assertTrue(a.isUsingFixedSubTimeSteps)
        a.numberOfFixedSubTimeSteps = 42
        self.assertEqual(a.numberOfFixedSubTimeSteps, 42)
        a.advanceSingleFrame()
        self.assertEqual(a.currentFrame.index, 0)
        self.assertAlmostEqual(a.currentTimeInSeconds, 0.0, delta=1e-12)
        a.advanceSingleFrame()
        self.assertEqual(a.currentFrame.index, 1)
        self.assertAlmostEqual(a.currentTimeInSeconds, 1.0 / 60.0, delta=1e-12)

    def testGridFluidSolver3(self):
        a = pyjet.FlipSolver3()
        a.gravity = (1.0, 2.0, 3.0)
        self.assertEqual(a.gravity, (1.0, 2.0, 3.0))
        a.viscosityCoefficient = 0.042
        self.assertEqual(a.viscosityCoefficient, 0.042)
        a.maxCfl = 3.0
        self.assertEqual(a.maxCfl, 3.0)
        a.resizeGrid((2, 3, 4), (5, 6, 7), (8, 9, 10))
        self.assertEqual(a.resolution, (2, 3, 4))
        self.assertEqual(a.gridSpacing, (5.0, 6.0, 7.0))
        self.assertEqual(a.gridOrigin, (8.0, 9.0, 10.0))

    def testFlipSolver3(self):
        a = pyjet.FlipSolver3()
        a.picBlendingFactor = 0.7
        self.assertEqual(a.picBlendingFactor, 0.7)

class QuaternionTest(unittest.TestCase):
    def testInit(self):
        a = pyjet.QuaternionD()
        self.assertEqual(a.w, 1.0)
        self.assertEqual(a.x, 0.0)
        self.assertEqual(a.y, 0.0)
        self.assertEqual(a.z, 0.0)
        b = pyjet.QuaternionD(-1, 2, 3, 4)
        self.assertEqual(b.w, -1.0)
        self.assertEqual(b.x, 2.0)
        self.assertEqual(b.y, 3.0)
        self.assertEqual(b.z, 4.0)
        c = pyjet.QuaternionD(x=2, z=4, w=1, y=3)
        self.assertEqual(c.w, 1.0)
        self.assertEqual(c.x, 2.0)
        self.assertEqual(c.y, 3.0)
        self.assertEqual(c.z, 4.0)
        d = pyjet.QuaternionD(c)
        self.assertEqual(d.w, 1.0)
        self.assertEqual(d.x, 2.0)
        self.assertEqual(d.y, 3.0)
        self.assertEqual(d.z, 4.0)

class Sphere3Test(unittest.TestCase):
    def testInit(self):
        a = pyjet.Sphere3()
        self.assertEqual(a.isNormalFlipped, False)
        self.assertEqual(a.center, (0, 0, 0))
        self.assertEqual(a.radius, 1.0)


class Vector3DTest(unittest.TestCase):
    def testInit(self):
        a = pyjet.Vector3D()
        self.assertEqual(a.x, 0.0)
        self.assertEqual(a.y, 0.0)
        self.assertEqual(a.z, 0.0)
        b = pyjet.Vector3D(1, 2, 3)
        self.assertEqual(b.x, 1.0)
        self.assertEqual(b.y, 2.0)
        self.assertEqual(b.z, 3.0)
        c = pyjet.Vector3D(y=2, x=1, z=3)
        self.assertEqual(c.x, 1.0)
        self.assertEqual(c.y, 2.0)
        self.assertEqual(c.z, 3.0)
        d = pyjet.Vector3D(c)
        self.assertEqual(d.x, 1.0)
        self.assertEqual(d.y, 2.0)
        self.assertEqual(d.z, 3.0)

def main():
    pyjet.Logging.mute()
    unittest.main()

if __name__ == '__main__':
    main()

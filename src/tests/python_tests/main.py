import pyjet
import unittest

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

def main():
    pyjet.Logging.mute()
    unittest.main()

if __name__ == '__main__':
    main()

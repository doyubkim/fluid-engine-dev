"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
import unittest


class FaceCenteredGrid2Tests(unittest.TestCase):
    def testFill(self):
        a = pyjet.FaceCenteredGrid2((10, 10))
        a.fill((3.0, 4.0))
        for j in range(10):
            for i in range(11):
                self.assertEqual(a.u(i, j), 3.0)
        for j in range(11):
            for i in range(10):
                self.assertEqual(a.v(i, j), 4.0)

        def filler(pt):
            return (pt.x, pt.y)

        a.fill(filler)
        for j in range(10):
            for i in range(11):
                self.assertEqual(a.u(i, j), i)
        for j in range(11):
            for i in range(10):
                self.assertEqual(a.v(i, j), j)

    def testForEach(self):
        a = pyjet.FaceCenteredGrid2((10, 10))
        # Workaround for Python 2.x which doesn't support nonlocal
        d = {'ei': 0, 'ej': 0}

        def checkU(i, j):
            self.assertEqual(i, d['ei'])
            self.assertEqual(j, d['ej'])
            d['ei'] += 1
            if d['ei'] >= 11:
                d['ei'] = 0
                d['ej'] += 1

        a.forEachUIndex(checkU)
        d = {'ei': 0, 'ej': 0}

        def checkV(i, j):
            self.assertEqual(i, d['ei'])
            self.assertEqual(j, d['ej'])
            d['ei'] += 1
            if d['ei'] >= 10:
                d['ei'] = 0
                d['ej'] += 1

        a.forEachVIndex(checkV)

    def testSerialization(self):
        a = pyjet.FaceCenteredGrid2((10, 10))

        def filler(pt):
            return (pt.x, pt.y)

        a.fill(filler)

        flatBuffer = a.serialize()

        b = pyjet.FaceCenteredGrid2()
        b.deserialize(flatBuffer)

        for j in range(10):
            for i in range(11):
                self.assertEqual(b.u(i, j), i)
        for j in range(11):
            for i in range(10):
                self.assertEqual(b.v(i, j), j)

def main():
    pyjet.Logging.mute()
    unittest.main()


if __name__ == '__main__':
    main()

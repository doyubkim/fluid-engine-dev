"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet


class MyAnimation(pyjet.Animation):
    def __init__(self):
        self.lastFrame = None
        super(MyAnimation, self).__init__()

    def onUpdate(self, frame):
        self.lastFrame = frame


def test_inheritance():
    anim = MyAnimation()
    f = pyjet.Frame(3, 0.02)
    anim.update(f)
    assert anim.lastFrame.index == 3
    assert anim.lastFrame.timeIntervalInSeconds == 0.02

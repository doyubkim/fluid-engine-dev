"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet


def create_collider2():
    sphere = pyjet.Sphere2()
    collider = pyjet.RigidBodyCollider2(surface=sphere)
    return collider


def create_collider3():
    sphere = pyjet.Sphere3()
    collider = pyjet.RigidBodyCollider3(surface=sphere)
    return collider


def test_collider_set2():
    collider_set = pyjet.ColliderSet2()
    assert collider_set.numberOfColliders == 0

    collider1 = create_collider2()
    collider2 = create_collider2()
    collider3 = create_collider2()
    collider_set.addCollider(collider1)
    collider_set.addCollider(collider2)
    collider_set.addCollider(collider3)
    assert collider_set.numberOfColliders == 3

    assert collider1 == collider_set.collider(0)
    assert collider2 == collider_set.collider(1)
    assert collider3 == collider_set.collider(2)


def test_collider_set3():
    collider_set = pyjet.ColliderSet3()
    assert collider_set.numberOfColliders == 0

    collider1 = create_collider3()
    collider2 = create_collider3()
    collider3 = create_collider3()
    collider_set.addCollider(collider1)
    collider_set.addCollider(collider2)
    collider_set.addCollider(collider3)
    assert collider_set.numberOfColliders == 3

    assert collider1 == collider_set.collider(0)
    assert collider2 == collider_set.collider(1)
    assert collider3 == collider_set.collider(2)

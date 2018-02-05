"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import sys

def render_still_trimesh(x, y, z, tri, output_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    verts = [zip(x, y, z)]

    min_x = min(x)
    min_y = min(y)
    min_z = min(z)
    max_x = max(x)
    max_y = max(y)
    max_z = max(z)
    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)
    mid_z = 0.5 * (min_z + max_z)
    width_x = max_x - min_x
    width_y = max_y - min_y
    width_z = max_z - min_z
    max_len = max([width_x, width_y, width_z])
    half_max_len = 0.5 * max_len

    ax.set_xlim(mid_x - half_max_len, mid_x + half_max_len)
    ax.set_ylim(mid_y - half_max_len, mid_y + half_max_len)
    ax.set_zlim(mid_z - half_max_len, mid_z + half_max_len)

    ax.plot_trisurf(x, y, z, triangles=tri, lw=0, color='grey', alpha=1.0)
    ax.set_aspect('equal')
    plt.savefig(output_filename)
    plt.close(fig)
    print ('Rendered <%s>' % output_filename)

def obj_to_xyzt(obj_filename):
    vert_x = []
    vert_y = []
    vert_z = []
    faces = []
    with open(obj_filename, 'r') as obj_file:
        lines = obj_file.readlines()
        for line in lines:
            tokens = line.split()
            if tokens[0] == 'v':
                # Swap y and z
                vert_x.append(float(tokens[1]))
                vert_y.append(float(tokens[3]))
                vert_z.append(float(tokens[2]))
            elif tokens[0] == 'f':
                face_x = int(tokens[1].split('/')[0]) - 1
                face_y = int(tokens[2].split('/')[0]) - 1
                face_z = int(tokens[3].split('/')[0]) - 1
                faces.append((face_x, face_y, face_z))
        return (vert_x, vert_y, vert_z, faces)
    return ([], [], [])


if __name__ == '__main__':
    x, y, z, tri = obj_to_xyzt(sys.argv[1])
    render_still_trimesh(x, y, z, tri, sys.argv[2])

# Usage: python visualize_xyz_mayavi.py [INPUT DIR] [TMP OUTPUT DIR]
# Required package: mayavi, install with pip install mayavi
# the png files can be combined with ImageMagick, such as
# convert -delay 30 /tmp/simsph/*.png /tmp/simsph.gif
import os
import sys
import numpy as np
import glob
from mayavi import mlab

mlab.options.offscreen = True
CUBE_MAX = 3.0 # maximum size of each side of the cube
outpath = sys.argv[2]
if (os.path.exists(outpath) == False):
    os.mkdir(outpath)

path = sys.argv[1]
files = sorted(glob.glob(path + "/*"))

fig = mlab.figure(figure=None, fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), engine=None)
for i,f in enumerate(files):
    print (f)
    df = np.loadtxt(f)    

    r = np.ones(len(df))*0.03

    color=(0.2, 0.4, 0.5)
    mlab.points3d(df[:,0], df[:,2], df[:,1], r, color=color, colormap = 'gnuplot', scale_factor=1, figure=fig)

    mlab.plot3d([0.0,0.0],[0.0, 0.0],[0.0, CUBE_MAX], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0.0,CUBE_MAX],[0.0, 0.0],[0.0, 0.0], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0.0,0.0],[0.0, CUBE_MAX/2],[0.0, 0.0], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0.0,0.0],[0.0, CUBE_MAX/2],[CUBE_MAX, CUBE_MAX], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0.0,CUBE_MAX],[0.0,0.0],[CUBE_MAX,CUBE_MAX], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([CUBE_MAX,CUBE_MAX],[0.0,CUBE_MAX/2],[CUBE_MAX,CUBE_MAX], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([CUBE_MAX,0],[CUBE_MAX/2,CUBE_MAX/2],[CUBE_MAX,CUBE_MAX], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0,0],[CUBE_MAX/2,CUBE_MAX/2],[CUBE_MAX,0], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([CUBE_MAX,CUBE_MAX],[0.0,0.0],[0.0,CUBE_MAX], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([CUBE_MAX,CUBE_MAX],[0.0,CUBE_MAX/2],[0.0,0.0], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([CUBE_MAX,0.0],[CUBE_MAX/2,CUBE_MAX/2],[0.0,0.0], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([CUBE_MAX,CUBE_MAX],[CUBE_MAX/2,CUBE_MAX/2],[0.0,CUBE_MAX], color=(0,0,0), tube_radius=None, figure=fig)

    mlab.view(azimuth=50, elevation=80, focalpoint=[1, 0.2, 1.1], distance=8.0, figure=fig)
    mlab.savefig(filename=outpath + '/out-%02d.png' % i)
    mlab.clf()

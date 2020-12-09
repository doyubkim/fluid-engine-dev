# the png files can be combined with ImageMagick,
# convert -delay 30 /tmp/simsph/*.png $HOME/Downloads/simsph.gif
import pandas as pd, os, sys
import numpy as np, glob
from mayavi import mlab

mlab.options.offscreen = True
BS = 3.0
if (os.path.exists("/tmp/simsph") == False):
    os.mkdir("/tmp/simsph")

path = sys.argv[1]
files = sorted(glob.glob(path + "/*"))

fig = mlab.figure(figure=None, fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), engine=None)
for i,f in enumerate(files):
    print (f)
    df = pd.read_csv(f,sep=' ',header=None)
    df = np.array(df)

    r = np.ones(len(df))*0.03

    color=(0.2, 0.4, 0.5)
    mlab.points3d(df[:,0], df[:,2], df[:,1], r, color=color, colormap = 'gnuplot', scale_factor=1, figure=fig)

    mlab.plot3d([0.0,0.0],[0.0, 0.0],[0.0, BS], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0.0,BS],[0.0, 0.0],[0.0, 0.0], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0.0,0.0],[0.0, BS/2],[0.0, 0.0], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0.0,0.0],[0.0, BS/2],[BS, BS], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0.0,BS],[0.0,0.0],[BS,BS], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([BS,BS],[0.0,BS/2],[BS,BS], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([BS,0],[BS/2,BS/2],[BS,BS], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0,0],[BS/2,BS/2],[BS,0], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([BS,BS],[0.0,0.0],[0.0,BS], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([BS,BS],[0.0,BS/2],[0.0,0.0], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([BS,0.0],[BS/2,BS/2],[0.0,0.0], color=(0,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([BS,BS],[BS/2,BS/2],[0.0,BS], color=(0,0,0), tube_radius=None, figure=fig)

    mlab.view(azimuth=50, elevation=80, focalpoint=[1, 0.2, 1.1], distance=8.0, figure=fig)
    mlab.savefig(filename='/tmp/simsph/out-%02d.png' % i)
    mlab.clf()

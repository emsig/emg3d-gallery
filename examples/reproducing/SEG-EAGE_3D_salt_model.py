r"""
1. SEG-EAGE 3D Salt Model
=========================


In this example we reproduce the results by [Muld07]_, which uses the SEG/EAGE
salt model from [AmBK97]_.

Velocity to resistivity transform
---------------------------------

Quoting here the description of the velocity-to-resistivity transform used by
[Muld07]_:

    "The SEG/EAGE salt model (Aminzadeh et al. 1997), originally designed for
    seismic simulations, served as a template for a realistic subsurface model.
    Its dimensions are 13500 by 13480 by 4680 m. The seismic velocities of the
    model were replaced by resistivity values. The water velocity of 1.5 km/s
    was replaced by a resistivity of 0.3 Ohm m. Velocities above 4 km/s,
    indicative of salt, were replaced by 30 Ohm m. Basement, beyond 3660 m
    depth, was set to 0.002 Ohm m. The resistivity of the sediments was
    determined by :math:`(v/1700)^{3.88}` Ohm m, with the velocity v in m/s
    (Meju et al. 2003). For air, the resistivity was set to :math:`10^8` Ohm
    m."

Equation 1 of [MeGM03]_, is given by

.. math::
    :label: meju

    \log_{10}\rho = m \log_{10}V_P + c \ ,

where :math:`\rho` is resistivity, :math:`V_P` is P-wave velocity, and for
:math:`m` and :math:`c` 3.88 and -11 were used, respectively.

The velocity-to-resistivity transform uses therefore a Faust model ([Faus53]_)
with some additional constraints for water, salt, and basement.

"""
import emg3d
import joblib
import zipfile
import pyvista
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Velocity-to-resistivity transform
# ---------------------------------
#
# The following cell loads the resistivity model ``res-model.lzma`` (~14 MB),
# if it already exists in ``../data/SEG/``, or alternatively loads the velocity
# model ``Saltf@@``, carries out the velocity-to-resistivity transform, and
# stores the resistivity model.
#
# You can get the data from the `SEG-website
# <https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models>`_ or via this
# `direct link
# <https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/Salt_Model_3D.tar.gz>`_.
# The zip-file is 513.1 MB big. Unzip the archive, and place the file
# ``Salt_Model_3D/3-D_Salt_Model/VEL_GRIDS/SALTF.ZIP`` (20.0 MB) into
# ``../data/SEG/`` (or adjust the path in the following cell).

path = '../data/SEG/'
try:
    # Get resistivities if we already computed them
    res = joblib.load(path+'res-model.lzma')

    # Get dimension
    nx, ny, nz = res.shape

except FileNotFoundError:  # THE ORIGINAL DATA ARE REQUIRED!

    # Dimensions
    nx, ny, nz = 676, 676, 210

    # Extract Saltf@@ from SALTF.ZIP
    zipfile.ZipFile(path+'SALTF.ZIP', 'r').extract('Saltf@@', path=path)

    # Load data
    with open(path+'Saltf@@', 'r') as file:
        v = np.fromfile(file, dtype=np.dtype('float32').newbyteorder('>'))
        v = v.reshape(nx, ny, nz, order='F')

    # Velocity to resistivity transform for whole cube
    res = (v/1700)**3.88  # Sediment resistivity = 1

    # Overwrite basement resistivity from 3660 m onwards
    res[:, :, np.arange(nz)*20 > 3660] = 500.  # Resistivity of basement

    # Set sea-water to 0.3
    res[:, :, :15][v[:, :, :15] <= 1500] = 0.3

    # Fix salt resistivity
    res[v == 4482] = 30.

    # Save it in compressed form
    # THE SEG/EAGE salt-model uses positive z downwards; discretize positive
    # upwards. Hence:
    # => for res, use np.flip(res, 2) to flip the z-direction
    res = np.flip(res, 2)

    # Very fast, but not so effective (118.6 MB).
    # joblib.dump(res, './res-model', compress=True)

    # lzma: very slow, but very effective (~ 18.6 MB).
    joblib.dump(res, path+'res-model.lzma')

# Create a discretize-mesh
mesh = emg3d.TensorMesh(
        [np.ones(nx)*20., np.ones(ny)*20., np.ones(nz)*20.], x0='00N')
models = {'res': np.log10(res.ravel('F'))}

# Limit colour-range
# We're cutting here the colour-spectrum at 50 Omega.m (affects only
# the basement) to have a better resolution in the sediments.
clim = np.log10([np.nanmin(res), 50])

mesh

###############################################################################
# 3D-slicer
# ---------

mesh.plot_3d_slicer(models['res'], zslice=-2000, clim=clim)

###############################################################################
# PyVista plot
# ------------
#
# Create an interactive 3D render of the data.

dataset = mesh.toVTK(models)

# Create the rendering scene and add a grid axes
p = pyvista.Plotter(notebook=True)
p.show_grid(location='outer')

dparams = {'rng': clim, 'cmap': 'viridis', 'show_edges': False}
# Add spatially referenced data to the scene
xyz = (5000, 6000, -3200)
p.add_mesh(dataset.slice('x', xyz), name='x-slice', **dparams)
p.add_mesh(dataset.slice('y', xyz), name='y-slice', **dparams)
p.add_mesh(dataset.slice('z', xyz), name='z-slice', **dparams)

# Get the salt body
p.add_mesh(dataset.threshold([1.47, 1.48]), name='vol', **dparams)

# Show the scene!
p.camera_position = [(27000, 37000, 5800), (6600, 6600, -3300), (0, 0, 1)]
p.show()

###############################################################################
# Forward modelling
# -----------------
#
# Survey parameters
# `````````````````

src = [6400, 6600, 6500, 6500, -50, -50]  # source location
freq = 1.0                                # Frequency

###############################################################################
# Initialize computation mesh
# ```````````````````````````

# Get computation domain as a function of frequency (resp., skin depth)
hx_min, xdomain = emg3d.meshes.get_domain(
        x0=6500, freq=freq, limits=[0, 13500], min_width=[5, 100])
hz_min, zdomain = emg3d.meshes.get_domain(
        freq=freq, limits=[-4180, 0], min_width=[5, 20], fact_pos=40)

# Create stretched grid
nx = 2**7
hx = emg3d.meshes.get_stretched_h(hx_min, xdomain, nx, 6500)
hy = emg3d.meshes.get_stretched_h(hx_min, xdomain, nx, 6500)
hz = emg3d.meshes.get_stretched_h(hz_min, zdomain, nx, x0=-100, x1=0)
grid = emg3d.TensorMesh(
        [hx, hy, hz], x0=(xdomain[0], xdomain[0], zdomain[0]))
grid

###############################################################################
# Put the salt model onto the modelling mesh
# ``````````````````````````````````````````

# Interpolate resistivities from fine mesh to coarser grid
cres = emg3d.maps.grid2grid(mesh, res, grid, 'volume')

# Create model
model = emg3d.Model(grid, property_x=cres, mapping='Resistivity')

# Set air resistivity
iz = np.argmin(np.abs(grid.vectorNz))
model.property_x[:, :, iz:] = 1e8

# Ensure at least top layer is water
model.property_x[:, :, iz] = 0.3

cmodels = {'res': np.log10(model.property_x.ravel('F'))}

grid.plot_3d_slicer(
        cmodels['res'], zslice=-2000, zlim=(-4180, 500),
        clim=np.log10([np.nanmin(model.property_x), 50]))

###############################################################################
# Solve the system
# ````````````````

# Source field
sfield = emg3d.get_source_field(grid, src, freq, 0)

pfield = emg3d.solve(
    grid, model, sfield,
    sslsolver=True,
    semicoarsening=False,
    linerelaxation=False,
    verb=3)

###############################################################################

grid.plot_3d_slicer(
    pfield.fx.ravel('F'), zslice=-2000, zlim=(-4180, 500),
    view='abs', v_type='Ex',
    clim=[1e-16, 1e-9], pcolor_opts={'norm': LogNorm()})

###############################################################################

# Interpolate for a "more detailed" image
x = grid.vectorCCx
y = grid.vectorCCy
rx = np.repeat([x, ], np.size(x), axis=0)
ry = rx.transpose()
rz = -2000
data = emg3d.get_receiver(grid, pfield.fx, (rx, ry, rz))

# Colour limits
vmin, vmax = -16, -10.5

# Create a figure
fig, axs = plt.subplots(figsize=(8, 5), nrows=1, ncols=2)
axs = axs.ravel()
plt.subplots_adjust(hspace=0.3, wspace=0.3)

titles = [r'|Real|', r'|Imaginary|']
dat = [np.log10(np.abs(data.real)), np.log10(np.abs(data.imag))]

for i in range(2):
    plt.sca(axs[i])
    axs[i].set_title(titles[i])
    axs[i].set_xlim(min(x)/1000, max(x)/1000)
    axs[i].set_ylim(min(x)/1000, max(x)/1000)
    axs[i].axis('equal')
    cs = axs[i].pcolormesh(x/1000, x/1000, dat[i], vmin=vmin, vmax=vmax,
                           linewidth=0, rasterized=True,)
    plt.xlabel('Inline Offset (km)')
    plt.ylabel('Crossline Offset (km)')

# Colorbar
# fig.colorbar(cf0, ax=axs[0], label=r'$\log_{10}$ Amplitude (V/m)')

# Plot colorbar
cax, kw = plt.matplotlib.colorbar.make_axes(
        axs, location='bottom', fraction=.05, pad=0.2, aspect=30)
cb = plt.colorbar(cs, cax=cax, label=r"$\log_{10}$ Amplitude (V/m)", **kw)

# Title
fig.suptitle(f"SEG/EAGE Salt Model, depth = {rz/1e3} km.", y=1, fontsize=16)

plt.show()

###############################################################################

emg3d.Report(pyvista)

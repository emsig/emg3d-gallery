"""
4. Gradient of the misfit function
==================================

A basic example how to use the new :func:`emg3d.optimize.gradient` routine to
compute the adjoint-state gradient of the misfit function.

Here we just show the usage. The implementation currently follows [PlMu08]_;
you can find the maths in the description of the functions, namely

- :func:`emg3d.optimize.misfit`;
- :func:`emg3d.optimize.gradient`; and
- :func:`emg3d.optimize.data_weighting`.

For this example we use the survey and data as obtained in the example
:ref:`sphx_glr_gallery_tutorials_simulation.py`.

"""
import os
import emg3d
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# Load survey and data
# --------------------
#
# First we load the survey and accompanying data as obtained in the example
# :ref:`sphx_glr_gallery_tutorials_simulation.py`.

mname = 'GemPy-II-survey-A'
fname = mname+'.h5'
if not os.path.isfile(fname):
    url = ('https://github.com/empymod/emg3d-gallery/blob/master/examples/'
           'data/surveys/GemPy-II-survey-A.h5?raw=true')
    with open(fname, 'wb') as f:
        t = requests.get(url)
        f.write(t.content)

survey = emg3d.load(fname)['survey']

# Let's have a look
survey


###############################################################################
#
# We can see that the survey consists of three sources, 45 receivers, and two
# frequencies.
#
# Create an initial model
# -----------------------
#
# To create an initial model we load the true model, but set all subsurface
# resistivities to 1 Ohm.m. So we are left with a homogeneous three-layer model
# air-seawater-subsurface, which includes the topography of the seafloor.

# Load true model
mname = 'GemPy-II'
fname = mname+'.h5'
if not os.path.isfile(fname):
    url = ('https://github.com/empymod/emg3d-gallery/blob/master/examples/'
           'data/models/GemPy-II.h5?raw=true')
    with open(fname, 'wb') as f:
        t = requests.get(url)
        f.write(t.content)

data = emg3d.load(fname)
model, mesh = data['model'], data['mesh']


###############################################################################

# Overwrite all subsurface resistivity values with 1.0
res = model.property_x
subsurface = (res > 0.5) & (res < 1000)
res[subsurface] = 1.0
model.property_x = res

# QC the initial model and the survey.
mesh.plot_3d_slicer(model.property_x, clim=[0.3, 200],
                    xslice=12000, yslice=7000,
                    pcolor_opts={'norm': LogNorm()})

# Plot survey in figure above
fig = plt.gcf()
fig.suptitle('Initial resistivity model (Ohm.m)')
axs = fig.get_children()
axs[1].plot(survey.rec_coords[0], survey.rec_coords[1], 'bv')
axs[2].plot(survey.rec_coords[0], survey.rec_coords[2], 'bv')
axs[3].plot(survey.rec_coords[2], survey.rec_coords[1], 'bv')
axs[1].plot(survey.src_coords[0], survey.src_coords[1], 'r*')
axs[2].plot(survey.src_coords[0], survey.src_coords[2], 'r*')
axs[3].plot(survey.src_coords[2], survey.src_coords[1], 'r*')
plt.show()


###############################################################################
# Create computational mesh
# -------------------------
#
# In the not-so-distant future the simulation class will have some automatic,
# source- and frequency-dependent gridding included. But currently we still
# have to define the computational grid manually and provide it. You can define
# it yourself or make use of some of the helper routines.

gridinput = {'freq': 1.0, 'verb': 0}

# Get cell widths and origin in each direction
xx, x0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 10], fixed=survey.src_coords[0][1], min_width=100,
    domain=[survey.rec_coords[0].min()-100, survey.rec_coords[0].max()+100],
    **gridinput)
yy, y0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 10], fixed=survey.src_coords[1][1], min_width=100,
    domain=[survey.rec_coords[1].min()-100, survey.rec_coords[1].max()+100],
    **gridinput)
zz, z0 = emg3d.meshes.get_hx_h0(
    res=[0.3, 1., 0.3], domain=[-5500, -2000], min_width=50,
    alpha=[1.05, 1.5, 0.01], fixed=[-2200, -2400, -2000], **gridinput)

# Initiate mesh.
comp_grid = emg3d.TensorMesh([xx, yy, zz], x0=np.array([x0, y0, z0]))
comp_grid


###############################################################################
# Create the Simulation
# ---------------------

data_weight_opts = {
    'gamma_d': 0.5,    # Offset weighting
    'beta_d': 1.0,     # Data weighting
    'beta_f': 0.25,    # Frequency weighting
    'min_off': 1000,   # Minimum offset
    'noise_floor': 0,  # We use all data in this example
}

simulation = emg3d.simulations.Simulation(
    name="Initial Model",    # A name for this simulation
    survey=survey,           # Our survey instance
    grid=mesh,               # The model mesh
    model=model,             # The model
    gridding=comp_grid,      # The computational mesh
    max_workers=4,           # How many parallel jobs
    # solver_opts=...,       # Any parameter to pass to emg3d.solve
    data_weight_opts=data_weight_opts,  # Data weighting options
)

# Let's QC our Simulation instance
simulation


###############################################################################
# Reference Model
# ---------------
#
# We can define a reference model. This is what is used for the data weighting.
# Usually, this would be the initial model. If no reference model is found, it
# will fall back to the observed data.
#
# The chosen reference model can be set in the parameters of the data weighting
# via ``reference``.

simulation.compute(reference=True)


###############################################################################
# Compute Gradient
# ----------------

grad = simulation.gradient


###############################################################################
# QC Gradient
# '''''''''''

# Set the gradient of air and water to NaN.
# This will eventually move directly into emgd3 (active and inactive cells).
grad[~subsurface] = np.nan


# Plot the gradient
mesh.plot_3d_slicer(
        grad.ravel('F'), clim=[-1e-11, 1e-11],
        xslice=12000, yslice=7000, zslice=-4000,
        pcolorOpts={'cmap': 'RdBu_r',
                    'norm': SymLogNorm(linthresh=1e-18, base=10)}
        )

# Add survey
fig = plt.gcf()
fig.suptitle('Gradient of the misfit function')
axs = fig.get_children()
axs[1].plot(survey.rec_coords[0], survey.rec_coords[1], 'bv')
axs[2].plot(survey.rec_coords[0], survey.rec_coords[2], 'bv')
axs[3].plot(survey.rec_coords[2], survey.rec_coords[1], 'bv')
axs[1].plot(survey.src_coords[0], survey.src_coords[1], 'r*')
axs[2].plot(survey.src_coords[0], survey.src_coords[2], 'r*')
axs[3].plot(survey.src_coords[2], survey.src_coords[1], 'r*')
plt.show()


###############################################################################

emg3d.Report()
